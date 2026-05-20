"""
federated/gossip.py
Topology-constrained gossip with similarity weighting.
Falls back to plain FedAvg when weights haven't diverged yet.
"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from config import TRAIN_CONFIG, get_ohe_env_weighted
import networkx as nx
MIN_DIVERGENCE_THRESHOLD = 0.05  # Minimum divergence across weight vectors before cluster partitioning is applied.

def check_divergence(sim_matrix, node_ids, agents, threshold=MIN_DIVERGENCE_THRESHOLD):
    """
    Checks whether hub-layer weights have diverged enough to warrant clustered federated learning.
    Focuses only on Tier 1 hubs to avoid noise from sparse satellites.
 
    Parameters
    sim_matrix : pairwise cosine similarity matrix across all agents
    node_ids   : ordered list of node IDs matching matrix rows/cols
    agents     : agent dict
    threshold  : minimum average hub distance to trigger clustering
    """
    hub_indices = [i for i, nid in enumerate(node_ids) if agents[nid]['node_tier'] != 3]
    
    if len(hub_indices) < 2:
        return False # Can't diverge if there's only one hub.
        
    hub_sims = sim_matrix[np.ix_(hub_indices, hub_indices)]
    
    # Average similarity of unique off-diagonal pairs.
    avg_hub_dist = 1.0 - (np.sum(hub_sims) - len(hub_indices)) / (len(hub_indices)**2 - len(hub_indices))
    
    print(f"Network Hub Divergence: {avg_hub_dist:.4f}")
    return avg_hub_dist > threshold

def get_weight_vector(state_dict):
    """
    Flattens all model weights into a single comparable vector using sorted keys.
 
    Parameters
    state_dict : model state dict for a single agent
    """
    sorted_keys = sorted(state_dict.keys())
    return torch.cat([state_dict[k].detach().cpu().float().flatten() for k in sorted_keys])

def compute_similarity_matrix(agents):
    """
    Computes pairwise cosine similarity between all agent weight vectors.
 
    Parameters
    agents : agent dict
 
    Returns
    sim_matrix : np.ndarray [n_nodes, n_nodes]
    node_ids   : list ordered to match matrix rows/cols
    """
    node_ids = list(agents.keys())
    vecs = torch.stack([
        get_weight_vector(agents[nid]['global_model_state']) # Computes similarity on global model state
        for nid in node_ids
    ])
    norms = vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    vecs_norm = vecs / norms
    sim_matrix = (vecs_norm @ vecs_norm.T).cpu().numpy()
    return sim_matrix, node_ids

def compute_node_momentum(agent, base_momentum=0.7):
    """
    Adapts gossip momentum based on recent loss trajectory.
    Fast-converging nodes resist outside influence (high momentum).
    Plateauing nodes welcome neighbor signal (low momentum).
 
    Parameters
    agent         : single agent dict with 'loss_history' key
    base_momentum : default momentum value
    """
    history = agent.get('loss_history',[])
    if len(history) < 2:
        return base_momentum
    recent_drop = history[-2] - history[-1]
    if recent_drop > 0.05:
        return min(base_momentum + 0.15, 0.95)
    elif recent_drop < 0.005:
        return max(base_momentum - 0.2, 0.3)
    return base_momentum

def clusteredFedLearning(agents, cos_sim=0.7): 
    """
    Groups agents into dynamic federated learning clusters using agglomerative clustering.
    Clustering begins once network weight divergence exceeds MIN_DIVERGENCE_THRESHOLD.
 
    Agent fingerprints combine weighted environmental features and PCA-reduced last-layer
    model weights to capture both data distribution and learned behavior similarity.
 
    Parameters
    agents  : agent dict
    cos_sim : cosine similarity threshold for cluster membership (distance = 1 - cos_sim)
 
    Returns
    new_cluster_map : dict mapping cluster label to list of node_ids
    """
    features = None
    for n_id in agents.keys():
        features = agents[n_id]['train_loader'].dataset.df.columns.tolist()
        break
    get_ohe_env_weighted(features)
    feat_weights = TRAIN_CONFIG['env_cluster_weighted'] # grab the weighted dict
    node_ids = list(agents.keys())    
    target_feats = sorted(feat_weights.keys())
    weight_vector = np.array([feat_weights[i] for i in target_feats])
    
    all_model_weights = []
    env_vectors = []
    # Get each agent's gradient/model state.
    for n_id in node_ids:
        df = agents[n_id]['train_loader'].dataset.df
        valid_feats = [f for f in target_feats if f in df.columns]
        
        if not valid_feats:
            # Fall back to all numeric cols if no weighted feats found.
            env_vec = df.select_dtypes(include=['number']).mean().values
            env_vectors.append(env_vec[:len(weight_vector)])
        else:
            node_env_vector = df[target_feats].mean().values
            valid_weights = np.array([feat_weights[f] for f in valid_feats])
            # Apply weights to "stretch" dims.
            env_vectors.append(node_env_vector * valid_weights)
        
        # Get final classifier layer. Reduce layer with PCA to match to env vector dimensions. (32+ to 25 env dimension)
        state = agents[n_id]['global_model_state']
        last_layer_key = list(state.keys())[-4] # classifier weights
        model_weights = state[last_layer_key].flatten().cpu().numpy()     
        all_model_weights.append(model_weights)
    
    if len(all_model_weights) > min(15, len(node_ids)):
        pca = PCA(n_components=min(15, len(node_ids)))
        reduced_model_weights = pca.fit_transform(all_model_weights).tolist() # fit transform on a list; returns a list
    else: # Use raw weights.
        reduced_model_weights = all_model_weights
    
    agent_vectors = []
    # Attach the model weights to env params.
    for idx, n_id in enumerate(node_ids):
        combined_fingerprint = np.concatenate([env_vectors[idx], reduced_model_weights[idx]])
        agent_vectors.append(combined_fingerprint)
        
    scaler = StandardScaler()
    final_matrix = scaler.fit_transform(agent_vectors)
    dist_threshold = 1.0 - cos_sim
    clustering_model = AgglomerativeClustering(
        n_clusters=None, 
        metric='cosine',
        linkage='average',
        distance_threshold=dist_threshold
        )

    labels = clustering_model.fit_predict(final_matrix)

    new_cluster_map = {i: [] for i in range(len(set(labels)))}
    for idx, label in enumerate(labels):
        new_cluster_map[label].append(node_ids[idx])
    
    return new_cluster_map 

def gossipUpdate(agents, G):
    """
    Standard Federated Averaging constrained by the physical graph G.
    Uses data volume weighting to ensure robust global convergence.
    Tier 3 anomaly nodes do not contribute — they only receive updates.
 
    Parameters
    agents : agent dict
    G      : NetworkX graph defining allowed communication edges
    """
    new_states = {}

    for node_id, agent in agents.items():
        neighbors = list(G.neighbors(node_id)) if G.has_node(node_id) else []
        # Anomalies don't send communications.
        if agent.get('node_tier') == 3:
            # Tier 3 waits to pull from the nearest Hub.
            new_states[node_id] = agent['global_model_state']
            continue

        # Aggregate weights based on Data Volume.
        own_state = agent['global_model_state']
        own_vol = agent.get('data_volume', 1.0)
        
        # Start with own weighted weights.
        weighted_params = {k: v.clone() * own_vol for k, v in own_state.items()}
        total_vol = own_vol

        for nb_id in neighbors:
            if nb_id not in agents: continue
            
            # During Standard Federated averaging, we usually only aggregate with Tier 1 and 2
            if agents[nb_id].get('node_tier') == 3: continue 

            nb_state = agents[nb_id]['global_model_state']
            nb_vol = agents[nb_id].get('data_volume', 1.0)
            
            for k in weighted_params:
                weighted_params[k] += nb_state[k] * nb_vol
            total_vol += nb_vol

        new_states[node_id] = {k: v / total_vol for k, v in weighted_params.items()}

    # Update agents.
    for node_id, state in new_states.items():
        agents[node_id]['global_model_state'] = state
