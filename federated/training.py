import numpy as np
import torch
from pipeline.bayesModel import PFASLocal, train_global, create_personalized_tail, reptile_adapt_fin, train_local, finetune_global
from federated.gossip import clusteredFedLearning, compute_similarity_matrix, check_divergence, gossipUpdate
from utils.logging import save_checkpoint
from federated.topology import intra_p2p_topology
from config import TRAIN_CONFIG, MOLECULAR_DESCRIPTORS
from pipeline.features import analyte_meta_tensor
from utils.metrics import update_adaptive_alpha
# The global, train, etc features will be passed in through the function calls
MIN_WARMUP_ROUNDS = 5

def get_gradient_vector(model):
    """
    Captures gradients only from the hypernetwork to detect cluster divergence.
    Fills zeros for parameters without gradients to keep vector length consistent.
 
    Parameters
    model : trained PFASGlobal model with 'hypernet_scaled' submodule
    """
    grads = []
    for name, param in model.named_parameters():
        if "hypernet_scaled" in name:
            if param.grad is not None:
                grads.append(param.grad.view(-1))
            else:
                grads.append(torch.zeros_like(param).view(-1))
    return torch.cat(grads) if grads else None

def compute_gradient_similarity(gv1, gv2):
    """Returns cosine similarity between two gradient vectors."""

    return torch.nn.functional.cosine_similarity(gv1, gv2, dim=0).item()

def get_nearest_state(agent, agents):
    """
    Returns the global model state from the nearest hub neighbor.
    Falls back to the first available neighbor if no hub is found.
 
    Parameters
    agent  : single agent dict with 'neighbors' and 'is_hub' keys
    agents : full agent dict
    """
    neighbor_ids = agent['neighbors']
    for nb_id in neighbor_ids:
        nb_agent = agents.get(nb_id)
        if nb_agent and nb_agent['is_hub']==True:
            return nb_agent['global_model_state'] 
    # No hubs found - fall back to first neighbor.
    if neighbor_ids:
        first_id = neighbor_ids[0]
        return agents[first_id]['global_model_state']  
    
def check_intra_cluster_divergence(agents, threshold=0.8):
    """
    Identifies clusters whose internal weight similarity has dropped below threshold.
    Tier 3 nodes are excluded from the check.
 
    Parameters
    agents    : agent dict
    threshold : cosine similarity below which a cluster is considered diverging
 
    Returns
    diverging_clusters : list of cluster_ids with avg internal similarity < threshold
    """
    diverging_clusters = []
    
    # Group agents by their assigned cluster_id.
    clusters = {}
    for nid, data in agents.items():
        if data.get('node_tier')==3:
            continue

        cid = data['train_df']['cluster_id'].iloc[0] 
        if cid not in clusters: clusters[cid] = []
        clusters[cid].append(nid)

    for cid, node_ids in clusters.items():
        if len(node_ids) < 2: continue
        
        # Calculate mean pairwise similarity within the cluster.
        sims = []
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                # Flatten weights to vectors
                w1 = torch.cat([p.flatten() for p in agents[node_ids[i]]['model'].parameters()])
                w2 = torch.cat([p.flatten() for p in agents[node_ids[j]]['model'].parameters()])
                
                sim = torch.nn.functional.cosine_similarity(w1, w2, dim=0)
                sims.append(sim.item())
        
        avg_internal_sim = np.mean(sims)
        
        # If the cluster is drifting apart internally, it's diverging.
        if avg_internal_sim < threshold:
            diverging_clusters.append(cid)
            
    return diverging_clusters

def run_federated_rounds(agents, G, config, N_global, run_dir, GLOBAL_CHAIN_FIN, LOCAL_CHAIN_FIN, GLOBAL_FEATS_FIN, LOCAL_FEATS_FIN, adaptive_alpha):
    """
    Primary federated learning loop. Runs local training, gossip aggregation,
    and adaptively switches from warmup FedAvg to clustered federated learning
    when hub weights diverge or gradient conflict is detected.
 
    Parameters
    agents           : agent dict with models, loaders, and metadata
    G                : physical topology graph
    config           : training config dict (comm_rounds, local_epochs, etc.)
    N_global         : total number of training samples across all agents
    run_dir          : path to save checkpoints
    GLOBAL_CHAIN_FIN : list of global target analyte columns
    LOCAL_CHAIN_FIN  : list of local target analyte columns
    GLOBAL_FEATS_FIN : list of global feature columns
    LOCAL_FEATS_FIN  : list of local feature columns
    adaptive_alpha   : per-analyte loss weighting tensor
 
    Returns
    dict with keys: 'bce', 'kl', 'total', 'gossip_logs', 'final_meta_clusters', 'network_divergence'
    """

    comm_rounds  = config['comm_rounds']
    local_epochs = config['local_epochs']
    gossip_every = config['gossip_every']

    bce_history  = []
    kl_history   = []
    loss_history = []
    gossip_logs  = []
    network_divergence = []
    
    meta_clusters = None

    # Tracking cfl phase.
    cfl_active = False
    cluster_map = {}

    for r in range(comm_rounds):
        print(f"\n--- Round {r+1}/{comm_rounds} | Phase: {'CFL' if cfl_active else 'Warmup'} ---")
        round_bce, round_kl, round_loss = [], [], []
        grad_vectors = {}
        # Local Global Head Training for each agent.
        for node_id, agent in agents.items():
            # Broadcast current aggregated state into model before training
            agent['model'].load_state_dict(agent['global_model_state']) # Global model states are instantiated with random seed; Model updates are sent to global model states.
            
            # Skip nodes with all nan global chains.
            has_any_labels = agent['train_df'][[c for c in GLOBAL_CHAIN_FIN]].notna().any().any() if GLOBAL_CHAIN_FIN else False

            if not has_any_labels:
                # Flag this and write to the "Flag" file
                # with open("{run_dir}/flag.txt", "a") as f:
                #     f.write(f"Round {r} Node {node_id}: no global chain labels, skipping training")
                continue

            # Inside the local training loop, pass kl_beta.
            new_state, history = train_global(
                agent['model'],
                agent['train_loader'],
                agent['optimizer'],
                epochs=local_epochs,
                kl_beta=TRAIN_CONFIG['kl_beta'], 
                N_global=N_global,
                adaptive_alpha=adaptive_alpha
            )

            # Capture Gradients right after training.
            gv = get_gradient_vector(agent['model'])
            if gv is not None:
                grad_vectors[node_id] = gv.clone().detach()
            
            # Write trained weights back to state dict.
            agent['global_model_state'] = new_state
            agent['last_round_loss'] = history['total'][-1]
            # Per-node loss history drives adaptive momentum in gossip.
            agent.setdefault('loss_history', []).append(history['total'][-1])

            round_bce.append(history['data'][-1])
            round_kl.append(history['kl'][-1])
            round_loss.append(history['total'][-1])

        # Adaptive Personalization every gossip_every rounds,
        if (r%gossip_every)==0:
            run_local_personalization(agents, LOCAL_CHAIN_FIN, [*LOCAL_CHAIN_FIN, *GLOBAL_CHAIN_FIN], LOCAL_FEATS_FIN, config, MOLECULAR_DESCRIPTORS=MOLECULAR_DESCRIPTORS,
                                      adaptive_alpha=adaptive_alpha)
        

        # Sync .model with updated state dict after all gossip is done
        for node_id, agent in agents.items():
            agent['model'].load_state_dict(agent['global_model_state'])

        # 2. INSERT FINETUNE STEP HERE (The "Global Head Adaptation")
        # This takes the consensus weights from gossip and aligns them with local x_lc
        print(f"  [Fine-tuning] Adapting global heads to local environmental inputs...")
        new_state = finetune_global(agents, config, GLOBAL_CHAIN_FIN, adaptive_alpha)

        # 3. Update the state dicts again so the fine-tuned weights 
        # are saved back for the next communication round.
        for node_id, agent in agents.items():
            agent['global_model_state'] = new_state

        # Sync .model with updated state dict after all gossip is done.
        for node_id, agent in agents.items():
            agent['model'].load_state_dict(agent['global_model_state'])

        # Compute similarity on global states.
        network_sim_matrix, node_ids = compute_similarity_matrix(agents)
        diverged = check_divergence(network_sim_matrix, agents.keys(), agents)
        
        # Calculate gradient conflicts.
        hub_ids = [nid for nid, a in agents.items() if a['node_tier'] == 1]
        grad_sims = []

        # Inside run_federated_rounds in training.py
        for i in range(len(hub_ids)):
            for j in range(i + 1, len(hub_ids)):
                v1 = grad_vectors[hub_ids[i]]
                v2 = grad_vectors[hub_ids[j]]
                if v1.shape != v2.shape:
                    print(f"Mismatch: Node {hub_ids[i]} size {v1.shape} vs Node {hub_ids[j]} size {v2.shape}")
                    assert IndexError, "MISMATCHED NODE"

                grad_sims.append(compute_gradient_similarity(v1, v2))

        for i in range(len(hub_ids)):
            for j in range(i+1, len(hub_ids)):
                if hub_ids[i] in grad_vectors and hub_ids[j] in grad_vectors:
                    grad_sims.append(compute_gradient_similarity(grad_vectors[hub_ids[i]], grad_vectors[hub_ids[j]]))
        
        avg_grad_sim = np.mean(grad_sims) if grad_sims else 1.0
        print(f"  [Diagnostics] Weight Divergence: {diverged} | Hub Grad Similarity: {avg_grad_sim:.4f}")

        # Switch to CFL if weights differ OR gradients are fighting (< 0.1).
        if (diverged or avg_grad_sim < 0.1) and not cfl_active and r >= MIN_WARMUP_ROUNDS:
            print(f"  *** CONFLICT DETECTED — switching to CFL permanently ***")
            cfl_active = True

        # Warmup Gossips.
        if not cfl_active:
            if r>0 and r%gossip_every == 0:
                gossipUpdate(agents, G)
                gossip_logs.append({'round': r, 'type': 'warmup_global'})
                print(f"  [Phase 1] Global gossip — round {r+1}")
        
        # CFL.
        else: 
            # Recompute clusters at CFL entry and then every gossip_every rounds (bc gossiping can change model weights and optimal cluster groups).
            
            if cluster_map == {} or r%gossip_every == 0:
                # Check the intra cluster divergence.
                check_intra_cluster_divergence(agents)
                cluster_map = clusteredFedLearning(agents)
                meta_clusters = cluster_map
                print(f"  [Phase 2] Recomputed {len(cluster_map)} CFL clusters")

            for cluster_id, node_ids in cluster_map.items():
                if len(node_ids) <= 1:
                    continue
                tiers = [agents[nid]['node_tier'] for nid in node_ids]
                if sum(1 for t in tiers if t == 3) / len(tiers) > 0.95:
                    continue  # Skip all-anomaly clusters.
                
                # Build intra-cluster graph and gossip within it.
                subset = {nid: agents[nid] for nid in node_ids if nid in agents}
                intra_G = intra_p2p_topology(subset)
                gossipUpdate(subset, intra_G)  # Momentum ON in gossip communication.

            gossip_logs.append({
                'round': r,
                'type': 'cfl',
                'n_clusters': len(cluster_map),
                'cluster_sizes': {k: len(v) for k, v in cluster_map.items()}
            })
        # Bookkeeping.
        bce_history.append(np.mean(round_bce) if round_bce else float('nan'))
        kl_history.append(np.mean(round_kl))
        loss_history.append(np.mean(round_loss) if round_loss else float('nan'))

        print(f"  Avg Loss: {loss_history[-1]:.4f} "
              f"(BCE: {bce_history[-1]:.4f}, KL: {kl_history[-1]:.6f})")
        
        adaptive_alpha = update_adaptive_alpha(agents, GLOBAL_CHAIN_FIN, adaptive_alpha)
        # Checkpoint every 5 rounds.
        if (r + 1) % 5 == 0:
            save_checkpoint(
                run_dir, r + 1, agents,
                bce_history, kl_history, loss_history,
                gossip_meta_clusters=meta_clusters
            )

    return {
        'bce': bce_history,
        'kl': kl_history,
        'total': loss_history,
        'gossip_logs': gossip_logs,
        'final_meta_clusters': meta_clusters,
        'network_divergence': network_divergence
    }

def run_local_personalization(agents, LOCAL_CHAIN_FIN, TOTAL_CHAIN, LOCAL_FEATS_FIN, config, MOLECULAR_DESCRIPTORS, adaptive_alpha):
    """
    Builds and trains a personalized local tail for each agent.
    Tier 1 nodes use standard finetuning; tier 2 nodes use Reptile meta-adaptation.
    Agents with no qualifying local chain fall back to their global model state.
 
    Parameters
    agents            : agent dict
    LOCAL_CHAIN_FIN   : local analyte columns available across the dataset
    TOTAL_CHAIN       : combined global + local chain for analyte metadata lookup
    LOCAL_FEATS_FIN   : local feature columns
    config            : training config dict
    MOLECULAR_DESCRIPTORS : analyte metadata for the local tail
    adaptive_alpha    : per-analyte loss weighting tensor
    """

    for node_id, agent in agents.items(): # Per agent personalization.
        # Creating personalized local tail.
        local_chain = create_personalized_tail( 
            agent['train_df'], LOCAL_CHAIN_FIN,
            thres_ratio=TRAIN_CONFIG['local_thres_ratio'],
            min_detections=TRAIN_CONFIG['local_min_detections']
        )
        if local_chain: # Local chain exists.
            local_meta = analyte_meta_tensor(TOTAL_CHAIN, MOLECULAR_DESCRIPTORS)
            local_model = PFASLocal(
                global_model=agent['model'],
                local_chain=local_chain,
                input_dim_loc=len(LOCAL_FEATS_FIN) * 2,
                molecular_descriptors=local_meta
            )
            if agent['node_tier']==1: # Tier 1 nodes trained by adaptive finetuning.
                train_local(local_model, agent['train_loader'],
                            local_chain, epochs=config['finetune_epochs'])
            elif agent['node_tier']==2: # Tier 2 nodes trained by reptile meta learning. 
                reptile_adapt_fin(local_model, agent['train_loader'], local_chain)
            
            agent['personalized_model'] = local_model # Local model predicts on the sparser analytes.
            agent['final_model_state']  = local_model.state_dict() # Aggregated model state for the global head and the finetuned tail.
            agent['local_chain_names']  = local_chain
        else:
            agent['final_model_state'] = agent['global_model_state'] # No personalization.
            agent['local_chain_names'] = []

        # Re-enable gradients on the global model so round n+1 can train it.
        agent['model'].train()
        for param in agent['model'].parameters():
            param.requires_grad_(True)
