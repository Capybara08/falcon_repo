
import networkx as nx
import numpy as np

def fedAvgTopology(df):
    """
    Builds the physical communication graph for standard federated averaging.
 
    Tier structure:
    - Hubs (tier 1): fully connected backbone, bidirectional
    - Links (tier 2): connect to all hubs both ways, plus horizontal mesh to other links
    - Satellites (tier 3): receive-only, edges flow from hubs to satellites only
 
    Parameters
    df : clustered DataFrame with 'node_tier' and 'cluster_id' columns
    """
    G = nx.DiGraph() 
    hubs = df[df['node_tier'] == 1]['cluster_id'].unique().tolist()
    links = df[df['node_tier'] == 2]['cluster_id'].unique().tolist()
    satellites = df[df['node_tier'] == 3]['cluster_id'].unique().tolist()
    
    # Hub backbone - bidirectional so hubs communicate mutually.
    for i in range(len(hubs)):
        for j in range(len(hubs)):
            if i != j:
                G.add_edge(hubs[i], hubs[j], weight=1.0)

    # Link nodes - connect to all hubs both ways, plus horizontal mesh.
    for l_id in links:
        for h_id in hubs:
            G.add_edge(l_id, h_id, weight=0.8) # Link contributes to Hub.
            G.add_edge(h_id, l_id, weight=1.0) # Hub contributes to Link.

        for other_l_id in links:
            if l_id != other_l_id:
                G.add_edge(l_id, other_l_id, weight=0.5)

    # Satellites/anomalies - receiving only, no edge back to hubs.
    for s_id in satellites:
        for h_id in hubs:
            G.add_edge(h_id, s_id, weight=1.0) 

    return G

def intra_p2p_topology(cluster_agents):
    """
    Builds a fully connected intra-cluster graph for clustered federated learning.
    Clustered FL already groups environmentally similar nodes, so all members
    can communicate freely. Outbound contribution is scaled by tier.
 
    Tier weights: tier 1 = 1.0, tier 2 = 0.8, tier 3 = 0.0 (receive only).
 
    Parameters
    cluster_agents : subset of the agents dict containing only nodes in this cluster
    """
    G = nx.DiGraph() # Fully connected.
    agent_ids = list(cluster_agents.keys())

    # Map tiers to specific weights.
    tier_weights = {1: 1.0, 2: 0.8, 3: 0.0} # Tier 3 doesn't update anything; only receives updates.

    for agent_id in agent_ids:
        # Determine the weight this specific node "contributes" to others.
        tier = cluster_agents[agent_id]['node_tier']
        outbound_weight = tier_weights.get(tier, 0.5) # Default to 0.5 if tier unknown.

        for target_id in agent_ids:
            if agent_id != target_id:
                # Add a directed edge from source to target.
                G.add_edge(agent_id, target_id, weight=outbound_weight)
    return G