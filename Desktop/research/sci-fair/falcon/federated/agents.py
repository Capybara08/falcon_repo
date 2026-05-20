import torch
from torch.utils.data import DataLoader
from pipeline.bayesModel import PFASGlobal, PFASDataset
from config import TRAIN_CONFIG, MOLECULAR_DESCRIPTORS

def createAgents(G, train_df, GLOBAL_FEATS, LOCAL_FEATS, GLOBAL_CHAIN, LOCAL_CHAIN, MOLECULAR_DESCRIPTORS):
    """
    Maps each graph node to an agent with its own model, optimizer, DataLoader, and metadata.
 
    Parameters
    G                     : NetworkX graph from hierarchicalTopologyGem
    train_df              : clustered training DataFrame with 'cluster_id' column
    GLOBAL_FEATS          : global feature column list (post scaling/encoding)
    LOCAL_FEATS           : local feature column list (post scaling/encoding)
    GLOBAL_CHAIN          : global target column list
    LOCAL_CHAIN           : local target column list
    MOLECULAR_DESCRIPTORS : analyte metadata tensor
    """
    print(f"Building agents with GLOBAL_CHAIN length: {len(GLOBAL_CHAIN)}") 
    agents = {}
    for node_id in G.nodes():
        node_train = train_df[train_df['cluster_id'] == node_id]
        if len(node_train) == 0:
            with open('output.txt', 'a') as f:
                f.write(f"Skipping node {node_id}: no rows in train_df (cluster_ids present: {train_df['cluster_id'].unique()})")
            continue
        train_batch_size = min(len(node_train), 32)

        model = PFASGlobal(len(GLOBAL_FEATS)*2, len(GLOBAL_CHAIN),
                           hidden_dim=TRAIN_CONFIG['hidden_dim'],
                           dropout=TRAIN_CONFIG['dropout'],
                           molecular_descriptors=MOLECULAR_DESCRIPTORS)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        agents[node_id] = {
            'model': model, # Holistic model
            'global_model_state': model.state_dict(),
            'optimizer': optimizer,
            'train_df': node_train,
            'train_loader': DataLoader(PFASDataset(node_train, GLOBAL_FEATS, LOCAL_FEATS, GLOBAL_CHAIN,
                                        LOCAL_CHAIN), batch_size=train_batch_size, shuffle=True),
            'neighbors': list(G.neighbors(node_id)),
            'data_volume': len(node_train),
            'is_hub': train_df.loc[train_df['cluster_id'] == node_id, 'node_tier'].iloc[0] == 1,  # tier 1 = hub
            'node_tier': node_train['node_tier'].iloc[0] 
        }

    print(f"createAgents: {len(agents)} agents created "
          f"({sum(1 for a in agents.values() if a['is_hub'])} hubs, "
          f"{sum(1 for a in agents.values() if not a['is_hub'])} satellites/anomalies)")
    
    tier_3_nodes = [node_id for node_id, a in agents.items() if a.get('node_tier') == 3]
    print(f"Tier 3 Nodes: {tier_3_nodes}")
    return agents