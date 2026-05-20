import os
import json
import torch
import pandas as pd
from datetime import datetime

def make_run_dir(config_idx, config, base='runs'):
    """
    Creates a timestamped run directory with subdirectories for figures,
    results, and checkpoints. Writes the config to config.json.
 
    Parameters
    config_idx : integer index of the config in the configs list
    config     : training config dict
    base       : root directory for all runs
 
    Returns
    run_dir : path string to the created directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = (
        f"r{config['comm_rounds']}"
        f"_e{config['local_epochs']}"
        f"_f{config['finetune_epochs']}"
        f"_g{config['gossip_every']}"
        f"_cfg{config_idx}"
        f"_{timestamp}"
    )
    run_dir = os.path.join(base, run_name)
    for sub in ['figs/loss', 'figs/uncertainty', 'figs/roc', 
                'figs/performance', 'figs/gossip',
                'results', 'checkpoints']:
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)

    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump({'config_idx': config_idx, **config}, f, indent=2)

    print(f"Run dir created: {run_dir}")
    return run_dir

def save_checkpoint(run_dir, round_num, agents, bce_history, 
                    kl_history, loss_history, gossip_meta_clusters=None):
    """
    Saves agent model states and loss histories to a .pt checkpoint file.
 
    Parameters
    run_dir              : path to the current run directory
    round_num            : current communication round number
    agents               : agent dict
    bce_history          : list of per-round BCE losses
    kl_history           : list of per-round KL losses
    loss_history         : list of per-round total losses
    gossip_meta_clusters : optional dict of CFL cluster assignments at this round
    """
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)  
    path = os.path.join(checkpoint_dir, f"round_{round_num}.pt")    
    ckpt = {
        'round': round_num,
        'agent_states': {
            nid: agent['global_model_state']
            for nid, agent in agents.items()
        },
        'bce_history': bce_history,
        'kl_history': kl_history,
        'loss_history': loss_history,
        'gossip_meta_clusters': gossip_meta_clusters,  # Save cluster assignments.
    }
    path = os.path.join(run_dir, 'checkpoints', f'round_{round_num}.pt')
    torch.save(ckpt, path)
    print(f"  Checkpoint saved: round_{round_num}.pt")

def log_run_registry(run_dir, config, final_loss, final_bce, 
                     final_kl, n_meta_clusters=None):
    """
    Appends a summary row for this run to the shared registry CSV.
 
    Parameters
    run_dir          : path to the current run directory
    config           : training config dict
    final_loss       : total loss at the last round
    final_bce        : BCE loss at the last round
    final_kl         : KL loss at the last round
    n_meta_clusters  : number of CFL meta-clusters at termination (optional)
    """
    registry_path = os.path.join('runs', 'registry.csv')
    row = {
        'run_dir': run_dir,
        'timestamp': datetime.now().isoformat(),
        'comm_rounds': config['comm_rounds'],
        'local_epochs': config['local_epochs'],
        'finetune_epochs': config['finetune_epochs'],
        'gossip_every': config['gossip_every'],
        'final_loss': round(final_loss, 6),
        'final_bce': round(final_bce, 6),
        'final_kl': round(final_kl, 6),
        'n_meta_clusters': n_meta_clusters,
    }
    df_row = pd.DataFrame([row])
    if os.path.exists(registry_path):
        df_row.to_csv(registry_path, mode='a', header=False, index=False)
    else:
        os.makedirs('runs', exist_ok=True)
        df_row.to_csv(registry_path, index=False)
    print(f"Registry updated: {registry_path}")