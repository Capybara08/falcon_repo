"""
utils/metrics.py
All evaluation logic for federated PFAS prediction.
Import these functions rather than inlining in run.py.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from pipeline.bayesModel import PFASDataset
from pipeline.bayesModel import get_bayes_pred


def evaluate_node_global(agent, pfas_idx):
    """
    Runs Bayesian inference on a single node's test batch using the global model.

    Parameters
    agent    : single agent dict with 'test_batch' and 'model' keys
    pfas_idx : index of the target analyte in the global chain

    Returns
    y_true   : np.ndarray of ground-truth labels
    g_scores : np.ndarray of predicted probabilities
    mask     : bool np.ndarray of non-NaN positions
    """
    if agent.get('test_batch') is None:
        return None, None, None

    x_gb, y_gb, x_lc, y_lc = agent['test_batch']

    # always use the base global model for global evaluation — personalized model has a different output shape
    eval_model = agent.get('model')

    if eval_model is None:
        return None, None, None

    y_true = y_gb[:, pfas_idx].numpy()

    mask = ~np.isnan(y_true)
    if mask.sum() == 0 or len(np.unique(y_true[mask])) < 2:
        return None, None, None

    g_means, _ = get_bayes_pred(eval_model, x_gb, x_lc, target_type='global')

    return y_true, g_means[:, pfas_idx], mask


def evaluate_node_personalized(agent, analyte, local_pfas_idx=None):
    """
    Runs Bayesian inference using the personalized local model tail.
    Falls back gracefully if no local model is available for this analyte.

    Parameters
    agent          : single agent dict
    analyte        : analyte column name string
    local_pfas_idx : optional override for index into y_gb instead of y_lc

    Returns
    y_true     : np.ndarray or None
    p_scores   : np.ndarray or None
    mask       : bool np.ndarray or None
    used_local : True if the personalized model was used
    """
    if agent.get('test_batch') is None:
        return None, None, None, False

    x_gb, y_gb, x_lc, y_lc = agent['test_batch']

    if 'personalized_model' not in agent:
        return None, None, None, False

    if analyte not in agent.get('local_chain_names', []):
        return None, None, None, False

    loc_idx = agent['local_chain_names'].index(analyte)
    dataset_local_chain = getattr(agent.get('test_loader').dataset, 'y_loc_chain', None)
    if dataset_local_chain is None or analyte not in dataset_local_chain:
        return None, None, None, False

    dataset_loc_idx = dataset_local_chain.index(analyte)

    y_true = y_lc[:, dataset_loc_idx].numpy() if local_pfas_idx is None else \
             y_gb[:, loc_idx].numpy()

    mask = ~np.isnan(y_true)
    if mask.sum() == 0 or len(np.unique(y_true[mask])) < 2:
        return None, None, None, False

    p_means, _ = get_bayes_pred(agent['personalized_model'], x_gb, x_loc=x_lc, debug_name=f"personalized:{analyte}", target_type='personalized')

    pred_col = p_means[:, loc_idx]
    is_valid_pred = ~np.isnan(pred_col)
    final_eval_mask = mask & is_valid_pred  # intersection where both labels and predictions are valid

    if final_eval_mask.sum() < 2 or len(np.unique(y_true[final_eval_mask])) < 2:
        print(f"  Skipping {analyte}: Not enough valid overlap between data and predictions")
        return None, None, None, False

    return y_true[final_eval_mask], pred_col[final_eval_mask], final_eval_mask[final_eval_mask], True


def evaluate_all_analytes(agents, pfas_classes, global_chain, local_chain_fin):
    """
    Evaluates AUROC and AUPRC across all agents and analytes.
    Logs prevalence independently of model type to enable random-baseline comparisons.

    Parameters
    agents          : agent dict
    pfas_classes    : dict mapping class label to list of analyte column names
    global_chain    : list of global target analyte column names
    local_chain_fin : list of local target analyte column names

    Returns
    summary_stats : dict of per-analyte AUROC lists (global and personalized)
    auprc_stats   : dict of per-analyte AUPRC lists (global and personalized)
    """
    all_analytes = [a for sublist in pfas_classes.values() for a in sublist]
    summary_stats = {a: {'global': [], 'personalized': [], 'prevalence': []} for a in all_analytes}
    auprc_stats   = {a: {'global': [], 'personalized': []} for a in all_analytes}

    for analyte in all_analytes:
        is_global = analyte in global_chain
        is_local  = analyte in local_chain_fin

        if not is_global and not is_local:
            continue

        pfas_idx = global_chain.index(analyte) if is_global else None

        for node_id, agent in agents.items():
            if agent.get('test_batch') is None:
                continue

            x_gb, y_gb, x_lc, y_lc = agent['test_batch']

            # log prevalence independently so every node contributes to the baseline
            if is_global:
                y_true_raw = y_gb[:, pfas_idx].numpy()
            elif analyte in agent.get('local_chain_names', []):
                loc_idx = agent['local_chain_names'].index(analyte)
                dataset_local_chain = getattr(agent.get('test_loader').dataset, 'y_loc_chain', [])
                if analyte in dataset_local_chain:
                    d_idx = dataset_local_chain.index(analyte)
                    y_true_raw = y_lc[:, d_idx].numpy()
                else:
                    y_true_raw = None
            else:
                y_true_raw = None

            if y_true_raw is not None:
                mask_prev = ~np.isnan(y_true_raw)
                if mask_prev.sum() > 0:
                    summary_stats[analyte]['prevalence'].append(np.mean(y_true_raw[mask_prev]))

            # global model scoring
            if is_global and pfas_idx is not None:
                y_true, g_scores, mask = evaluate_node_global(agent, pfas_idx)
                if y_true is not None:
                    valid_true = y_true[mask]
                    valid_scores = g_scores[mask]
                    if not np.isnan(valid_scores).any():
                        summary_stats[analyte]['global'].append(roc_auc_score(valid_true, valid_scores))
                        auprc_stats[analyte]['global'].append(average_precision_score(valid_true, valid_scores))

            # personalized model scoring
            y_true_p, p_scores, mask_p, used_local = evaluate_node_personalized(agent, analyte)
            if used_local and y_true_p is not None:
                valid_true_p   = y_true_p[mask_p]
                valid_scores_p = p_scores[mask_p]
                if not np.isnan(valid_scores_p).any():
                    summary_stats[analyte]['personalized'].append(roc_auc_score(valid_true_p, valid_scores_p))
                    auprc_stats[analyte]['personalized'].append(average_precision_score(valid_true_p, valid_scores_p))

    return summary_stats, auprc_stats


def diagnose_node_class_distribution(agents, auc_analytes, global_chain_fin):
    """
    Prints per-analyte breakdown of viable vs skipped nodes.
    Useful for debugging NaN or missing AUC values.

    Parameters
    agents          : agent dict
    auc_analytes    : list of analyte names to check
    global_chain_fin: list of global target analyte column names
    """
    for analyte in auc_analytes:
        if analyte not in global_chain_fin:
            print(f"  {analyte}: not in global chain, skipping.")
            continue

        pfas_idx = global_chain_fin.index(analyte)
        skipped_no_loader    = 0
        skipped_no_diversity = 0
        skipped_all_nan      = 0
        viable = 0

        print(f"\n--- {analyte} ---")
        for node_id, agent in agents.items():
            if agent.get('test_loader') is None:
                skipped_no_loader += 1
                continue

            x_gb, y_gb, x_lc, y_lc = next(iter(agent['test_loader']))
            y_true = y_gb[:, pfas_idx].numpy()
            non_nan = ~np.isnan(y_true)

            if non_nan.sum() == 0:
                skipped_all_nan += 1
                continue

            y_valid = y_true[non_nan]
            n_pos = (y_valid == 1.0).sum()
            n_neg = (y_valid == 0.0).sum()

            if len(np.unique(y_valid)) < 2:
                skipped_no_diversity += 1
                print(f"  Node {node_id}: n={len(y_valid)}, "
                      f"pos={n_pos}, neg={n_neg} — NO DIVERSITY")
            else:
                viable += 1

        print(f"  Viable: {viable} | "
              f"No loader: {skipped_no_loader} | "
              f"All NaN: {skipped_all_nan} | "
              f"No diversity: {skipped_no_diversity}")


def update_adaptive_alpha(current_metrics, num_analytes, current_alpha=None):
    """
    Adjusts per-analyte loss weights based on AUPRC performance.
    Analytes with low AUPRC get higher weight; well-performing analytes are relaxed.
    Output is clamped to the range [1.0, 10.0].

    Parameters
    current_metrics : summary dict from evaluate_all_analytes
    num_analytes    : len(GLOBAL_CHAIN)
    current_alpha   : existing alpha tensor, or None to initialize at 1.0
    """
    if current_alpha is None:
        current_alpha = torch.ones(num_analytes)

    for i in range(len(num_analytes)):
        auprc = current_metrics.get(i, {}).get('auprc', 0.5)

        if auprc < 0.4:
            current_alpha[i] += 0.2
        elif auprc > 0.7:
            current_alpha[i] -= 0.1

    return torch.clamp(current_alpha, 1.0, 10.0)


def build_summary_report(summary_stats, auprc_stats, pfas_classes,
                          config, run_dir):
    """
    Prints and saves a formatted per-class AUROC/AUPRC summary table to results/summary.txt.

    Parameters
    summary_stats : dict of per-analyte AUROC lists
    auprc_stats   : dict of per-analyte AUPRC lists
    pfas_classes  : dict mapping class label to list of analyte column names
    config        : training config dict
    run_dir       : path to the current run directory
    """
    import os
    os.makedirs(os.path.join(run_dir, 'results'), exist_ok=True)
    summary_path = os.path.join(
        run_dir, 'results',
        f"summary_r{config['comm_rounds']}"
        f"_e{config['local_epochs']}"
        f"_f{config['finetune_epochs']}.txt"
    )

    header = (
        f"\n{'='*80}\n"
        f"{'FULL FEDERATED EVALUATION SUMMARY':^80}\n"
        f"{'='*80}\n"
        f"Config: comm_rounds={config['comm_rounds']}, "
        f"local_epochs={config['local_epochs']}, "
        f"finetune_epochs={config['finetune_epochs']}, "
        f"gossip_every={config['gossip_every']}\n"
    )
    print(header)

    lines = [header]

    for cls, analytes in pfas_classes.items():
        section = f"\n--- Class {cls} ---\n"
        print(section, end='')
        lines.append(section)

        rows = []
        for analyte in analytes:
            g_aucs  = summary_stats.get(analyte, {}).get('global', [])
            p_aucs  = summary_stats.get(analyte, {}).get('personalized', [])
            g_auprc = auprc_stats.get(analyte,   {}).get('global', [])
            p_auprc = auprc_stats.get(analyte,   {}).get('personalized', [])

            if not g_aucs and not p_aucs:
                continue

            rows.append({
                'Analyte':    analyte.replace('_MCL_Status', ''),
                'G_AUROC':    f"{np.mean(g_aucs):.4f}"  if g_aucs  else '—',
                'P_AUROC':    f"{np.mean(p_aucs):.4f}"  if p_aucs  else '—',
                'AUROC_Lift': (f"{np.mean(p_aucs) - np.mean(g_aucs):+.4f}"
                               if g_aucs and p_aucs else '—'),
                'G_AUPRC':    f"{np.mean(g_auprc):.4f}" if g_auprc else '—',
                'P_AUPRC':    f"{np.mean(p_auprc):.4f}" if p_auprc else '—',
                'AUPRC_Lift': (f"{np.mean(p_auprc) - np.mean(g_auprc):+.4f}"
                               if g_auprc and p_auprc else '—'),
                'N_Nodes':    max(len(g_aucs), len(p_aucs)),
            })

        if rows:
            table = pd.DataFrame(rows).to_string(index=False)
            print(table)
            lines.append(table + '\n')
        else:
            msg = "  No viable nodes for this class.\n"
            print(msg, end='')
            lines.append(msg)

    with open(summary_path, 'w') as f:
        f.writelines(lines)
    print(f"\nSummary saved to {summary_path}")

    return summary_path


def assign_test_loaders(agents, test_df_labeled, global_feats, local_feats, global_chain, local_chain):
    """
    Assigns a DataLoader to each agent based on the test rows matching their cluster_id.

    Parameters
    agents          : agent dict
    test_df_labeled : test DataFrame with 'cluster_id' column assigned
    global_feats    : global feature column list
    local_feats     : local feature column list
    global_chain    : global target column list
    local_chain     : local target column list
    """
    for node_id, agent in agents.items():
        node_test = test_df_labeled[test_df_labeled['cluster_id'] == node_id]
        if len(node_test) > 0:
            agent['test_loader'] = DataLoader(
                PFASDataset(node_test, global_feats, local_feats,
                            global_chain, local_chain),
                batch_size=len(node_test),
                shuffle=False
            )
        else:
            agent['test_loader'] = None


def cache_test_batches(agents):
    """
    Pre-loads one batch per agent so evaluation avoids repeated next(iter(...)) calls.

    Parameters
    agents : agent dict with 'test_loader' keys
    """
    for node_id, agent in agents.items():
        if agent.get('test_loader') is not None:
            x_gb, y_gb, x_lc, y_lc = next(iter(agent['test_loader']))
            agent['test_batch'] = (x_gb, y_gb, x_lc, y_lc)
        else:
            agent['test_batch'] = None