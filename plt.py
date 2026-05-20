import matplotlib.pyplot as plt
import numpy as np
import os
from pipeline.bayesModel import get_bayes_pred
from sklearn.metrics import roc_curve, auc
from utils.metrics import evaluate_node_global
import seaborn as sns
from scipy.spatial import ConvexHull


def plot_pfas_uncertainty(y_actual, config_idx, save_dir, y_mean, y_std, pfas_name):
    """
    Plots Bayesian predictive uncertainty for a single analyte at a single node.

    Parameters
    y_actual   : ground-truth binary labels (0 or 1), NaNs already masked out
    config_idx : config index for filename disambiguation
    save_dir   : directory to save the figure
    y_mean     : mean predicted probability from Bayesian sampling
    y_std      : std dev from Bayesian sampling (epistemic uncertainty)
    pfas_name  : analyte name for the plot title
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    samples = np.arange(len(y_actual))

    ax.plot(samples, y_mean, label='Predicted Prob.', color='blue', lw=2)
    ax.fill_between(
        samples,
        np.clip(y_mean - 2 * y_std, 0, 1),
        np.clip(y_mean + 2 * y_std, 0, 1),
        color='blue', alpha=0.2, label='95% Confidence (Uncertainty)'
    )
    ax.scatter(samples, y_actual, color='red', s=10,
               label='Actual Detection', alpha=0.5)

    ax.set_title(f"Bayesian Predictive Uncertainty for {pfas_name}")
    ax.set_xlabel("Test Samples (Chronological)")
    ax.set_ylabel("Probability of Detection")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        os.path.join(save_dir, f"uncertainty{config_idx}.png"),
        dpi=300, bbox_inches='tight'
    )
    plt.close(fig)


def plot_best_node_uncertainty(agents, config_idx, save_dir, summary_stats, GLOBAL_CHAIN_FIN):
    """
    Finds the highest-AUC node for PFOA and plots its Bayesian uncertainty.

    Parameters
    agents          : agent dict
    config_idx      : config index for filename disambiguation
    save_dir        : directory to save the figure
    summary_stats   : per-analyte AUROC dict from evaluate_all_analytes
    GLOBAL_CHAIN_FIN: list of global target analyte column names
    """
    from utils.metrics import evaluate_node_global
    from sklearn.metrics import roc_auc_score

    if 'PFOA_MCL_Status' not in GLOBAL_CHAIN_FIN:
        print("plot_best_node_uncertainty: PFOA_MCL_Status not in chain.")
        return

    pfas_idx = GLOBAL_CHAIN_FIN.index('PFOA_MCL_Status')

    best_node, best_auc = None, 0.0
    for node_id, agent in agents.items():
        if agent.get('test_loader') is None:
            continue
        y_true, g_scores, mask = evaluate_node_global(agent, pfas_idx)
        if y_true is None:
            continue
        node_auc = roc_auc_score(y_true[mask], g_scores[mask])
        if node_auc > best_auc:
            best_auc, best_node = node_auc, node_id

    if best_node is None:
        print("plot_best_node_uncertainty: no viable node found.")
        return

    agent = agents[best_node]
    x_gb, y_gb, x_lc, y_lc = next(iter(agent['test_loader']))

    y_true = y_gb[:, pfas_idx].numpy()
    mask = ~np.isnan(y_true)

    if mask.sum() == 0:
        print("plot_best_node_uncertainty: all NaN targets for best node.")
        return

    g_means, g_stds = get_bayes_pred(agent['model'], x_gb)

    plot_pfas_uncertainty(
        y_actual=y_true[mask],
        config_idx=config_idx,
        y_mean=g_means[mask, pfas_idx],
        y_std=g_stds[mask, pfas_idx],
        pfas_name=f"PFOA — Node {best_node}",
        save_dir=save_dir
    )


def plot_roc_curve(y_true, config_idx, save_dir, y_probs, pfas_name):
    """
    Plots and saves a single ROC curve.

    Parameters
    y_true     : ground-truth binary labels
    config_idx : config index for filename disambiguation
    save_dir   : directory to save the figure
    y_probs    : predicted probabilities
    pfas_name  : analyte name for the plot title
    """
    os.makedirs(save_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--',
            label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve: {pfas_name}')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        os.path.join(save_dir, f"roc_curve{config_idx}.png"),
        dpi=300, bbox_inches='tight'
    )
    plt.close(fig)


def plot_best_roc(agents, config_idx, save_dir, summary_stats, GLOBAL_CHAIN_FIN):
    """
    Pools predictions across all nodes and plots one ROC curve per analyte
    for a fixed set of key PFAS.

    Parameters
    agents          : agent dict
    config_idx      : config index for filename disambiguation
    save_dir        : directory to save the figures
    summary_stats   : per-analyte AUROC dict from evaluate_all_analytes
    GLOBAL_CHAIN_FIN: list of global target analyte column names
    """
    for analyte in ['PFOA_MCL_Status', 'PFOS_MCL_Status', 'PFHxS_MCL_Status', 'PFHxA_MCL_Status', 'HFPO_DA_MCL_Status', 'PFNA_MCL_Status']:
        if analyte not in GLOBAL_CHAIN_FIN:
            continue

        all_true, all_probs = [], []
        pfas_idx = GLOBAL_CHAIN_FIN.index(analyte)

        for node_id, agent in agents.items():
            if agent.get('test_loader') is None:
                continue
            x_gb, y_gb, x_lc, y_lc = next(iter(agent['test_loader']))
            y_true = y_gb[:, pfas_idx].numpy()
            mask = ~np.isnan(y_true)

            if mask.sum() == 0 or len(np.unique(y_true[mask])) < 2:
                continue

            g_means, _ = get_bayes_pred(agent['model'], x_gb)
            all_true.extend(y_true[mask].tolist())
            all_probs.extend(g_means[mask, pfas_idx].tolist())

        if len(all_true) > 0 and len(np.unique(all_true)) >= 2:
            plot_roc_curve(
                np.array(all_true), config_idx,
                save_dir,
                np.array(all_probs), analyte,
            )
        else:
            print(f"plot_best_roc: skipping {analyte} — insufficient data.")


def plot_federated_loss(history, config_idx, save_dir, bce=None, kl=None):
    """
    Plots total loss over rounds, with optional BCE/KL breakdown.

    Parameters
    history    : list of total loss per round
    config_idx : config index for filename disambiguation
    save_dir   : directory to save the figure
    bce        : list of BCE loss per round (optional)
    kl         : list of KL loss per round (optional)
    """
    os.makedirs(save_dir, exist_ok=True)
    rounds = np.arange(1, len(history) + 1)
    losses = np.array(history)
    n_plots = 3 if (bce is not None and kl is not None) else 2

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    fig.suptitle("Federated Training Loss", fontsize=14, fontweight='bold')

    axes[0].plot(rounds, losses, color='steelblue', lw=2)
    axes[0].set_title("Total Loss (Raw)")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rounds, losses, color='steelblue', lw=2)
    axes[1].set_yscale('log')
    axes[1].set_title("Total Loss (Log Scale)")
    axes[1].set_xlabel("Round")
    axes[1].grid(True, alpha=0.3, which='both')

    if bce is not None and kl is not None:
        axes[2].plot(rounds, bce, color='darkorange', lw=2, label='BCE')
        axes[2].plot(rounds, kl,  color='crimson',    lw=2, label='KL')
        axes[2].set_title("BCE vs KL Breakdown")
        axes[2].set_xlabel("Round")
        axes[2].set_yscale('log')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig(
        os.path.join(save_dir, f"federated_loss{config_idx}.png"),
        dpi=300, bbox_inches='tight'
    )
    plt.close(fig)

    print(f"[Loss] Round 1→Final: {losses[0]:.4f} → {losses[-1]:.4f} "
          f"({losses[-1] - losses[0]:+.4f})")
    if bce is not None:
        print(f"[BCE]  Round 1→Final: {bce[0]:.4f} → {bce[-1]:.4f}")
        print(f"[KL]   Round 1→Final: {kl[0]:.6f} → {kl[-1]:.6f}")


def plot_gossip_clusters(gossip_logs, save_dir):
    """
    Plots four gossip diagnostics over rounds: cluster count, average cluster size,
    average pairwise weight similarity, and cluster membership stability.

    Parameters
    gossip_logs : list of gossip log dicts from run_federated_rounds
    save_dir    : directory to save the figure
    """
    if not gossip_logs:
        print("plot_gossip_clusters: no gossip logs to plot.")
        return

    os.makedirs(save_dir, exist_ok=True)

    rounds     = [g['round'] for g in gossip_logs]
    n_clusters = [g.get('n_meta_clusters', g.get('n_clusters')) for g in gossip_logs]
    avg_sizes  = [
        g.get('avg_cluster_size',
              (sum(g['cluster_sizes'].values()) / len(g['cluster_sizes'])
               if g.get('cluster_sizes') else None))
        for g in gossip_logs
    ]
    avg_sim    = [g.get('avg_similarity', None) for g in gossip_logs]

    # fraction of nodes that remained in the same cluster vs the previous round
    stability = [None]
    for i in range(1, len(gossip_logs)):
        prev_map = gossip_logs[i-1].get('cluster_map', {})
        curr_map = gossip_logs[i].get('cluster_map', {})

        if not prev_map or not curr_map:
            stability.append(None)
            continue

        prev_assignment = {nid: label for label, members in prev_map.items() for nid in members}
        curr_assignment = {nid: label for label, members in curr_map.items() for nid in members}

        shared_nodes = set(prev_assignment) & set(curr_assignment)
        if not shared_nodes:
            stability.append(None)
            continue

        same = sum(1 for nid in shared_nodes if prev_assignment[nid] == curr_assignment[nid])
        stability.append(same / len(shared_nodes))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Clustered Federated Learning — Gossip Diagnostics",
                 fontsize=13, fontweight='bold')

    axes[0, 0].plot(rounds, n_clusters, marker='o', color='steelblue', lw=2)
    axes[0, 0].set_title("Number of Meta-Clusters per Round")
    axes[0, 0].set_xlabel("Round")
    axes[0, 0].set_ylabel("N Clusters")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(rounds, avg_sizes, marker='s', color='darkorange', lw=2)
    axes[0, 1].set_title("Average Cluster Size (Nodes)")
    axes[0, 1].set_xlabel("Round")
    axes[0, 1].set_ylabel("Avg Nodes per Cluster")
    axes[0, 1].grid(True, alpha=0.3)

    if any(v is not None for v in avg_sim):
        axes[1, 0].plot(rounds, avg_sim, marker='^', color='mediumseagreen', lw=2)
        axes[1, 0].axhline(y=0.98, color='red', linestyle='--', alpha=0.5,
                           label='Near-identical (>0.98)')
        axes[1, 0].set_title("Avg Pairwise Weight Similarity")
        axes[1, 0].set_xlabel("Round")
        axes[1, 0].set_ylabel("Cosine Similarity")
        axes[1, 0].set_ylim(0, 1.05)
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].set_visible(False)

    valid_rounds = [r for r, s in zip(rounds, stability) if s is not None]
    valid_stab   = [s for s in stability if s is not None]

    if valid_stab:
        axes[1, 1].plot(valid_rounds, valid_stab, marker='D',
                        color='mediumpurple', lw=2)
        axes[1, 1].axhline(y=1.0, color='green', linestyle='--',
                           alpha=0.4, label='Perfectly stable')
        axes[1, 1].axhline(y=0.5, color='red', linestyle='--',
                           alpha=0.4, label='50% reshuffling')
        axes[1, 1].set_title("Cluster Membership Stability")
        axes[1, 1].set_xlabel("Round")
        axes[1, 1].set_ylabel("Fraction of Nodes Stable")
        axes[1, 1].set_ylim(0, 1.05)
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, "Only one gossip round —\nno stability to compute",
                        ha='center', va='center', transform=axes[1, 1].transAxes,
                        color='gray')
        axes[1, 1].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        os.path.join(save_dir, "gossip_evolution.png"),
        dpi=300, bbox_inches='tight'
    )
    plt.close(fig)


def plot_network_divergence(divergences, save_dir):
    """
    Plots cosine similarity across Tier 1 and 2 nodes over communication rounds.

    Parameters
    divergences : list of (round, cos_sim) tuples from run_federated_rounds
    save_dir    : directory to save the figure
    """
    os.makedirs(save_dir, exist_ok=True)
    y = []
    x = []
    for round, cos_sim in divergences:
        y.append(round)
        x.append(cos_sim)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, linestyle='--')
    ax.set_title('Network Divergence (Tier 1 and 2 nodes) Over N Communication Rounds')
    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel('Cosine similarity across network')
    fig.savefig(os.path.join(save_dir, "cos_sim_divergence.png"),
        dpi=300, bbox_inches='tight'
    )


def plot_metric_bars(summary_stats, config_idx, auprc_stats, pfas_classes, save_dir):
    """
    Plots AUROC and AUPRC bars (global vs personalized) per PFAS class.
    AUROC plots include a random baseline at 0.5.
    AUPRC plots include per-analyte prevalence lines as the random baseline.

    Parameters
    summary_stats : per-analyte AUROC dict from evaluate_all_analytes
    config_idx    : config index for filename disambiguation
    auprc_stats   : per-analyte AUPRC dict from evaluate_all_analytes
    pfas_classes  : dict mapping class label to list of analyte column names
    save_dir      : directory to save the figures
    """
    os.makedirs(save_dir, exist_ok=True)

    for cls, analytes in pfas_classes.items():
        viable = [
            a for a in analytes
            if summary_stats.get(a, {}).get('global') or
               summary_stats.get(a, {}).get('personalized')
        ]
        if not viable:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Class {cls} PFAS Performance (FALCON-PFAS) — Config {config_idx}",
                     fontsize=13, fontweight='bold')

        names = [a.replace('_MCL_Status', '') for a in viable]
        x = np.arange(len(names))
        width = 0.35

        for ax, stat_dict, title in [
            (axes[0], summary_stats, 'AUROC'),
            (axes[1], auprc_stats,   'AUPRC')
        ]:
            g_vals = [np.mean(stat_dict.get(a, {}).get('global', [0])) for a in viable]
            p_vals = [np.mean(stat_dict.get(a, {}).get('personalized', [0])) for a in viable]

            ax.bar(x - width / 2, g_vals, width, label='Global',
                   color='steelblue', alpha=0.8)
            ax.bar(x + width / 2, p_vals, width, label='Personalized',
                   color='darkorange', alpha=0.8)

            if title == 'AUROC':
                ax.axhline(0.5, color='red', linestyle='--', lw=1,
                           label='Random (0.5)')
            else:
                # draw per-analyte prevalence as the AUPRC random baseline
                for i, a in enumerate(viable):
                    stats = summary_stats.get(a, {})
                    prev_list = stats.get('prevalence', [])
                    avg_prevalence = np.mean(prev_list) if prev_list else 0

                    if avg_prevalence > 0:
                        ax.hlines(y=avg_prevalence,
                                  xmin=i - width,
                                  xmax=i + width,
                                  color='red', linestyle='--', lw=2, zorder=3)

                ax.plot([], [], color='red', linestyle='--', label='Random (Prevalence)')
                print(f"Plotting for {a}: prevalence={avg_prevalence}, x_range=[{i-width/2}, {i+width/2}]")

            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_title(title)
            ax.set_ylabel('Score')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        fig.tight_layout()
        fig.savefig(
            os.path.join(save_dir, f"class_{cls}_performance{config_idx}.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.close(fig)


def plot_federated_geography(df, histories, run_dir):
    """
    Plots well locations colored by physical cluster, with convex hull overlays
    showing which clusters were grouped together by the federated learning process.

    Parameters
    df        : DataFrame with 'latitude', 'longitude', and 'cluster_id' columns
    histories : output dict from run_federated_rounds containing 'final_meta_clusters'
    run_dir   : path to the current run directory
    """
    meta_clusters = histories.get('final_meta_clusters', {})

    if meta_clusters is None:
        print("Warning: CFL never activated — no federated clusters to plot.")
        return

    # build inverse map from agent_id to meta-cluster label
    agent_to_meta = {}
    for meta_id, agents_in_meta in meta_clusters.items():
        for agent_id in agents_in_meta:
            agent_to_meta[agent_id] = meta_id

    df['federated_cluster'] = df['cluster_id'].map(agent_to_meta)

    plt.figure(figsize=(10, 12))

    sns.scatterplot(
        data=df, x='longitude', y='latitude',
        hue='cluster_id', palette='tab20',
        s=10, alpha=0.4, legend=False
    )

    # draw convex hull around each federated meta-cluster
    for meta_id in df['federated_cluster'].dropna().unique():
        meta_data = df[df['federated_cluster'] == meta_id][['longitude', 'latitude']].values

        if len(meta_data) >= 3:
            hull = ConvexHull(meta_data)
            hull_points = np.vstack((meta_data[hull.vertices], meta_data[hull.vertices[0]]))

            plt.plot(hull_points[:, 0], hull_points[:, 1], color='blue', lw=2, alpha=0.8)
            plt.fill(hull_points[:, 0], hull_points[:, 1], color='blue', alpha=0.1)

            centroid = meta_data.mean(axis=0)
            plt.text(centroid[0], centroid[1], f"Fed-{int(meta_id)}",
                     color='blue', fontsize=12, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.6))

    plt.title("California Federated Learning Geography: Basin Alliances")
    plt.savefig(f"{run_dir}/federated_map.png", dpi=300)