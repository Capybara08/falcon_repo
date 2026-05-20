from pipeline.preprocess import cleaned_data, preprocess, featEngineering, label_mcls, perCounty, perBasin, chem_informed
from pipeline.features import partition_pfas_universe, analyte_meta_tensor
from pipeline.wellCluster import env_clustering, test_env_clustering
from pipeline.bayesModel import PFASLocal, PFASGlobal, train_local, create_personalized_tail, reptile_adapt_fin
from pipeline.plt import (plot_federated_loss, plot_metric_bars, plot_best_node_uncertainty,
                           plot_best_roc, plot_gossip_clusters, plot_network_divergence,
                           plot_federated_geography)
from federated.agents import createAgents
from federated.topology import fedAvgTopology
from federated.training import run_federated_rounds
from utils.logging import make_run_dir, log_run_registry
from utils.metrics import (evaluate_all_analytes, build_summary_report,
                            diagnose_node_class_distribution, cache_test_batches,
                            assign_test_loaders, update_adaptive_alpha)
from config import configs, GLOBAL_FEATS, GLOBAL_CHAIN, PFAS_CLASSES, TRAIN_CONFIG, PATHS, MOLECULAR_DESCRIPTORS
from pathlib import Path
import pickle, torch, pandas as pd, numpy as np, logging, os
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
import seaborn as sns
import shap


def set_seed(seed=42):
    """Sets all random seeds for reproducibility across torch, numpy, and python random."""
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Load preprocessed data from disk or run the full pipeline.
    df_path = Path(PATHS['processed_df'])
    if df_path.is_file():
        with open(df_path, 'rb') as f:
            df = pickle.load(f)
            if 'years_since_2016' in df.columns:
                print("YEARS SINCE 2016 IN DF COLS")
            else:
                print("WARNING: MISSING YEARS SINCE 2016")
            df = chem_informed(df)
    else:
        df = cleaned_data()
        df = preprocess(df)
        df = featEngineering(df)
        df = label_mcls(df)
        df = perCounty(df)
        df = perBasin(df)
        os.makedirs('data', exist_ok=True)
        with open(df_path, 'wb') as f:
            pickle.dump(df, f)

    pd.set_option('display.max_columns', None)

    with open('df_analytics.txt', 'a') as f:
        f.write(f"{df}")

    sns.heatmap(df[GLOBAL_CHAIN].isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.show()

    for i, config in enumerate(configs):
        run_dir = Path(make_run_dir(i, config)).resolve()

        # Configure per-run file logging.
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
        logging.basicConfig(
            filename=os.path.join(run_dir, 'run.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        set_seed(42)
        meta_df = df.copy()

        # Temporal split — earlier 80% trains, later 20% tests.
        if 'years_since_2016' in df.columns:
            print("YEARS SINCE 2016 IN COLS LINE 121")
            df_sorted = df.sort_values('years_since_2016')
        else:
            print("WARNING: years_since_2016 missing — cannot split")
            return

        split = round(len(df_sorted) * 0.8)
        train_df = df_sorted.iloc[:split]
        test_df = df_sorted.iloc[split:]

        # Scale and encode features, separate global / local targets.
        train_sets, test_sets = partition_pfas_universe(
            train_df, test_df, GLOBAL_FEATS, GLOBAL_CHAIN, PFAS_CLASSES
        )
        TRAIN_GEO_DF = train_sets['geo'].copy()
        GLOBAL_FEATS_FIN = train_sets['X_glob'].columns.tolist()
        GLOBAL_CHAIN_FIN = train_sets['Y_glob'].columns.tolist()
        LOCAL_FEATS_FIN = train_sets['X_loc'].columns.tolist()
        LOCAL_CHAIN_FIN = train_sets['Y_loc'].columns.tolist()

        TRAIN_DF = pd.concat([train_sets['X_glob'], train_sets['Y_glob'],
                               train_sets['X_loc'],  train_sets['Y_loc']], axis=1)
        TEST_DF = pd.concat([test_sets['X_glob'],  test_sets['Y_glob'],
                              test_sets['X_loc'],   test_sets['Y_loc']], axis=1)

        # Cluster training wells by environmental features only.
        TRAIN_FEATS_ONLY = pd.concat([train_sets['X_glob'], train_sets['X_loc']], axis=1)
        train_meta = meta_df.loc[TRAIN_DF.index]

        TRAIN_CLUSTERED_DF, cluster_artifacts = env_clustering(TRAIN_FEATS_ONLY)
        TRAIN_GEO_DF['cluster_id'] = TRAIN_CLUSTERED_DF['cluster_id']
        TRAIN_GEO_DF['node_tier'] = TRAIN_CLUSTERED_DF['node_tier']

        # Reattach target labels after clustering.
        TRAIN_CLUSTERED_DF = TRAIN_CLUSTERED_DF.join(
            pd.concat([train_sets['Y_glob'], train_sets['Y_loc']], axis=1)
        )

        TRAIN_CLUSTERED_DF['global'] = TRAIN_CLUSTERED_DF[GLOBAL_CHAIN_FIN].isna().all(axis=1)

        tier_dist = TRAIN_CLUSTERED_DF.groupby('cluster_id')['node_tier'].first().value_counts()
        print("Node tier distribution:\n", tier_dist)

        N_global = len(TRAIN_CLUSTERED_DF)

        logging.info(
            f"Config {i}: {len(TRAIN_CLUSTERED_DF)} train samples, "
            f"{len(TEST_DF)} test samples, "
            f"{TRAIN_CLUSTERED_DF['cluster_id'].nunique()} clusters"
        )

        # Build topology graph and instantiate agents.
        G = fedAvgTopology(TRAIN_CLUSTERED_DF)
        GLOBAL_META_MASTER = analyte_meta_tensor(GLOBAL_CHAIN_FIN, MOLECULAR_DESCRIPTORS)

        agents = createAgents(
            G, TRAIN_CLUSTERED_DF,
            GLOBAL_FEATS_FIN, LOCAL_FEATS_FIN,
            GLOBAL_CHAIN_FIN, LOCAL_CHAIN_FIN,
            MOLECULAR_DESCRIPTORS=GLOBAL_META_MASTER
        )
        print(f"{len(agents)} agents created")

        # Federated training.
        num_analytes = len(GLOBAL_CHAIN)
        adaptive_alpha = torch.ones(num_analytes)

        histories = run_federated_rounds(
            agents, G, config, N_global, run_dir,
            LOCAL_CHAIN_FIN, GLOBAL_CHAIN_FIN, GLOBAL_FEATS_FIN, LOCAL_FEATS_FIN,
            adaptive_alpha=adaptive_alpha
        )

        null_models = [nid for nid, a in agents.items() if a.get('model') is None]
        print(f"{len(null_models)} null models found")

        # Terminal finetuning — overwrites the mid-training personalized models.
        print("\n--- Terminal finetuning ---")
        for node_id, agent in agents.items():
            local_chain = create_personalized_tail(
                agent['train_df'], LOCAL_CHAIN_FIN,
                thres_ratio=TRAIN_CONFIG['local_thres_ratio'],
                min_detections=TRAIN_CONFIG['local_min_detections']
            )
            local_meta = analyte_meta_tensor(LOCAL_CHAIN_FIN, MOLECULAR_DESCRIPTORS)
            if local_chain:
                print("Personalizing...")
                local_model = PFASLocal(
                    global_model=agent['model'],
                    local_chain=local_chain,
                    input_dim_loc=len(LOCAL_FEATS_FIN) * 2,
                    molecular_descriptors=local_meta
                )
                if agent['node_tier'] == 1:
                    local_model = train_local(local_model, agent['train_loader'],
                                              local_chain, epochs=config['finetune_epochs'])
                if agent['node_tier'] == 2:
                    local_model = reptile_adapt_fin(local_model, agent['train_loader'], local_chain)

                agent['personalized_model'] = local_model
                agent['final_model_state'] = local_model.state_dict()
                agent['local_chain_names'] = local_chain
            else:
                agent['personalized_model'] = agent['model']
                agent['final_model_state'] = agent['global_model_state']
                agent['local_chain_names'] = []

        # NaN check on personalized model weights.
        print("\n--- Personalized Model NaN Check ---")
        for node_id, agent in agents.items():
            model = agent.get('personalized_model')
            if model is None:
                print("WARNING: Found a None model")
                return

            bad_params = [name for name, param in model.named_parameters()
                          if torch.isnan(param).any()]
            if bad_params:
                print(f"Node {node_id}: NaNs in personalized params -> {bad_params}")

        # Assign test data and evaluate.
        TEST_DF_LABELED = test_env_clustering(TRAIN_CLUSTERED_DF, TEST_DF, cluster_artifacts)

        print("\nTest cluster assignment counts:")
        print(TEST_DF_LABELED['cluster_id'].value_counts(dropna=False).head(20))
        print("Num assigned test clusters:", TEST_DF_LABELED['cluster_id'].nunique())
        print("Train clusters:", TRAIN_CLUSTERED_DF['cluster_id'].nunique())

        assign_test_loaders(agents, TEST_DF_LABELED,
                            GLOBAL_FEATS_FIN, LOCAL_FEATS_FIN,
                            GLOBAL_CHAIN_FIN, LOCAL_CHAIN_FIN)
        cache_test_batches(agents)

        diagnose_node_class_distribution(agents, GLOBAL_CHAIN_FIN, GLOBAL_CHAIN_FIN)

        summary_stats, auprc_stats = evaluate_all_analytes(
            agents, PFAS_CLASSES, GLOBAL_CHAIN_FIN, LOCAL_CHAIN_FIN
        )
        build_summary_report(summary_stats, auprc_stats, PFAS_CLASSES, config, run_dir)

        # Plotting.
        plot_federated_loss(
            histories['total'], i, save_dir=os.path.join(run_dir, 'figs/loss'),
            bce=histories['bce'], kl=histories['kl'],
        )
        if histories.get('gossip_logs'):
            plot_gossip_clusters(
                histories['gossip_logs'],
                save_dir=os.path.join(run_dir, 'figs/gossip')
            )
        plot_metric_bars(
            summary_stats, i, auprc_stats, PFAS_CLASSES,
            save_dir=os.path.join(run_dir, 'figs/performance')
        )
        plot_best_node_uncertainty(
            agents, i, os.path.join(run_dir, 'figs/uncertainty'),
            summary_stats, GLOBAL_CHAIN_FIN,
        )
        plot_best_roc(
            agents, i, os.path.join(run_dir, 'figs/roc'),
            summary_stats, GLOBAL_CHAIN_FIN,
        )
        plot_network_divergence(
            histories['network_divergence'],
            save_dir=os.path.join(run_dir, 'figs/divergence')
        )

        log_run_registry(
            run_dir, config,
            final_loss=histories['total'][-1],
            final_bce=histories['bce'][-1],
            final_kl=histories['kl'][-1],
            n_meta_clusters=len(histories.get('final_meta_clusters') or {})
        )

        # Build per-agent hypernet prior heatmap.
        all_agent_priors = {}
        for agent in agents:
            local_chain_names = agents[agent]['local_chain_names']
            agent_priors = {}
            for analyte, meta_data in MOLECULAR_DESCRIPTORS.items():
                if analyte not in GLOBAL_CHAIN_FIN and analyte not in local_chain_names:
                    continue
                meta_tensor = torch.tensor([list(meta_data.values())], dtype=torch.float32)
                with torch.no_grad():
                    model = agents[agent]['personalized_model']
                    if isinstance(model, PFASGlobal):
                        prior_vec = model.hypernet_scaled(meta_tensor).numpy().flatten()
                        prior_vec = np.clip(prior_vec, -1e6, 1e6)
                    elif isinstance(model, PFASLocal):
                        prior_vec = model.global_head.hypernet_scaled(meta_tensor).numpy().flatten()
                        prior_vec = np.clip(prior_vec, -1e6, 1e6)
                    else:
                        continue
                    agent_priors[analyte] = float(prior_vec.mean())
            all_agent_priors[agent] = agent_priors

        analyte_names = sorted({a for ap in all_agent_priors.values() for a in ap})
        agent_ids_ordered = list(all_agent_priors.keys())

        df_priors = pd.DataFrame(
            {node_id: [all_agent_priors[node_id].get(a, float('nan')) for a in analyte_names]
             for node_id in agent_ids_ordered},
            index=analyte_names
        )

        TIER_COLORS = {1: '#c0392b', 2: '#e67e22', 3: '#7f8c8d'}
        TIER_LABELS = {1: 'Tier 1 – Hub', 2: 'Tier 2 – Link', 3: 'Tier 3 – Anomaly'}

        print(f"Matrix NaNs: {np.isnan(df_priors.values).sum()}")
        print(f"Matrix Infs: {np.isinf(df_priors.values).sum()}")

        if np.isinf(df_priors.values).any():
            problem_agents = df_priors.columns[np.isinf(df_priors.values).any(axis=0)]
            print(f"Agents with Infinity priors: {problem_agents.tolist()}")
            return

        raw_matrix = df_priors.values
        matrix = np.nan_to_num(raw_matrix, nan=0.0, posinf=1.0, neginf=0.0)
        matrix = np.clip(matrix, 0, 1)

        n_analytes, n_agents = matrix.shape
        col_link = linkage(pdist(matrix.T, metric='euclidean'), method='average')
        row_link = linkage(pdist(matrix, metric='euclidean'), method='average')
        col_order = dendrogram(col_link, no_plot=True)['leaves']
        row_order = dendrogram(row_link, no_plot=True)['leaves']

        data_ord = matrix[np.ix_(row_order, col_order)]
        nodes_ord = [agent_ids_ordered[i] for i in col_order]
        analyt_ord = [analyte_names[i] for i in row_order]

        col_grp = list(fcluster(col_link, t=3, criterion='maxclust'))
        col_grp = [col_grp[i] for i in col_order]

        fig = plt.figure(figsize=(22, 9))
        gs = gridspec.GridSpec(3, 3, figure=fig,
                               height_ratios=[1.6, 0.22, 5.5],
                               width_ratios=[0.16, 7, 2.4],
                               hspace=0.03, wspace=0.04)

        ax_cdend = fig.add_subplot(gs[0, 1])
        ax_tier = fig.add_subplot(gs[1, 1])
        ax_heat = fig.add_subplot(gs[2, 1])
        ax_rdend = fig.add_subplot(gs[2, 0])
        ax_leg = fig.add_subplot(gs[:, 2])

        dendrogram(col_link, ax=ax_cdend, color_threshold=0,
                   above_threshold_color='#444', link_color_func=lambda k: '#444')
        ax_cdend.set_axis_off()

        tier_colors_col = [TIER_COLORS.get(agents[nid].get('node_tier', 2), '#bdc3c7')
                           for nid in nodes_ord]
        tier_cmap = matplotlib.colors.ListedColormap(tier_colors_col)
        ax_tier.imshow(np.arange(n_agents).reshape(1, -1), aspect='auto',
                       cmap=tier_cmap, vmin=0, vmax=n_agents - 1)
        ax_tier.set_xticks([])
        ax_tier.set_yticks([0])
        ax_tier.set_yticklabels(['Node Tier'], fontsize=8)
        ax_tier.tick_params(left=False)

        grp_names = {1: 'High Prior', 2: 'Medium Prior', 3: 'Low Prior'}
        grp_colors = {1: '#7b1a1a', 2: '#c0392b', 3: '#e8a0a0'}
        start, prev = 0, col_grp[0]
        boundaries, ranges = [], []
        for idx, g_lbl in enumerate(col_grp[1:], 1):
            if g_lbl != prev:
                boundaries.append(idx)
                ranges.append((start, idx - 1, prev))
                start, prev = idx, g_lbl
        ranges.append((start, n_agents - 1, prev))

        for s, e, grp in ranges:
            ax_tier.annotate(grp_names.get(grp, ''),
                             xy=((s + e) / 2, -1.4), xycoords='data',
                             ha='center', va='top', fontsize=7.5,
                             color=grp_colors.get(grp, '#333'), fontweight='bold')

        sns.heatmap(data_ord, ax=ax_heat, cmap='Reds', vmin=0, vmax=1,
                    xticklabels=nodes_ord, yticklabels=analyt_ord,
                    linewidths=0.3, linecolor='#ddd', cbar=False)

        ax_heat.set_xlabel('Agent (Cluster) ID', fontsize=11, labelpad=8)
        ax_heat.set_ylabel('')
        ax_heat.tick_params(axis='x', labelsize=7.5, rotation=90)
        ax_heat.tick_params(axis='y', labelsize=9, rotation=0)

        for b in boundaries:
            ax_heat.axvline(x=b, color='white', linewidth=2.5, linestyle='--')

        dendrogram(row_link, ax=ax_rdend, orientation='left',
                   color_threshold=0, above_threshold_color='#444',
                   link_color_func=lambda k: '#444')
        ax_rdend.set_axis_off()
        ax_rdend.invert_yaxis()

        ax_leg.set_axis_off()
        y = 0.97

        def _sec(ax, title, y_pos):
            ax.text(0.05, y_pos, title, transform=ax.transAxes,
                    fontsize=10, fontweight='bold', va='top', color='#222')
            return y_pos - 0.048

        y = _sec(ax_leg, 'Prior Strength', y)
        cb_ax = ax_leg.inset_axes([0.05, y - 0.12, 0.30, 0.11])
        cb = plt.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='Reds'),
                          cax=cb_ax, orientation='vertical')
        cb.set_ticks([0, 0.35, 0.70, 1.0])
        cb.set_ticklabels(['0.0\n(Low)', '0.35', '0.70', '1.0\n(High)'])
        cb.ax.tick_params(labelsize=8)
        ax_leg.text(0.42, y - 0.01,
                    'Mean hypernet activation\nper analyte × agent.\n'
                    'Higher = model expects\nMCL exceedance more strongly.',
                    transform=ax_leg.transAxes, fontsize=8, va='top', color='#444')
        y -= 0.20

        y = _sec(ax_leg, '② Node Tier', y)
        for tier, label in TIER_LABELS.items():
            ax_leg.add_patch(mpatches.FancyBboxPatch(
                (0.05, y - 0.027), 0.09, 0.024,
                boxstyle='round,pad=0.002', facecolor=TIER_COLORS[tier],
                transform=ax_leg.transAxes, clip_on=False))
            ax_leg.text(0.17, y - 0.013, label,
                        transform=ax_leg.transAxes, fontsize=8.5, va='center')
            y -= 0.050
        y -= 0.02

        y = _sec(ax_leg, '③ Contamination Groups', y)
        grp_descs = [
            ('#7b1a1a', 'High  (≥ 0.70)',    'Persistently contaminated zones'),
            ('#c0392b', 'Medium (0.35–0.69)', 'Mixed contamination history'),
            ('#f2c5c5', 'Low  (< 0.35)',      'Rarely exceed MCL'),
        ]
        for color, label, desc in grp_descs:
            ax_leg.add_patch(mpatches.FancyBboxPatch(
                (0.05, y - 0.027), 0.09, 0.024,
                boxstyle='round,pad=0.002', facecolor=color, edgecolor='#aaa',
                transform=ax_leg.transAxes, clip_on=False))
            ax_leg.text(0.17, y - 0.005, label,
                        transform=ax_leg.transAxes, fontsize=8.5,
                        va='top', fontweight='bold', color='#222')
            ax_leg.text(0.17, y - 0.028, desc,
                        transform=ax_leg.transAxes, fontsize=7.5, va='top', color='#555')
            y -= 0.080
        y -= 0.02

        y = _sec(ax_leg, '④ How to Read', y)
        ax_leg.text(0.05, y - 0.01,
                    '• Each column = one federated agent\n'
                    '  (a geographic well cluster)\n\n'
                    '• Each row = one PFAS analyte\n\n'
                    '• Cell = mean prior activation\n'
                    '  (collapsed from latent vector)\n\n'
                    '• Dashed lines separate contamination\n'
                    '  level groups\n\n'
                    '• Dendrograms group similar agents\n'
                    '  and analytes',
                    transform=ax_leg.transAxes,
                    fontsize=8, va='top', color='#333', linespacing=1.5)

        fig.suptitle(
            'Hypernetwork Latent Priors — Per-Agent × Per-Analyte MCL Exceedance Prior\n'
            'Rows = PFAS analytes  |  Columns = Federated agents (well clusters)',
            fontsize=13, fontweight='bold', y=1.01)

        plt.savefig(f"{run_dir}/latent_heatmap.png", dpi=180,
                    bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"[latent_heatmap] Saved → {run_dir}/latent_heatmap.png")

        # SHAP waterfall plots per agent.
        shap_base_dir = Path(f"{run_dir}/figs/shap_plots")

        def model_wrapper(model, input_tensor):
            """Wraps a model for SHAP by applying the same mask-concat logic as the Dataset."""
            x_tensor = torch.tensor(input_tensor, dtype=torch.float32)
            mask = (~torch.isnan(x_tensor)).float()
            x_combined = torch.cat([torch.nan_to_num(x_tensor, nan=0.0), mask], dim=-1)
            model.eval()
            with torch.no_grad():
                output = model(x_combined)
                if isinstance(output, tuple):
                    logits = output[1]
                else:
                    logits = output
                return logits.detach().cpu().numpy()

        agent_ids = []
        for agent in agents:
            agent_ids.append(agent)
            if agents[agent]['test_loader'] is None:
                print(f"Skipping agent {agent}: no test data assigned")
                continue

            agent_shap_dir = Path(f"{shap_base_dir}/agent_{agent}")
            agent_shap_dir.mkdir(parents=True, exist_ok=True)

            model = agents[agent]['personalized_model']
            data = agents[agent]['test_loader'].dataset.X_glob_tensors[:50].numpy()
            explainer = shap.Explainer(lambda x: model_wrapper(model, x), data)
            shap_values = explainer(data)
            is_multiclass = len(shap_values.values.shape) == 3

            local_chain_names = agents[agent]['local_chain_names']

            for out_idx, analyte_name in enumerate(local_chain_names):
                if is_multiclass:
                    analyte_values_all = shap_values.values[:, :, out_idx]
                    analyte_base_all = shap_values.base_values[:, out_idx]
                else:
                    analyte_values_all = shap_values.values
                    analyte_base_all = shap_values.base_values

                current_val_vector = analyte_values_all[0, :]
                current_base_scalar = np.array(analyte_base_all).flatten()[0].item()

                sv = shap.Explanation(
                    values=current_val_vector,
                    base_values=float(current_base_scalar),
                    data=shap_values.data[0],
                    feature_names=GLOBAL_FEATS_FIN
                )
                shap.plots.waterfall(sv, show=False)
                plt.savefig(f"{agent_shap_dir}/waterfall_{analyte_name}.png",
                            bbox_inches='tight', dpi=300)
                plt.close()

        with open('output.txt', 'a') as f:
            f.write(f"ALL AGENTS: {agent_ids}")
            f.write(f"Len of agents: {len(agents.keys())}")

        plot_federated_geography(TRAIN_GEO_DF, histories, run_dir)


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main()
