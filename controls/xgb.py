"""
xgb.py
Centralized XGBoost baseline for FALCON-PFAS comparison.

Two modes:
  A) Independent per-analyte models (matches literature, e.g. Huang et al.)
  B) Classifier chain XGBoost (mirrors the FALCON chain architecture,
     isolating federation as the experimental variable)

Design constraint: must use the identical train/test split and feature encoding
as run.py. No test data is seen during training.
"""
from typing import List
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.impute import SimpleImputer
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import os
import json

from pipeline.preprocess import cleaned_data, preprocess, featEngineering, label_mcls, perCounty, perBasin
from config import configs, GLOBAL_FEATS, GLOBAL_CHAIN, PFAS_CLASSES, TRAIN_CONFIG, PATHS
from pipeline.features import partition_pfas_universe
from pipeline.plt import plot_roc_curve

n_estimators = [100, 300, 500]

LOCAL_CHAIN = []
for pfas_list in PFAS_CLASSES.values():
    for pfas in pfas_list:
        if pfas not in GLOBAL_CHAIN:
            LOCAL_CHAIN.append(pfas)


def prep_X(df_X: pd.DataFrame):
    """
    Median-imputes a feature DataFrame for XGBoost.
    Imputation is needed in chain mode where previous predictions
    are concatenated as features.

    Parameters
    df_X : DataFrame of model input features
    """
    imp = SimpleImputer(strategy='median')
    return imp.fit_transform(df_X)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray, metric='auroc') -> float | None:
    """
    Returns AUROC or AUPRC, or None if the computation is not possible
    (e.g. only one class present in y_true).

    Parameters
    y_true  : ground-truth binary labels, may contain NaN
    y_score : predicted probabilities
    metric  : 'auroc' or 'auprc'
    """
    mask = ~np.isnan(y_true)
    y_t = y_true[mask]
    y_s = y_score[mask]
    if len(np.unique(y_t)) < 2 or len(y_t) == 0:
        return None
    if metric == 'auroc':
        return roc_auc_score(y_t, y_s)
    return average_precision_score(y_t, y_s)


def get_pos_weight(y_series: pd.Series) -> float:
    """
    Computes neg/pos class ratio for XGBoost scale_pos_weight.
    Equivalent to the positive class weighting in adaptive_masked_loss.

    Parameters
    y_series : binary label Series for one analyte
    """
    n_neg = (y_series == 0).sum()
    n_pos = (y_series == 1).sum()
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def train_independent_xgb(
    num_trees: int,
    X_train: np.ndarray,
    Y_train: pd.DataFrame,
    X_test: np.ndarray,
    Y_test: pd.DataFrame,
    analyte_names: List[str],
) -> dict:
    """
    Trains one XGBClassifier per analyte independently.
    This is the standard literature approach (Huang et al., Dong et al.).

    Parameters
    num_trees     : number of boosting estimators
    X_train       : training feature array
    Y_train       : training target DataFrame
    X_test        : test feature array
    Y_test        : test target DataFrame
    analyte_names : list of analyte column names to model

    Returns
    results : dict mapping analyte name to model, auroc, auprc, predictions, and metadata
    """
    results = {}

    for analyte in analyte_names:
        if analyte not in Y_train.columns:
            continue

        y_tr = Y_train[analyte].values
        y_te = Y_test[analyte].values

        train_mask = ~np.isnan(y_tr)
        if train_mask.sum() < 10:
            print(f"  [{analyte}] Skipped — fewer than 10 labeled training samples")
            continue
        if len(np.unique(y_tr[train_mask])) < 2:
            print(f"  [{analyte}] Skipped — only one class in training data")
            continue

        scale_pos = get_pos_weight(pd.Series(y_tr[train_mask]))

        model = XGBClassifier(
            n_estimators=num_trees,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=None
        )
        model.fit(X_train[train_mask], y_tr[train_mask])
        assert len(model.get_booster().get_dump()) == num_trees, \
            f"Expected {num_trees} trees, got {len(model.get_booster().get_dump())}"

        y_prob = model.predict_proba(X_test)[:, 1]
        auroc = safe_auc(y_te, y_prob, 'auroc')
        auprc = safe_auc(y_te, y_prob, 'auprc')

        results[analyte] = {
            'model': model,
            'auroc': auroc,
            'auprc': auprc,
            'y_prob': y_prob.tolist(),
            'y_true': y_te.tolist(),
            'n_train': train_mask.sum(),
            'n_test': (~np.isnan(y_te)).sum(),
            'pos_rate_train': y_tr[train_mask].mean()
        }

        status = f"AUROC={auroc:.4f}" if auroc is not None else "AUROC=N/A (no test diversity)"
        if auprc:
            print(f"  [{analyte}] {status} | AUPRC={auprc:.4f}")
        else:
            print(f"N/A | n_train={train_mask.sum()} | pos_rate={y_tr[train_mask].mean():.3f}")

    return results


def train_chain_xgb(
    num_trees: int,
    X_train: np.ndarray,
    Y_train: pd.DataFrame,
    X_test: np.ndarray,
    Y_test: pd.DataFrame,
    chain_order: list[str],
) -> dict:
    """
    Trains a classifier chain XGBoost, where each analyte's predicted probability
    is appended as a feature for subsequent analytes.
    Chain order should match GLOBAL_CHAIN from run.py to isolate federation as the variable.

    Skipped analytes still append a zero column to keep all subsequent feature dimensions consistent.

    Parameters
    num_trees   : number of boosting estimators
    X_train     : training feature array
    Y_train     : training target DataFrame
    X_test      : test feature array
    Y_test      : test target DataFrame
    chain_order : list of analyte column names defining chain sequence

    Returns
    results : dict mapping analyte name to model, auroc, auprc, predictions, and metadata
    """
    results = {}
    models = {}

    X_tr_aug = X_train.copy()
    X_te_aug = X_test.copy()

    for analyte in chain_order:
        if analyte not in Y_train.columns:
            X_tr_aug = np.hstack([X_tr_aug, np.zeros((len(X_tr_aug), 1))])
            X_te_aug = np.hstack([X_te_aug, np.zeros((len(X_te_aug), 1))])
            continue

        y_tr = Y_train[analyte].values
        y_te = Y_test[analyte].values
        train_mask = ~np.isnan(y_tr)

        if train_mask.sum() < 10 or len(np.unique(y_tr[train_mask])) < 2:
            X_tr_aug = np.hstack([X_tr_aug, np.zeros((len(X_tr_aug), 1))])
            X_te_aug = np.hstack([X_te_aug, np.zeros((len(X_te_aug), 1))])
            print(f"  [{analyte}] Chain: skipped, appending 0s")
            continue

        scale_pos = get_pos_weight(pd.Series(y_tr[train_mask]))

        model = XGBClassifier(
            n_estimators=num_trees,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=None
        )
        model.fit(X_tr_aug[train_mask], y_tr[train_mask])
        models[analyte] = model

        tr_prob = model.predict_proba(X_tr_aug)[:, 1].reshape(-1, 1)
        te_prob = model.predict_proba(X_te_aug)[:, 1].reshape(-1, 1)

        # Append predicted probability as feature for the next link.
        X_tr_aug = np.hstack([X_tr_aug, tr_prob])
        X_te_aug = np.hstack([X_te_aug, te_prob])

        auroc = safe_auc(y_te, te_prob.flatten(), 'auroc')
        auprc = safe_auc(y_te, te_prob.flatten(), 'auprc')

        results[analyte] = {
            'model': model,
            'auroc': auroc,
            'auprc': auprc,
            'y_prob': te_prob.flatten().tolist(),
            'y_true': y_te.tolist(),
            'n_train': train_mask.sum(),
            'pos_rate_train': y_tr[train_mask].mean()
        }

        status = f"AUROC={auroc:.4f}" if auroc is not None else "AUROC=N/A"
        print(f"  [{analyte}] Chain {status} | n_train={train_mask.sum()}")

    return results


def print_comparison_table(
    independent_results: dict,
    glob_chain: dict,
    loc_chain: dict,
    pfas_classes: dict
):
    """
    Prints a side-by-side AUROC comparison of independent, global chain, and local chain
    XGBoost results grouped by PFAS class.

    Parameters
    independent_results : results dict from train_independent_xgb
    glob_chain          : results dict from train_chain_xgb on global chain
    loc_chain           : results dict from train_chain_xgb on local chain
    pfas_classes        : PFAS_CLASSES dict from config.py
    """
    print(f"\n{'='*110}")
    print(f"{'CENTRALIZED XGBOOST BASELINE — COMPARISON TABLE':^110}")
    print(f"{'='*110}")
    print(f"{'Analyte':<25} {'Indep AUROC':>12} {'GlobChain AUROC':>16} {'LocChain AUROC':>16} {'N Train':>8}")
    print("-" * 110)

    for cls, analytes in pfas_classes.items():
        print(f"\n  Class {cls}:")
        for analyte in analytes:
            ind = independent_results.get(analyte, {})
            g = glob_chain.get(analyte, {})
            l = loc_chain.get(analyte, {})

            name = analyte.replace('_MCL_Status', '')
            i_auroc = f"{ind['auroc']:.4f}" if ind.get('auroc') is not None else "—"
            g_auroc = f"{g['auroc']:.4f}" if g.get('auroc') is not None else "—"
            l_auroc = f"{l['auroc']:.4f}" if l.get('auroc') is not None else "—"
            n_tr = str(ind.get('n_train', '—'))

            print(f"  {name:<23} {i_auroc:>12} {g_auroc:>16} {l_auroc:>16} {n_tr:>8}")


def plot_xgb_roc(results, pfas_name, config_idx, save_dir):
    """
    Plots an ROC curve for one analyte using stored XGBoost results.
    Skips if data is missing or has no class diversity in the test set.

    Parameters
    results    : results dict from train_independent_xgb or train_chain_xgb
    pfas_name  : analyte column name string
    config_idx : run index for filename disambiguation
    save_dir   : directory to save the figure
    """
    if pfas_name not in results or 'y_prob' not in results[pfas_name]:
        print(f"Skipping ROC for {pfas_name}: Data not found in results.")
        return

    y_prob = np.array(results[pfas_name]['y_prob'])
    y_true = np.array(results[pfas_name]['y_true'])

    mask = ~np.isnan(y_true)
    if mask.sum() == 0 or len(np.unique(y_true[mask])) < 2:
        return

    plot_roc_curve(y_true[mask], config_idx, save_dir, y_prob[mask], f"XGBoost: {pfas_name}")


def plot_xgb_comparison_bars(indep_stats, glob_stats, loc_stats, config_idx, pfas_classes, save_dir):
    """
    Plots AUROC and AUPRC bars comparing the three XGBoost baseline modes per PFAS class.
    AUPRC plots include per-analyte prevalence lines as the random baseline.

    Parameters
    indep_stats  : results dict for independent models
    glob_stats   : results dict for global chain
    loc_stats    : results dict for local chain
    config_idx   : run index for filename disambiguation
    pfas_classes : PFAS_CLASSES dict from config.py
    save_dir     : directory to save the figures
    """
    os.makedirs(save_dir, exist_ok=True)

    for cls, analytes in pfas_classes.items():
        viable = [a for a in analytes if a in indep_stats or a in glob_stats or a in loc_stats]
        if not viable:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"XGBoost Baseline Comparison: Class {cls} (Config {config_idx})",
                     fontsize=14, fontweight='bold')

        names = [a.replace('_MCL_Status', '') for a in viable]
        x = np.arange(len(names))
        width = 0.25

        for i, metric in enumerate(['auroc', 'auprc']):
            ax = axes[i]
            ind_vals = [indep_stats.get(a, {}).get(metric, 0) or 0 for a in viable]
            glb_vals = [glob_stats.get(a, {}).get(metric, 0) or 0 for a in viable]
            loc_vals = [loc_stats.get(a, {}).get(metric, 0) or 0 for a in viable]

            ax.bar(x - width, ind_vals, width, label='Independent', color='#3498db')
            ax.bar(x,         glb_vals, width, label='Global Chain', color='#e67e22')
            ax.bar(x + width, loc_vals, width, label='Local Chain',  color='#2ecc71')

            if metric == 'auroc':
                ax.axhline(0.5, color='red', linestyle='--', alpha=0.6, label='Random')
            else:
                for j, a in enumerate(viable):
                    avg_prevalence = indep_stats.get(a, {}).get('pos_rate_train', 0)
                    ax.hlines(y=avg_prevalence,
                              xmin=j - width,
                              xmax=j + width,
                              color='red', linestyle='--', lw=1.5)
                ax.plot([], [], color='red', linestyle='--', label='Random (Prevalence)')

            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.set_title(metric.upper())
            ax.set_ylim(0, 1.05)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_dir, f"xgb_comparison_class_{cls}.png"), dpi=300)
        plt.close()


if __name__ == "__main__":
    print("Loading and preprocessing data...")

    df = cleaned_data()
    df = preprocess(df)
    df = featEngineering(df)
    df = label_mcls(df)
    print("Global features: ", GLOBAL_FEATS)

    file_path = '/Users/JasL/Desktop/research/sci-fair/pfas_code.py/data/df.pkl'
    df_path = Path(file_path)
    if not df_path.is_file():
        print(f"Does file exist? {Path('/Users/JasL/Desktop/research/sci-fair/pfas_code.py/data/df.pkl').exists()}")
        df = perCounty(df)
        df = perBasin(df)
        with open(file_path, 'wb') as f:
            pickle.dump(df, f)
            print(f"Pickled the LATEST (county filled, basin filled) version to {file_path}")

    for i, n_estimator in enumerate(n_estimators):
        df = df.sort_values(by='years_since_2016')
        train_split = round(len(df) * 0.8)
        train_df = df[:train_split]
        test_df = df[train_split:]

        print(f"Train: {len(train_df)} | Test: {len(test_df)}")

        train_sets, test_sets = partition_pfas_universe(
            train_df=train_df,
            test_df=test_df,
            global_feats_list=GLOBAL_FEATS,
            global_chain_list=GLOBAL_CHAIN,
            pfas_classes=PFAS_CLASSES
        )

        X_train = prep_X(train_sets['X_glob'])
        X_test = prep_X(test_sets['X_glob'])

        X_train_loc = prep_X(train_sets['X_loc'])
        X_test_loc = prep_X(test_sets['X_loc'])

        Y_train_glob = train_sets['Y_glob']
        Y_test_glob = test_sets['Y_glob']

        Y_train_loc = train_sets['Y_loc']
        Y_test_loc = test_sets['Y_loc']

        Y_train_all = pd.concat([Y_train_glob, Y_train_loc], axis=1)
        Y_test_all = pd.concat([Y_test_glob, Y_test_loc], axis=1)
        print(Y_test_all['PFOA_MCL_Status'].value_counts())

        all_analytes = [a for sublist in PFAS_CLASSES.values() for a in sublist]

        print("\n── Mode A: Independent Per-Analyte XGBoost ──")
        independent_results = train_independent_xgb(
            n_estimator,
            X_train, Y_train_all,
            X_test, Y_test_all,
            analyte_names=all_analytes,
        )

        print("\n── Mode B: XGBoost Classifier Chain (Global Chain) ──")
        glob_chain_results = train_chain_xgb(
            n_estimator,
            X_train, Y_train_glob,
            X_test, Y_test_glob,
            chain_order=GLOBAL_CHAIN
        )

        print("\n── Mode C: XGBoost Classifier Chain (Local Chain) ──")
        loc_chain_results = train_chain_xgb(
            n_estimator,
            X_train_loc, Y_train_loc,
            X_test_loc, Y_test_loc,
            chain_order=LOCAL_CHAIN
        )

        print_comparison_table(independent_results, glob_chain_results, loc_chain_results, PFAS_CLASSES)

        output = {
            'independent': {k: {m: v[m] for m in ['auroc', 'auprc', 'y_prob', 'y_true', 'n_train', 'pos_rate_train']}
                            for k, v in independent_results.items()},
            'chain': {
                'global': {k: {m: v[m] for m in ['auroc', 'auprc', 'y_prob', 'y_true', 'n_train', 'pos_rate_train']}
                           for k, v in glob_chain_results.items()},
                'local':  {k: {m: v[m] for m in ['auroc', 'auprc', 'n_train', 'pos_rate_train']}
                           for k, v in loc_chain_results.items()}
            }
        }

        pkl_path = 'data/xgb_baseline_results.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(output, f)
        print("\nResults saved to data/xgb_baseline_results.pkl")

        # PFOA feature importance bar chart.
        model = independent_results['PFOA_MCL_Status']['model']
        feat_imp = pd.Series(model.feature_importances_, index=train_sets['X_glob'].columns)
        feat_imp.sort_values().plot(kind='barh')
        plt.title('PFOA Feature Importance')
        plt.tight_layout()
        plt.savefig('figs/pfoa_feature_importance.png', dpi=300, bbox_inches='tight')

        # Save human-readable JSON copy of results.
        json_file_path = f"results/xgb_output_run{i}.json"
        os.makedirs('results', exist_ok=True)
        with open(pkl_path, 'rb') as f:
            output = pickle.load(f)
        with open(json_file_path, 'w') as f:
            json.dump(output, f, default=lambda x: int(x) if isinstance(x, np.int64) else x, indent=2)

        indep_summary_stats = output['independent']
        loc_chain_summary_stats = output['chain']['local']
        glob_chain_summary_stats = output['chain']['global']

        roc_dir = Path(f"results/roc_run{i}_{n_estimator}_estimators")
        os.makedirs(roc_dir, exist_ok=True)
        target_analytes = ['PFOA_MCL_Status', 'PFOS_MCL_Status', 'PFHxS_MCL_Status',
                           'PFHxA_MCL_Status', 'HFPO_DA_MCL_Status', 'PFNA_MCL_Status']

        for analyte in target_analytes:
            plot_xgb_roc(indep_summary_stats, analyte, i, os.path.join(roc_dir, '_indep'))
            plot_xgb_roc(glob_chain_summary_stats, analyte, i, os.path.join(roc_dir, '_glob'))
            plot_xgb_roc(loc_chain_summary_stats, analyte, i, os.path.join(roc_dir, '_loc'))

        metric_dir = Path(f"results/metric_run{i}_{n_estimator}_estimators")
        plot_xgb_comparison_bars(
            indep_summary_stats,
            glob_chain_summary_stats,
            loc_chain_summary_stats,
            i,
            PFAS_CLASSES,
            metric_dir
        )

        print("Successfully converted xgb pickle to readable json")