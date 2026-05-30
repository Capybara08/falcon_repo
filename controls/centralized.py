import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from pipeline.bayesModel import PFASGlobal, PFASDataset, train_global, get_bayes_pred
from pipeline.features import analyte_meta_tensor
from config import GLOBAL_FEATS, GLOBAL_CHAIN, MOLECULAR_DESCRIPTORS, TRAIN_CONFIG, PATHS
from pipeline.features import partition_pfas_universe
from pipeline.preprocess import cleaned_data, preprocess, featEngineering, label_mcls, perCounty, perBasin, chem_informed
from pathlib import Path
import pickle
"""
Baseline centralized ML performance. Bayesian MLP.
"""

# grab the data
def run_centralized_baseline():
    # Load data — same pipeline as run.py
    df_path = Path(PATHS['processed_df'])
    with open(df_path, 'rb') as f:
        df = pickle.load(f)
    df = chem_informed(df)

    df_sorted = df.sort_values('years_since_2016')
    split = round(len(df_sorted) * 0.8)
    train_df = df_sorted.iloc[:split]
    test_df  = df_sorted.iloc[split:]

    # Reuse exact same feature pipeline — critical for fair comparison
    train_sets, test_sets = partition_pfas_universe(
        train_df, test_df, GLOBAL_FEATS, GLOBAL_CHAIN, {}  # empty PFAS_CLASSES → no local chain
    )
    GLOBAL_FEATS_FIN = train_sets['X_glob'].columns.tolist()
    GLOBAL_CHAIN_FIN = train_sets['Y_glob'].columns.tolist()

    # Dummy local feats (zeros) — PFASDataset needs them but centralized ignores them
    local_feats_dummy = []

    TRAIN_DF = pd.concat([train_sets['X_glob'], train_sets['Y_glob']], axis=1)
    TEST_DF  = pd.concat([test_sets['X_glob'],  test_sets['Y_glob']],  axis=1)

    # Need cluster_id col for PFASDataset — assign a single dummy cluster
    TRAIN_DF['cluster_id'] = 0

    mol_meta = analyte_meta_tensor(GLOBAL_CHAIN_FIN, MOLECULAR_DESCRIPTORS)
    model = PFASGlobal(
        input_dim=len(GLOBAL_FEATS_FIN) * 2,
        output_dim=len(GLOBAL_CHAIN_FIN),
        hidden_dim=TRAIN_CONFIG['hidden_dim'],
        dropout=TRAIN_CONFIG['dropout'],
        molecular_descriptors=mol_meta
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    train_loader = DataLoader(
        PFASDataset(TRAIN_DF, GLOBAL_FEATS_FIN, local_feats_dummy,
                    GLOBAL_CHAIN_FIN, []),
        batch_size=64, shuffle=True
    )

    N = len(TRAIN_DF)
    adaptive_alpha = torch.ones(len(GLOBAL_CHAIN_FIN))

    print("Training centralized Bayesian baseline...")
    for epoch in range(50):  # straightforward training, no rounds
        state, history = train_global(
            model, train_loader, optimizer,
            epochs=1,
            kl_beta=TRAIN_CONFIG['kl_beta'],
            N_global=N,
            adaptive_alpha=adaptive_alpha
        )
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: loss={history['total'][-1]:.4f}")

    # Evaluate
    TEST_DF_EVAL = pd.concat([test_sets['X_glob'], test_sets['Y_glob']], axis=1)
    TEST_DF_EVAL['cluster_id'] = 0
    test_loader = DataLoader(
        PFASDataset(TEST_DF_EVAL, GLOBAL_FEATS_FIN, local_feats_dummy,
                    GLOBAL_CHAIN_FIN, []),
        batch_size=len(TEST_DF_EVAL), shuffle=False
    )

    x_gb, y_gb, _, _ = next(iter(test_loader))
    means, _ = get_bayes_pred(model, x_gb, None, target_type='global')

    print("\n--- Centralized Baseline AUROC ---")
    for i, analyte in enumerate(GLOBAL_CHAIN_FIN):
        y_true = y_gb[:, i].numpy()
        mask = ~np.isnan(y_true)
        if mask.sum() > 0 and len(np.unique(y_true[mask])) == 2:
            auc = roc_auc_score(y_true[mask], means[:, i][mask])
            print(f"  {analyte.replace('_MCL_Status',''):20s}  AUROC: {auc:.4f}")

if __name__ == '__main__':
    run_centralized_baseline()