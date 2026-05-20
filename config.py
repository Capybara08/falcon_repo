"""
config.py
All experiment constants and hyperparameter configs in one place.
Import from here rather than defining inline in run.py.
"""

# FEATURE DEFINITIONS

GLOBAL_FEATS = [
    'Facility_1km',
    'Facility_3km'
    'TDS',
    'SO4',
    'FE',
    'MN',
    'NO3N',
    'GW_recharge', 
    'Precipitation_mm', 
    'Soil_Moisture_mm', 
    'Evapotranspiration_mm',
    'Texture_USDA', 
    '%Pass_2_Micron_Sieve', 
    'Gradation_Uniformity'
    'redox_proxy',
    'leaching_index',
    'env_ionic_strength',
    'env_redox_reducing',
    'Ratio_Water_Clay',
    'atm_loading',
    'Fac_Conf_type'
    'years_since_2016',
    'Weight_%_0.05_2mm_Clay_Free',
    'Weight_%_0.25_0.5mm_Clay_Free_<2mm'
]

"""
OG GLOBAL FEATS
    'gm_well_category',
    'redox_proxy',
    'env_redox_reducing',
    'leaching_index',
    'atm_loading',
    'env_ionic_strength',
    'Facility_50km', 'Facility_1km', 'Facility_3km', 'Facility_10km',
    'Weight_%_0.25_0.5mm_Clay_Free_<2mm',
    'years_since_2016',
    'Fac_Conf_type',
    'Weight_%_0.05_2mm_Clay_Free',
    'air_humidity',
    'Gradation_Uniformity',
    'NO3N',
    'Ratio_Water_Clay',
    'pm10_atm',
    'air_NO2',
    'air_SO2',
    'Soil_Moisture_mm',
    'pm2_5_atm',
"""

# 4 global chain PFAS analytes for Classifier Chain.
GLOBAL_CHAIN = [
    'PFOA_MCL_Status',
    'PFHxA_MCL_Status',
    'PFOS_MCL_Status',
    'PFHxS_MCL_Status'
   ]

# Experiments - PFCA vs PFSA chain.
PFCA_CHAIN = [
    'PFOA_MCL_Status',
    'PFNA_MCL_Status',
    'PFDA_MCL_Status',
    'PFUnA_MCL_Status',
    'PFTrDA_MCL_Status',
    'PFHxA_MCL_Status',
    'HFPO_DA_MCL_Status'
]
PFSA_CHAIN = [
    'PFOS_MCL_Status',
    'PFHxS_MCL_Status',
    'PFHpS_MCL_Status',
    'PFBS_MCL_Status',
]

# All 35 PFAS grouped by data density (dense -> sparse).
PFAS_CLASSES = {
    'A': [
        'PFHxA_MCL_Status', 'PFHpA_MCL_Status', 'PFOA_MCL_Status',
        'PFNA_MCL_Status', 'PFDA_MCL_Status', 'PFUnA_MCL_Status',
        'PFDoA_MCL_Status', 'PFTrDA_MCL_Status', 'PFTA_MCL_Status',
        'PFBS_MCL_Status', 'PFHxS_MCL_Status', 'PFOS_MCL_Status',
        'NETFOSAA_MCL_Status', 'NMEFOSAA_MCL_Status',
    ],
    'B': [
        'ADONA_MCL_Status', 'HFPO_DA_MCL_Status',
        '11ClPF3OUDS_MCL_Status', '9ClPF3ONS_MCL_Status',
    ],
    'C': [
        'PFBA_MCL_Status', 'PFPeA_MCL_Status',
        '4:2FTS_MCL_Status', '6:2FTS_MCL_Status', '8:2FTS_MCL_Status',
        'PFPeS_MCL_Status', 'PFHpS_MCL_Status', 'PFNS_MCL_Status',
        'PFDS_MCL_Status', 'FOSA_MCL_Status',
    ],
    'D': [
        'PFHxDA_MCL_Status', 'PFODA_MCL_Status', '10:2FTS_MCL_Status',
        'ETFOSE_MCL_Status', 'ETFOSA_MCL_Status',
        'MEFOSE_MCL_Status', 'MEFOSA_MCL_Status',
    ],
}

# Flat list of all PFAS targets — used for LOCAL_CHAIN construction.
ALL_PFAS_TARGETS = [a for targets in PFAS_CLASSES.values() for a in targets]

# Admin columns — never used as model features.
ADMIN_COLS = [
    'gm_well_id', 'gm_gis_county', 'gm_gis_dwr_basin',
    'latitude', 'longitude', 'cluster_id', 'node_tier',
    'clustered_by', 'raw_id', 'basin_cluster_id',
]

MOLECULAR_DESCRIPTORS = {
    
"PFHxA_MCL_Status": {"is_pfca": 1, "is_pfsa": 0, "chain_len": 6, "ether": 0},  # C6F13COO⁻
    "PFHpA_MCL_Status": {"is_pfca": 1, "is_pfsa": 0, "chain_len": 7,  "ether": 0},  # C7F15COO⁻
    "PFOA_MCL_Status":  {"is_pfca": 1, "is_pfsa": 0, "chain_len": 8,  "ether": 0},  # C8F17COO⁻
    "PFNA_MCL_Status":  {"is_pfca": 1, "is_pfsa": 0, "chain_len": 9,  "ether": 0},  # C9F19COO⁻
    "PFDA_MCL_Status":  {"is_pfca": 1, "is_pfsa": 0, "chain_len": 10,  "ether": 0},  # C10F21COO⁻
    "PFUnA_MCL_Status": {"is_pfca": 1, "is_pfsa": 0, "chain_len": 11, "ether": 0},  # C11F23COO⁻
    "PFDoA_MCL_Status": {"is_pfca": 1, "is_pfsa": 0, "chain_len": 12,  "ether": 0},  # C12F25COO⁻
    "PFTrDA_MCL_Status": {"is_pfca": 1, "is_pfsa": 0, "chain_len": 13,  "ether": 0},  # C13F27COO⁻
    "PFTA_MCL_Status":  {"is_pfca": 1, "is_pfsa": 0, "chain_len": 14,  "ether": 0},  # C14F29COO⁻

    "PFBS_MCL_Status":  {"is_pfca": 0, "is_pfsa": 1, "chain_len": 4,  "ether": 0},  # C4F9SO₃⁻
    "PFHxS_MCL_Status": {"is_pfca": 0, "is_pfsa": 1, "chain_len": 6,  "ether": 0},  # C6F13SO₃⁻
    "PFOS_MCL_Status":  {"is_pfca": 0, "is_pfsa": 1, "chain_len": 8,  "ether": 0},  # C8F17SO₃⁻
    "NETFOSAA_MCL_Status": {"is_pfca": 0, "is_pfsa": 1, "chain_len": 8,  "ether": 0},  # N‑ethyl C8F17SO₃⁻
    "NMEFOSAA_MCL_Status": {"is_pfca": 0, "is_pfsa": 1, "chain_len": 8,  "ether": 0},  # N‑methyl C8F17SO₃⁻

    "ADONA_MCL_Status":     {"is_pfca": 1, "is_pfsa": 0, "chain_len": 9,  "ether": 1},  # ether‑linked C9F17OCH₂COO⁻ (approx)
    "HFPO_DA_MCL_Status":   {"is_pfca": 1, "is_pfsa": 0, "chain_len": 6,  "ether": 1},  # CF₃CF₂CF₂OCHFCF₂COO⁻
    "11ClPF3OUDS_MCL_Status": {"is_pfca": 0, "is_pfsa": 1, "chain_len": 11,  "ether": 1},  # chlorinated ether C11F21OCH₂CH₂SO₃⁻ (approx)
    "9ClPF3ONS_MCL_Status": {"is_pfca": 0, "is_pfsa": 1, "chain_len": 9, "ether": 1},   # chlorinated ether C9F17OCH₂SO₃⁻ (approx)

    "PFBA_MCL_Status":  {"is_pfca": 1, "is_pfsa": 0, "chain_len": 4, "ether": 0},  # C4F9COO⁻
    "PFPeA_MCL_Status": {"is_pfca": 1, "is_pfsa": 0, "chain_len": 5, "ether": 0},  # C5F11COO⁻
    "4:2FTS_MCL_Status": {"is_pfca": 0, "is_pfsa": 1, "chain_len": 4,  "ether": 0}, # C4F9SO₃⁻ (fluorotelomer sulfonate anion)
    "6:2FTS_MCL_Status": {"is_pfca": 0, "is_pfsa": 1, "chain_len": 6, "ether": 0}, # C6F13SO₃⁻
    "8:2FTS_MCL_Status": {"is_pfca": 0, "is_pfsa": 1, "chain_len": 8,  "ether": 0}, # C8F17SO₃⁻
    "PFPeS_MCL_Status":  {"is_pfca": 0, "is_pfsa": 1, "chain_len": 5,  "ether": 0},  # C5F11SO₃⁻
    "PFHpS_MCL_Status":  {"is_pfca": 0, "is_pfsa": 1, "chain_len": 7,  "ether": 0},  # C7F15SO₃⁻
    "PFNS_MCL_Status":   {"is_pfca": 0, "is_pfsa": 1, "chain_len": 9, "ether": 0},  # C9F19SO₃⁻
    "PFDS_MCL_Status":   {"is_pfca": 0, "is_pfsa": 1, "chain_len": 10,  "ether": 0}, # C10F21SO₃⁻
    "FOSA_MCL_Status":   {"is_pfca": 0, "is_pfsa": 1, "chain_len": 8,  "ether": 0},  # C8F17SO₂NH₂ anion (approx as sulfonamide)

    "PFHxDA_MCL_Status": {"is_pfca": 1, "is_pfsa": 0, "chain_len": 6,  "ether": 1},  # C6F13OCH₂COO⁻ (ether)
    "PFODA_MCL_Status":  {"is_pfca": 1, "is_pfsa": 0, "chain_len": 8, "ether": 1},  # C8F17OCH₂COO⁻ (ether)
    "10:2FTS_MCL_Status": {"is_pfca": 0, "is_pfsa": 1, "chain_len": 10,  "ether": 0}, # C10F21SO₃⁻
    "ETFOSE_MCL_Status":  {"is_pfca": 0, "is_pfsa": 1, "chain_len": 8, "ether": 0},  # C8F17SO₂NHCH₂CH₃ anion
    "ETFOSA_MCL_Status":  {"is_pfca": 0, "is_pfsa": 1, "chain_len": 8, "ether": 0},  # same core as above, sulfonamide anion
    "MEFOSE_MCL_Status":  {"is_pfca": 0, "is_pfsa": 1, "chain_len": 8, "ether": 0},  # C8F17SO₂NHCH₃ anion
    "MEFOSA_MCL_Status":  {"is_pfca": 0, "is_pfsa": 1, "chain_len": 8, "ether": 0},  # same core, sulfonamide anion
}


# EXPERIMENT CONFIGS

configs = [
    {
        'comm_rounds': 35,
        'local_epochs': 10,
        'finetune_epochs': 15,
        'gossip_every': 5,         # first gossip only at round 5 and 10
        'description': 'Baseline — let weights diverge before any aggregation',
    },
    # {
    #     'comm_rounds': 10,
    #    'local_epochs': 5,
    #     'finetune_epochs': 8,
    #     'gossip_every': 5,         # first gossip only at round 5 and 10
    #     'description': 'Baseline — let weights diverge before any aggregation',
    # },
    # {
    #     'comm_rounds': 15,
    #     'local_epochs': 8,
    #     'finetune_epochs': 8,
    #     'gossip_every': 5,         # gossip at rounds 4, 8, 12, 16, 20
    #     'description': 'Medium — moderate local training with periodic sync',
    # },
    # {
    #    'comm_rounds': 20,
    #     'local_epochs': 8,
    #     'finetune_epochs': 15,
    #     'gossip_every': 5,         # gossip at rounds 5, 10, 15, 20, 25, 30
    #     'description': 'Long — deep local training, infrequent sync',
    # }
    # {
    #     'comm_rounds': 30,
    #     'local_epochs': 15,
    #     'finetune_epochs': 15,
    #     'gossip_every': 10,        # only 3 gossip events — near-independent training
    #     'description': 'Long — minimal gossip, tests personalization over federation',
    # },
    # {
    #     'comm_rounds': 40,
    #     'local_epochs': 10,
    #     'finetune_epochs': 20,
    #     'gossip_every': 5,
    #     'description': 'Extended — longer finetune phase, tests personalized tail depth',
    # },
]

"""
Default Configs
    {
        'comm_rounds': 10,
        'local_epochs': 5,
        'finetune_epochs': 5,
        'gossip_every': 5,         # first gossip only at round 5 and 10
        'description': 'Baseline — let weights diverge before any aggregation',
    },
    {
        'comm_rounds': 10,
       'local_epochs': 5,
        'finetune_epochs': 8,
        'gossip_every': 5,         # first gossip only at round 5 and 10
        'description': 'Baseline — let weights diverge before any aggregation',
    },
    {
        'comm_rounds': 15,
        'local_epochs': 8,
        'finetune_epochs': 8,
        'gossip_every': 5,         # gossip at rounds 4, 8, 12, 16, 20
        'description': 'Medium — moderate local training with periodic sync',
    },
    {
       'comm_rounds': 20,
        'local_epochs': 8,
        'finetune_epochs': 15,
        'gossip_every': 5,         # gossip at rounds 5, 10, 15, 20, 25, 30
        'description': 'Long — deep local training, infrequent sync',
    }
"""

# TRAINING HYPERPARAMETERS

TRAIN_CONFIG = {
    # Bayesian KL scaling.
    'kl_beta': 0.001, # Base KL weight before N scaling.

    # # num federated clusters - TEMPORARY
    # 'n_meta_clusters': 8,

    # Feature weights for clustered fed learning - agglomerative
    # piloting some... (4/6)

    # Gossip / meta-clustering.
    'similarity_threshold': 0.80, # Cosine sim cutoff for clustering.
    'base_momentum': 0.70, # Blend weight: own weights vs averaged weights.

    # Model architecture.
    'hidden_dim': 32, # BayesianMLP hidden layer size
    'dropout': 0.2,

    # DataLoader
    'batch_size': 32,
    'shuffle_train': True,

    # Checkpointing.
    'checkpoint_every_n_rounds': 5,

    # Personalization density thresholds (create_personalized_tail).
    'local_thres_ratio': 0.003, # analyte must be tested in ≥3% of node rows
    'local_min_detections': 3, # must have ≥3 positive hits
    # Class imbalance
    'pos_weight_cap': 10.0, # Max pos_weight in masked_loss.
}

# Environmental features.

tier_1 = ['Facility_50km', 'Facility_1km', 'Facility_3km', 'Facility_10km']
tier_2 = ['Weight_%_0.25_0.5mm_Clay_Free_<2mm', 'Weight_%_0.02_0.05mm_Clay_Free_<2mm', 'Weight_%_0.05_2mm_Clay_Free', 'Gradation_Uniformity']
tier_3 = ['Ratio_Water_Clay', 'GW_Runoff_mm', 'Silt_Total', 'Sand_Total', 'Evapotranspiration_mm']
tier_4 = ['air_humidity', 'pm10_atm', 'air_NO2', 'air_SO2', 'pm2_5_atm', 'air_pm1', 'air_Ozone', 'Temp', 'WindSpeed']

## Call this func in run.py
def get_ohe_env_weighted(all_feat_cols): 
    """
    Appends OHE-expanded categorical columns to the appropriate tier lists
    so env_weights covers the full post-encoding feature space.
 
    Parameters
    all_feat_cols : list of all feature column names after OHE expansion
    """
    ohe_list = []
    for col in all_feat_cols:
        if any(x in col.lower() for x in ['gm_well_category', 'texture_usda', 'gm_gis_dwr_basin']):
            ohe_list.append(col)
    
    for col in ohe_list:
        if 'texture_usda' in col.lower():
            tier_3.append(col)
        elif 'gm_well_category' in col.lower():
            tier_1.append(col)
        elif 'gm_gis_dwr_basin' in col.lower():
            tier_3.append(col)

env_weights = {}
for f in tier_1: env_weights[f] = 4.0
for f in tier_2: env_weights[f] = 3.0
for f in tier_3: env_weights[f] = 2.0
for f in tier_4: env_weights[f] = 1.0

TRAIN_CONFIG['env_cluster_weighted'] = env_weights

# PATHS

PATHS = {
    'raw_data':        'data/cali_pfas.csv',
    'processed_df':    'data/df_KEEP.pkl',
    'meta_df':         'data/meta_df.pkl',
    'converted_counties': 'data/convertedCounties.pkl',
    'converted_basins':   'data/convertedBasins.pkl',
    'runs_base':       'runs/',
    'registry':        'runs/registry.csv',
    'basin_shapefile': 'pfas_code.py/i08_B118_CA_GroundWaterBasins_2003/'
                       'i08_B118_CA_GroundWaterBasins_2003.shp',
}
