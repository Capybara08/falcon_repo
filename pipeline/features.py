
from utils.tools import recommendScaler, customEncodeGem
import pandas as pd 
import torch

def analyte_meta_tensor(chain_names, analyte_metadata):
    """
    Builds a float tensor of molecular descriptors for a given list of analytes.
    Returns a zero row for any analyte missing from analyte_metadata.
 
    Parameters
    chain_names       : list of analyte column names
    analyte_metadata  : dict mapping analyte name to descriptor dict
    """
    rows = [] 
    # Standardizing the row structure for consistency.
    default_meta = {
        "is_pfca": 0,
        "is_pfsa": 0,
        "chain_len": 0,
        "ether": 0
    }
    
    for analyte in chain_names:
        # Use .get() to return default_meta if the analyte key is missing.
        meta = analyte_metadata.get(analyte, default_meta)
        
        rows.append([ 
            float(meta.get("is_pfca", 0)),
            float(meta.get("is_pfsa", 0)),
            float(meta.get("chain_len", 0)),
            float(meta.get("ether", 0)),        
        ])
        
    return torch.tensor(rows, dtype=torch.float32)

def partition_pfas_universe(train_df, test_df, global_feats_list, global_chain_list, pfas_classes):
    """
    Splits, scales, and encodes training and test DataFrames into global/local
    feature and target splits. Scalers and encoders are fit on train and applied to test.
 
    Parameters
    train_df          : raw training DataFrame
    test_df           : raw test DataFrame
    global_feats_list : list of global input feature column names
    global_chain_list : list of global PFAS target column names
    pfas_classes      : dict mapping class label to list of analyte column names
 
    Returns
    train_output : dict with keys 'X_glob', 'Y_glob', 'X_loc', 'Y_loc', 'geo'
    test_output  : same structure as train_output, scaled with train fit objects
    """

    def process_and_split(df, is_train=True, fit_objects=None):
        # Identify PFAS-related columns to exclude from features.
        pfas_related_cols = [c for c in df.columns if any(x in c.upper() for x in 
                     ['PF', 'FTS', 'MCL', 'HFPO', 'ADONA', 'PFAS', 
                      '11CL', '9CL', 'NETFO', 'NMEFO', 'FOSA', 'ETFO', 'MEFO'])]        
        
        # Define the columns we want to preserve for plotting.
        admin_cols = ['cluster_id', 'gm_well_id', 'node_tier', 'basin_cluster_id', 
                      'latitude', 'longitude', 'gm_gis_dwr_basin', 'gm_gis_county']
        
        # Ensure only try to grab admin cols that actually exist in the dataframe.
        available_admin = [c for c in admin_cols if c in df.columns]
        df_geo_final = df[available_admin].copy()

        if 'PFAS_total' in pfas_related_cols:
            pfas_related_cols.remove('PFAS_total')
            
        forbidden = set(pfas_related_cols + admin_cols)

        # X Features (Model inputs).
        valid_glob = [c for c in global_feats_list if c in df.columns and c not in forbidden]
        raw_glob_feat = df[valid_glob].copy()
        raw_loc_feat = df[[c for c in df.columns if c not in global_feats_list and c not in forbidden]].copy()

        # Y Targets.
        df_glob_chain_final = df[global_chain_list].copy()
        flat_pfas = [item for sublist in pfas_classes.values() for item in sublist]
        df_loc_chain_final = df[[
            p for p in flat_pfas
            if p not in global_chain_list and p in df.columns
        ]].copy()

        # Scaling/Encoding.
        gf_num = raw_glob_feat.select_dtypes(include=['number'])
        gf_cat = raw_glob_feat.select_dtypes(include=['object'])
        lf_num = raw_loc_feat.select_dtypes(include=['number'])
        lf_cat = raw_loc_feat.select_dtypes(include=['object'])
        
        if is_train:
            gf_scaled, gf_s_rec = recommendScaler(gf_num, fit=True)
            gf_enc_results, gf_e_rec = customEncodeGem(gf_cat, fit=True)
            lf_scaled, lf_s_rec = recommendScaler(lf_num, fit=True)
            lf_enc_results, lf_e_rec = customEncodeGem(lf_cat, fit=True)
            recs = {'gf_s': gf_s_rec, 'gf_e': gf_e_rec, 'lf_s': lf_s_rec, 'lf_e': lf_e_rec}
        else:
            gf_scaled = recommendScaler(gf_num, fit=False, scaler_map=fit_objects.get('gf_s'))
            gf_enc_results = customEncodeGem(gf_cat, fit=False, encoders=fit_objects.get('gf_e'))
            lf_scaled = recommendScaler(lf_num, fit=False, scaler_map=fit_objects.get('lf_s'))
            lf_enc_results = customEncodeGem(lf_cat, fit=False, encoders=fit_objects.get('lf_e'))
            recs = None

        # Assemble.
        X_glob = pd.concat([gf_scaled, gf_enc_results['enc_feats']], axis=1)
        X_loc = pd.concat([lf_scaled, lf_enc_results['enc_feats']], axis=1)

        # Return the dictionary including the 'geo' data.
        return {
            'X_glob': X_glob, 
            'Y_glob': df_glob_chain_final, 
            'X_loc': X_loc, 
            'Y_loc': df_loc_chain_final,
            'geo': df_geo_final 
        }, recs

    train_output, train_recs = process_and_split(train_df, is_train=True)
    test_output, _ = process_and_split(test_df, is_train=False, fit_objects=train_recs)
    
    return train_output, test_output