from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder, LabelEncoder
import pickle
import numpy as np
import pandas as pd

def recommendScaler(num_df, fit=None, scaler_map=None): 
    """
    Selects and fits the best scaler for each numeric column based on distribution shape.
    Columns that are all NaN are dropped. When fit=False, applies previously fit scalers.
 
    Parameters
    num_df     : DataFrame of numeric columns only
    fit        : if True, fit new scalers and return them; if False, apply scaler_map
    scaler_map : dict mapping column name to fitted scaler (required when fit=False)
 
    Returns (fit=True)
    scaled_df : DataFrame of scaled columns
    recs      : dict mapping column name to fitted scaler
 
    Returns (fit=False)
    scaled_df : DataFrame of scaled columns
    """
    filename = 'scalers.pkl'
    if fit: # Training, we are fitting scalers.
        recommendations = {} # Column as keys with vals as the actual Scaler type.
        valid_cols = []
        for col in num_df.columns:
            if num_df[col].isnull().all():
                continue
            col_df = num_df[[col]].dropna()
            if len(col_df) < 10: # Few data points, deemed "outliers" are relevant statistically
                scaler = RobustScaler()
            else:
                skew = stats.skew(col_df) # Get degree of skew.
                kurt = stats.kurtosis(col_df) # Get "tailedness" and outlier presence.
                
                # Safe normality test
                p_normal = 0.0
                if len(col_df) >= 20:
                    try:
                        stat, p = stats.normaltest(col_df) # Test for normalize distribution.
                        p_normal = p
                    except:
                        p_normal = 0.0  
                
                # Decision logic.
                if p_normal > 0.05 and abs(skew) < 0.5 and abs(kurt) < 1.0:
                    scaler = StandardScaler()
                elif abs(skew) > 2.0 or abs(kurt) > 3.0:
                    scaler = RobustScaler()
                elif abs(skew) > 1.0:
                    scaler = QuantileTransformer(output_distribution='normal')
                else:
                    scaler = StandardScaler()
            
            # Fit and store.
            scaler.fit(col_df)
            recommendations[col] = scaler
            valid_cols.append(col)

        num_df = num_df[valid_cols].copy() # Removes the non-valid cols (all NaNs).

        with open(filename, 'wb') as f:
            pickle.dump(recommendations, f)
        
        scaled_parts = [] # List of dfs.
        for col in valid_cols:
            scaled_col = pd.Series(np.nan, index=num_df.index, name=col)
            no_nan_idx = num_df[col].dropna().index
            scaled_values = (recommendations[col].transform(num_df.loc[no_nan_idx, [col]]))
            scaled_col.loc[no_nan_idx] = scaled_values.flatten()
                
            scaled_parts.append(scaled_col.to_frame())
        
        # Return scaled DataFrame.
        return pd.concat(scaled_parts, axis=1), recommendations # df, dict
    else:  
        if scaler_map is None:
            raise ValueError("scaler_map must be provided when fit=False to ensure consistent scaling.")

        scaled_parts = []
        for col in num_df.columns:
            if col in scaler_map:
                scaled_parts.append(pd.DataFrame(
                    scaler_map[col].transform(num_df[[col]]), 
                    columns=[col], index=num_df.index))
            else:
                # If it wasn't in training, we don't scale it. 
                # This prevents the KeyError: 'Facility_50km'.
                continue
                
        return pd.concat(scaled_parts, axis=1) if scaled_parts else pd.DataFrame(index=num_df.index)

def customEncodeGem(feat_df, target_df=None, fit=None, encoders=None):
    if fit: # Training.
        encoders = {'ohe': {}, 'le': {}}
        enc_dfs = {'enc_feats': {}} 

        if len(feat_df) > 0:
            for feat in feat_df.columns:
                ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                enc_data = ohe.fit_transform(feat_df[[feat]])
                feature_names = ohe.get_feature_names_out([feat])
                
                # Store the DF and the Object.
                enc_dfs['enc_feats'][feat] = pd.DataFrame(enc_data, columns=feature_names, index=feat_df.index)
                encoders['ohe'][feat] = ohe

        if target_df is not None:
            enc_targets = []
            for tar_col in target_df.columns:
                le = LabelEncoder()
                enc_targets.append(le.fit_transform(target_df[tar_col]).reshape(-1, 1))
                encoders['le'][tar_col] = le
            
            enc_dfs['enc_tar'] = np.hstack(enc_targets) if len(enc_targets) > 1 else enc_targets[0].flatten()

        # Flatten the feature dict into one DataFrame for easy concat.
        if enc_dfs['enc_feats']:
            enc_dfs['enc_feats'] = pd.concat(enc_dfs['enc_feats'].values(), axis=1)

        return enc_dfs, encoders

    else: # Testing.
        # Handle Features (Iterate through the dictionary of encoders).
        enc_feat_list = []
        if 'ohe' in encoders:
            for col, ohe_obj in encoders['ohe'].items():
                if col in feat_df.columns:
                    # Transform using the specific object for this column.
                    enc_data = ohe_obj.transform(feat_df[[col]])
                    feature_names = ohe_obj.get_feature_names_out([col])
                    enc_feat_list.append(pd.DataFrame(enc_data, columns=feature_names, index=feat_df.index))
        
        feat_result = pd.concat(enc_feat_list, axis=1) if enc_feat_list else pd.DataFrame()

        # Handle Targets.
        tar_result = None
        if target_df is not None and 'le' in encoders:
            enc_targets = []
            for col, le_obj in encoders['le'].items():
                if col in target_df.columns:
                    enc_targets.append(le_obj.transform(target_df[col]).reshape(-1, 1))
            if enc_targets:
                tar_result = np.hstack(enc_targets) if len(enc_targets) > 1 else enc_targets[0].flatten()

        return {'enc_feats': feat_result, 'enc_tar': tar_result}