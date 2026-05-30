import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import os
import re
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import folium 
import math
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter # Good practice for many calls
import pickle 
import geopandas as gpd
from shapely.geometry import Point
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from pathlib import Path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from config import PFAS_CLASSES
"""
Dropped columns include: 'gm_dataset_name'
"""
def perBasin(df, pkl_filepth=None):
    """
    Fills missing basin and DWR region data via spatial join with CA groundwater basin shapefiles.
    Loads from pickle if one exists and is already fully converted; otherwise runs the join and saves.
 
    Parameters
    df           : DataFrame with 'gm_gis_dwr_basin', 'latitude', 'longitude' columns
    pkl_filepth  : optional path to a pickled version of the converted DataFrame
    """
    # Check if pickled version already exists.
    if pkl_filepth and os.path.exists(pkl_filepth):
        with open(pkl_filepth, 'rb') as file:
            df_loaded = pickle.load(file)
            # Check for remaining unknown.
            remaining = df_loaded[df_loaded['gm_gis_dwr_basin'].isin(['unknown'])]
            if len(remaining) == 0:
                print("Loaded fully converted basin data from pickle.")
                return df_loaded
            else:
                print("Pickled file found but contains unknowns. Re-processing...")

    # Identify rows that need filling.
    # Adjust these strings to match how 'unknowns' are currently labeled.
    mask_missing = df['gm_gis_dwr_basin'].isin(['unknown'])
    null_basin_df = df[mask_missing].copy()

    if null_basin_df.empty:
        print("No missing basins found.")
        return df

    print(f"Processing {len(null_basin_df)} rows for Basin/Region identification...")

    try:
        # Load DWR Shapefile.
        # Path should point to the .shp file.
        shp_path = 'i08_B118_CA_GroundWaterBasins_2003'
        basin_gdf = gpd.read_file(shp_path)
        basin_gdf = basin_gdf.to_crs("EPSG:4326")

        # Convert only the null rows to a GeoDataFrame.
        geometry = [Point(xy) for xy in zip(null_basin_df['longitude'], null_basin_df['latitude'])]
        null_basin_gdf = gpd.GeoDataFrame(null_basin_df, geometry=geometry, crs="EPSG:4326")

        # Spatial Join.
        # This identifies which polygon each point is 'within'.
        joined = gpd.sjoin(null_basin_gdf, basin_gdf, how='left', predicate='within')

        # Update the original dataframe with the new info.
        # Standard B118 column names: Basin_Name, Basin_ID, and HR_NAME (Region).
        for idx in null_basin_df.index:
            new_basin = joined.loc[idx, 'Basin_Name']
            new_region = joined.loc[idx, 'HR_NAME']
            
            # If the spatial join found a match, update the original DF.
            if pd.notna(new_basin):
                df.at[idx, 'gm_gis_dwr_basin'] = new_basin
                df.at[idx, 'gm_gis_dwr_region'] = new_region
                print(f"Index {idx}: Found {new_basin} in {new_region}")
            else:
                df.at[idx, 'gm_gis_dwr_basin'] = 'OUTSIDE OF KNOWN REGION'
                df.at[idx, 'gm_gis_dwr_region'] = 'OUTSIDE OF KNOWN REGION'

    except Exception as e:
        print(f"Spatial join error: {e}")

    # Final check and Pickle.
    remaining_unknowns = df[df['gm_gis_dwr_basin'].isin(['unknown'])]
    if len(remaining_unknowns) == 0:
        save_path = pkl_filepth if pkl_filepth else 'convertedBasins.pkl'
        df.to_pickle(save_path)
        print(f"Pickled converted basins to {save_path}")
        return df
    else:
        print(f"Did not pickle. {len(remaining_unknowns)} rows still unknown.")
        return df
    
def perCounty(df, pkl_filepth=None):
    """
    Fills missing county data via reverse geocoding. Loads from pickle if available
    and fully converted; otherwise runs the geocoding and saves the result.
 
    Parameters
    df          : DataFrame with 'gm_gis_county', 'latitude', 'longitude' columns
    pkl_filepth : optional path to a pickled version of the converted DataFrame
    """
    print("YEARS SINCE 2017 IN PER COUNTY: ", df['years_since_2016'].head())

    if pkl_filepth: 
        with open(pkl_filepth, 'rb') as file:
            df = pickle.load(file)
            unknown_countyAFTER = df[df['gm_gis_county']=='unknown']
            no_countyAFTER = df[df['gm_gis_county']=='NO COUNTY FOUND']
            # Check that there are no unknowns or no county left.
            if len(unknown_countyAFTER)==0 and len(no_countyAFTER)==0:
                return df
            else:
                print("There are still unknowns or no counties left in df after conversion.")

    # Counties were not converted. Execute county convert and pickle the final df.
    no_county = df[df['gm_gis_county']=='NO COUNTY FOUND']
    unknown_county  = df[df['gm_gis_county']=='unknown']
    null_county = pd.concat([no_county, unknown_county], axis=0)
    null_county = null_county[['latitude', 'longitude']]

    # Drop San Benito because only 1 data sample
    rural_counties = ['SAN BENITO'] # Do not drop the rural counties.
    # Get county for the unknown/no county rows.
    geolocator = Nominatim(user_agent='my_county_finder_app')
    geocode_reverse = RateLimiter(geolocator.reverse, min_delay_seconds=2)

    print("LATITUDE AND LONG: ", null_county[['latitude', 'longitude']])
    print("Columns: ", null_county.columns)

    print("NO CONVERTED DF WAS FOUND — running reverse geocoding")
    for row in null_county.itertuples():
        lat = row.latitude
        long = row.longitude
        try:
            location = geocode_reverse((lat, long))
        except Exception as e:
            print(f"  Geocoding failed for ({lat}, {long}): {e} — skipping row")
            continue

        if not location:
            print(f"  No location returned for ({lat}, {long})")
            continue

        address = location.raw.get('address', {})
        # Nominatim uses 'county' for most CA counties but falls back to
        # 'city', 'town', or 'municipality' for some unincorporated areas.
        county_name = (
            address.get('county')
            or address.get('city')
            or address.get('town')
            or address.get('municipality')
            or address.get('state_district')
        )

        if county_name:
            # Normalize to uppercase to match the rest of your county values.
            df.loc[row.Index, 'gm_gis_county'] = county_name.upper().replace(' COUNTY', '').strip()
            print(f"  ({lat}, {long}) → {df.loc[row.Index, 'gm_gis_county']}")
        else:
            print(f"  No county field in address for ({lat}, {long}). Full: {location.address}")
            # Leave as-is — don't overwrite with None

    remaining_unknowns = df[df['gm_gis_county'].isin(['unknown', 'NO COUNTY FOUND'])]
    if len(remaining_unknowns) == 0:
        save_path = pkl_filepth if pkl_filepth else 'convertedCounties.pkl'
        df.to_pickle(save_path)
        print(f"Pickled converted counties to {save_path}")
        return df
    else:
        print(f"Still {len(remaining_unknowns)} rows unconverted after geocoding:")
        print(remaining_unknowns[['latitude', 'longitude', 'gm_gis_county']])
        # Return df anyway — don't crash the pipeline over a few unconverted rows
        return df

    #     print("NO CONVERTED DF WAS FOUND")
    #     for row in null_county.itertuples(): # Indexes are not reset.
    #         lat = row.latitude
    #         long = row.longitude
    #         location = geocode_reverse((lat, long))
    #         print(lat, long)
    #         print("LOCATION: ", location)
    #         if location:
    #             county_name = location.raw.get('address', {}).get('county')
    #             print("COUNTY NAME: ", county_name)
    #             df.loc[row.Index, 'gm_gis_county'] = county_name
    #             if county_name:
    #                 print(f"Coordinates ({lat}, {long}) are in County: {county_name}")
    #             else:
    #                 print(f"County data not directly found in 'county' field. Full address: {location.address}")
    #         else:
    #             print(f"Could not find location for ({lat}, {long})")

    # except Exception as e:
    #     print(f"An error occurred: {e}")
    
    # remaining_unknowns = df[df['gm_gis_county'].isin(['unknown', 'NO COUNTY FOUND'])]
    # if len(remaining_unknowns)==0:
    #     # Pickle it.
    #     save_path = pkl_filepth if pkl_filepth else 'convertedCounties.pkl'
    #     df.to_pickle(save_path)
    #     print(f"Pickled converted counties to {save_path}")
    #     return df
    # else:
    #     print(f"Did not pickle. Still {len(remaining_unknowns)} rows left to convert.")
    #     return df
def findBasins(df):
    """
    Fills in the missing basin data using the geopandas locator merged with the CA groundwater basins datasets
    """
    # Load DWR Basin shapefile.
    basin_gdf = gpd.read_file('pfas_code.py/i08_B118_CA_GroundWaterBasins_2003/i08_B118_CA_GroundWaterBasins_2003.shp')
    basin_gdf = basin_gdf.to_crs("EPSG:4326")
    point_list = []
    coor_df = df[['latitude', 'longitude']]
    
    for idx, row in coor_df.iterrows():
        lat, long = row
        point_list.append({'geometry': Point(long, lat)})
    
    # Create the geodataframe and add the points to it.
    coordinates_gdf = gpd.GeoDataFrame(point_list, crs="EPSG:4326")

    # Spatial join.
    joined = gpd.sjoin(coordinates_gdf, basin_gdf, how='left', predicate='within')
    print("Columns of joined geodf: ", joined.columns)
    print("BASIN IDS: ", joined['Basin_ID'])
    print("BASIN NAMES: ", joined['Basin_Name'])
    print("TOTAL NANS: ", joined.isna().sum())

def cleaned_data():
    """
    Loads the raw PFAS CSV, drops unused admin columns, and plots basic
    well abundance and missingness diagnostics.
 
    Returns
    df : raw DataFrame ready for preprocess()
    """
    df = pd.read_csv('/Users/JasL/falcon_repo/data/cali_pfas.csv')
    cols_to_drop = ['gm_dataset_name']
    print(f"Dropping cols: ", cols_to_drop)
    df = df.drop(columns=cols_to_drop)
    # Get the frequency of each unique well ID.
    well_counts = df['gm_well_id'].value_counts()
    # Calculate statistics from those frequencies.
    min_occ = well_counts.min()
    max_occ = well_counts.max()
    avg_occ = well_counts.mean()
    total_unique = len(well_counts)
    fig, ax = plt.subplots(figsize=(8,5))
    well_counts.hist(ax=ax, bins=50, edgecolor='black')
    ax.set_yscale('log') 
    ax.set_title(f"Histogram of Well Data Abundance")
    ax.set_xlabel('Samples per well')
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig('figs/preprocess_figs/well_dist_hist.png', dpi=300, bbox_inches='tight')
    pfas_list = ['PFHxA', 'PFHpA','PFOA','PFNA','PFDA','PFUnA','PFDoA','PFTrDA','PFTA','PFBS',
                'PFHxS','PFOS','NETFOSAA','NMEFOSAA',
            'ADONA','HFPO_DA','11ClPF3OUDS','9ClPF3ONS',
            'PFBA','PFPeA','4:2FTS','6:2FTS','8:2FTS','PFPeS','PFHpS','PFNS','PFDS','FOSA',
            'PFHxDA','PFODA','10:2FTS','ETFOSE','ETFOSA','MEFOSE','MEFOSA'
    ]
    pfas_availability_temporal(df)
    less_75 = 0
    less_50 = 0
    for count in well_counts:
        if count < 75:
            less_75 +=1
            if count < 50:
                less_50 +=1
    print(f"Percent of wells with less than 75 samples {(less_75/len(well_counts))*100}%")
    print(f"Percent of wells with less than 30 samples {(less_50/len(well_counts))*100}%")
    print(f"Num of unique wells: {df['gm_well_id'].nunique()}")

    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    ax.set_title("Missing Values Matrix")

    return df

def mapWells(df):
    """
    Renders an interactive Folium map of well locations colored by PFOA level.
    Saves the result to figs/pfas_wells_map.html.
 
    Parameters
    df : DataFrame with 'latitude', 'longitude', 'gm_well_id', and 'PFOA' columns
    """
    # Calculate the mean center of all wells to set the initial map view.
    map_center = [df['latitude'].mean(), df['longitude'].mean()]

    # Create a base map centered at the average location.
    m = folium.Map(location=map_center, zoom_start=12, control_scale=True)
    for index, row in df.iterrows():
        # This example uses red markers for high PFAS levels.
        marker_color = 'red' if row['PFOA'] > 4.0 else 'blue'

        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Well ID: {row['gm_well_id']}<br>PFAS Level: {row['PFOA']} ppt",
            icon=folium.Icon(color=marker_color)
        ).add_to(m)

    # Save the map as an HTML file.
    m.save('figs/pfas_wells_map.html')

def preprocess(df):
    """
    Renames raw columns to standardized names, adds the years_since_2016 temporal feature,
    and converts all detected PFAS columns to numeric.
 
    Parameters
    df : raw DataFrame from cleaned_data()
    """

    df.columns = df.columns.str.strip()    
    if 'date' in df.columns:
        print("date in df cols")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    f = df.sort_values(by=['date'])
    start_date = pd.Timestamp('2016-01-01') # Start of year.
    df['years_since_2016'] = (df['date'] - start_date).dt.days / 365.25 # Get fractional years (seasonal account).
    df = df.drop(columns=['date', 'PFAS_total'])

    df = df.rename(columns=lambda x: x.strip())

    # Standardize column names.
    pfas_rename_map = {
    # Class A
    'PFHA': 'PFHxA',
    'PFHPA': 'PFHpA',
    'PFOA': 'PFOA',         # Already matches
    'PFNA': 'PFNA',         # Already matches
    'PFNDCA': 'PFDA',
    'PFUNDCA': 'PFUnA',
    'PFDOA': 'PFDoA',
    'PFTRIDA': 'PFTrDA',
    'PFTEDA': 'PFTA',
    'PFBSA': 'PFBS',
    'PFHXSA': 'PFHxS',
    'PFOS': 'PFOS',         # Already matches
    'S6NETFOSAA': 'NETFOSAA',
    'NMEFOSAA': 'NMEFOSAA', # Already matches

    # Class B 
    'HFPA-DA': 'HFPO_DA',   # Standardizing hyphen to underscore.
    '11ClPF3OUDS': '11ClPF3OUDS',
    '9ClPF3ONS': '9ClPF3ONS',
    'ADONA': 'ADONA',

    # Class C
    'PFBTA': 'PFBA',
    'PFPA': 'PFPeA',
    '4:2FTS': '4:2FTS',     # Already matches
    '6:2FTS': '6:2FTS',     # Already matches
    '8:2FTS': '8:2FTS',     # Already matches
    'PFPES': 'PFPeS',
    'PFHPSA': 'PFHpS',
    'PFNS': 'PFNS',         # Already matches
    'PFDSA': 'PFDS',
    'PFOSA': 'FOSA',

    # Class D
    'PFHXDA': 'PFHxDA',     # Already matches
    'PFODA': 'PFODA',       # Already matches
    '10:2FTS': '10:2FTS',   # Already matches
    'ETFOSE': 'ETFOSE',     # Already matches
    'ETFOSA': 'ETFOSA',     # Already matches
    'MEFOSE': 'MEFOSE',     # Already matches
    'MEFOSA': 'MEFOSA',      # Already matches
    'Ratio_15_Bar_Water_to_Clay_<2mm': 'Ratio_Water_Clay'
}
    
    df = df.rename(columns=pfas_rename_map)
    rename_dict = {col: col.replace(' ', '_').replace('-', '_') for col in df.columns}
    df = df.rename(columns=rename_dict)
    print("Renamed!")
    print(f"Inside Preprocess - Is PFHxA present? {'PFHxA' in df.columns}")
    if 'PFHxA' not in df.columns:
        # If False, let's see what the column is actually called right now.
        # Look for anything starting with PFH.
        pfh_cols = [c for c in df.columns if c.startswith('PFH')]
        print(f"PFH-type columns found: {pfh_cols}")

    # Convert columns to numeric.
    pfas_pattern = re.compile(
    r'^PF'           # PF-prefixed (PFOA, PFOS, PFNA etc.)
    r'|^[0-9]+:[0-9]+FTS'  # FTS series (4:2FTS etc.)
    r'|^HFPO'        # HFPO_DA (GenX)
    r'|^(NET|NME)FOSAA'    # sulfonamide variants
    r'|^FOSA$'       # FOSA
    r'|^(ET|ME)FOS'  # ETFOSE, ETFOSA, MEFOSE, MEFOSA
    r'|^ADONA'       # ADONA
    r'|^[0-9]+Cl'    # 11ClPF3OUDS, 9ClPF3ONS
    r'|^[0-9]+:2FTS', # catches 10:2FTS if not already matched
    re.IGNORECASE
    )
    all_pfas_cols = [col for col in df.columns if pfas_pattern.match(col)]

    # Convert all found columns to numeric.
    for col in all_pfas_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def label_mcls(df):
    """
    Converts raw numeric PFAS concentrations to binary MCL exceedance labels
    and drops the original concentration columns.
 
    Parameters
    df : preprocessed DataFrame from preprocess()
    """
    # Individual MCL classifications.
    mcl_compounds = ['PFHxA','PFHpA','PFOA','PFNA','PFDA','PFUnA','PFDoA','PFTrDA','PFTA','PFBS','PFHxS','PFOS','NETFOSAA','NMEFOSAA',
        'ADONA','HFPO_DA','11ClPF3OUDS','9ClPF3ONS',
        'PFBA','PFPeA','4:2FTS','6:2FTS','8:2FTS','PFPeS','PFHpS','PFNS','PFDS','FOSA',
        'PFHxDA','PFODA','10:2FTS','ETFOSE','ETFOSA','MEFOSE','MEFOSA']
    
    # Create the new labeled columns.
    for col in mcl_compounds:
        if col in df.columns:
            df[f'{col}_MCL_Status'] = df[col].apply(lambda x: mcl(col, x))
            print("Dtype of mcl col after mcl transformation in preprocess: ", df[col].dtype)
    # Deletes old raw pfas cols.
    for col in mcl_compounds:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Convert column names to acceptable format.
    rename_dict = {col: col.replace(' ', '_').replace('-', '_') for col in df.columns}
    df = df.rename(columns=rename_dict)

    # SAVE SUMMARY STATS OF ENTIRE DATASET.
    # save = input("Finished data preprocessing. Save new summary stats? ")
    # if save.lower().strip() == 'yes':
    #     print("Saving summary stats from preprocessing...")
    #     # df.info()
    #     with open('data-info/df-info.txt', 'w', encoding='utf-8') as f:
    #         df.info(verbose=True, buf=f)
    #     # dtypes
    #     with open('data-info/dtypes.txt', 'w', encoding='utf-8') as f:
    #         df.dtypes.to_string(buf=f)   
    #     # describe()
    #     with open('data-info/summary-stats.txt', 'w', encoding='utf-8') as f:
    #         df.describe().to_string(buf=f)
    return df

def mcl(chemical, concentration):
    """
    Returns 1.0 if concentration >= the California MCL for the given analyte, 0.0 if below,
    or NaN if the concentration is missing or the analyte has no defined MCL.
 
    Parameters
    chemical      : analyte name string
    concentration : raw numeric concentration value (ppt)
    """
    mcls = {
        'PFOS': 4.0, 'PFOA': 4.0,
        'PFNA': 10.0, 'PFDA': 10.0, 'PFUnA': 10.0, 'PFDoA': 10.0,
        'PFTrDA': 10.0, 'PFTA': 10.0, 'PFHpS': 10.0, 'PFNS': 10.0,
        'PFDS': 10.0, 'PFHxS': 10.0, 'HFPO_DA': 10.0, 'PFHxDA': 10.0,
        'PFODA': 10.0, 'PFHpA': 10.0, 'ADONA': 10.0,
        '11ClPF3OUDS': 10.0, '9ClPF3ONS': 10.0,
        'PFHxA': 10.0,       
        'FOSA': 100.0, '8:2FTS': 100.0, 'NETFOSAA': 100.0, 'NMEFOSAA': 100.0,
        '6:2FTS': 200.0, '4:2FTS': 500.0,
        'PFBS': 2000.0, 'PFBA': 2000.0, 'PFPeS': 2000.0, 'PFPeA': 2000.0,
        '10:2FTS': 100.0,    
        'ETFOSE': 100.0,     
        'ETFOSA': 100.0,     
        'MEFOSE': 100.0,    
        'MEFOSA': 100.0,   
    }

    if chemical not in mcls:
        return np.nan

    try:
        conc = float(concentration)
    except (TypeError, ValueError):
        return np.nan

    if math.isnan(conc):
        return np.nan

    return 1.0 if conc >= mcls[chemical] else 0.0

def missing_val_classifiers(df):
    """
    Displays missing values in precursors, short, long chain PFAS
    """
    df = df[['PFBTA', 'PFPA', 'PFHxA', 'PFHPA',
            'PFOA', 'PFNA', 'PFNDCA', 'PFDOA', 'PFTRIDA', 'PFTEDA',
            'PFOSA','ETFOSE','ETFOSA','MEFOSE','MEFOSA','NETFOSAA','NMEFOSAA',
            'PFBS','4:2FTS', '6:2FTS', '8:2FTS','PFHxS', 'PFHPSA', 'PFOS','PFPES']]
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')

    # Add a title and labels for clarity
    ax.set_title("Missing Values Heatmap for Precursors, Short- and Long- chain PFAS")
    ax.set_xlabel("Features")
    ax.set_ylabel("Values")
    # plt.show()
    fig.savefig('figs/missing_val_map_precursor_pfas.png', dpi=300,  bbox_inches='tight')

def featEngineering(df):
    """
    Adds engineered environmental features: leaching index, atmospheric loading.
    Skips any feature if its source columns are missing.
 
    Parameters
    df : preprocessed DataFrame
    """
    df = df.copy()

    def has_cols(cols):
        return all(c in df.columns for c in cols)

    # Leaching index.
    if has_cols(['Soil_Moisture_mm', 'Gradation_Uniformity', 'Ratio_Water_Clay']):
        df['leaching_index'] = (
            df['Soil_Moisture_mm'] * df['Gradation_Uniformity'] /
            (df['Ratio_Water_Clay'] + 1e-6)
        )
    else:
        print("WARNING: leaching_index skipped")

    # Atmospheric loading.
    if has_cols(['pm2_5_atm', 'air_humidity', 'pm10_atm']):
        df['atm_loading'] = df['pm2_5_atm'] * df['air_humidity'] + df['pm10_atm']
    else:
        print("WARNING: atm_loading skipped")

    return df 

def chem_informed(df):
    """
    Adds chemistry-informed features: ionic strength proxy, redox proxy, and a binary
    reducing-conditions flag.
 
    Parameters
    df : preprocessed DataFrame with TDS, FE, SO4, MN, NO3N columns
    """
    # Ionic strength.
    df['env_ionic_strength'] = np.log1p(df['TDS'])
    
    # Calculate the proxy.
    df['redox_proxy'] = np.log1p((df['FE'] + df['SO4'] + df['MN'] + 0.1) / (df['NO3N'] + 0.1))    
    df['env_redox_reducing'] = (df['redox_proxy'] > 0).astype(int)

    return df

def pfas_availability_temporal(df):
    """
    Prints yearly detection counts, first detection years, and plots temporal coverage
    for all available PFAS columns.
 
    Parameters
    df : raw or preprocessed DataFrame with a 'date' column
    """    
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df['date'], errors='coerce')
    df_copy['year'] = df_copy['date'].dt.year
    print("EARLIEST YEAR: ", df_copy['year'].min())
    pfas_list = ['PFHxA', 'PFHpA','PFOA','PFNA','PFDA','PFUnA','PFDoA','PFTrDA','PFTA','PFBS',
                'PFHxS','PFOS','NETFOSAA','NMEFOSAA',
            'ADONA','HFPO_DA','11ClPF3OUDS','9ClPF3ONS',
            'PFBA','PFPeA','4:2FTS','6:2FTS','8:2FTS','PFPeS','PFHpS','PFNS','PFDS','FOSA',
            'PFHxDA','PFODA','10:2FTS','ETFOSE','ETFOSA','MEFOSE','MEFOSA'
    ]
    
    # Filter to columns that exist.
    available_pfas = [col for col in pfas_list if col in df_copy.columns]
    print(f"Available PFAS ({len(available_pfas)}/{len(pfas_list)}): {available_pfas}")
    
    # Yearly non-missing counts.
    yearly_stats = pd.DataFrame()
    for pfas in available_pfas:
        non_missing = df_copy[df_copy[pfas].notna() & (df_copy[pfas] > 0)]
        yearly_stats[pfas] = non_missing.groupby('year').size()
    
    print("\n=== YEARLY PFAS DETECTIONS ===")
    print(yearly_stats.fillna(0).astype(int))
    
    # Cumulative first detection.
    first_detections = {}
    for pfas in available_pfas:
        first_year = df_copy[df_copy[pfas].notna()]['year'].min()
        first_detections[pfas] = first_year
    
    print("\n=== FIRST DETECTION YEARS ===")
    first_df = pd.DataFrame(list(first_detections.items()), 
                           columns=['PFAS', 'First_Year']).sort_values('First_Year')
    print(first_df)
    
    # Coverage evolution plot.
    plt.figure(figsize=(15, 8))
    
    # Yearly coverage %.
    yearly_coverage = df_copy[available_pfas].notna().groupby(df_copy['year']).sum()
    yearly_coverage.plot(kind='bar', width=0.8)
    plt.title('PFAS Detection Frequency by Year')
    plt.ylabel('Number of Detections')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Missingness evolution.
    plt.figure(figsize=(15, 6))
    for i, pfas in enumerate(['PFOA', 'PFOS', 'HFPO_DA', 'PFBS', '4:2FTS']):
        if pfas in df.columns:
            plt.subplot(1, 5, i+1)
            missing_by_year = df_copy[pfas].isna().groupby(df_copy['year']).mean()
            missing_by_year.plot(marker='o')
            plt.title(f"{pfas}\nMissing %")
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
    plt.tight_layout()
    
    # RECOMMENDED TIMESTAMPS based on data.
    print("\n=== RECOMMENDED TIMESTAMP FEATURES ===")

def correlation_matrices(df_main):
    """
    Plots Pearson and Spearman correlation heatmaps and a hierarchical clustering
    dendrogram for a subset of PFAS, VOC, and water quality columns.
 
    Parameters
    df_main : preprocessed DataFrame
    """    
    
    short_PFCA = [col for col in ['PFBTA', 'PFPA', 'PFHxA', 'PFHPA'] if col in df_main.columns]
    long_PFCA = [col for col in ['PFOA', 'PFNA', 'PFNDCA', 'PFDOA', 'PFTRIDA', 'PFTEDA'] if col in df_main.columns]
    vocs = [col for col in ['PCE', 'MTBE', 'VC', 'DCA12', 'DCE11', 'TCLME', 'FC11', 'FC12', 
                            'CTCL', 'TCA111', 'BZME', 'TMB124', 'XYLENES'] if col in df_main.columns]
    water_qual = [col for col in ['FE', 'TDS', 'MN', 'NO3N', 'SO4', 'AS'] if col in df_main.columns]

    spearman_df = df.corr(method='spearman')
    pearson_df = df.corr()

    # Pearson correlation matrix / heatmap.
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(pearson_df, 
                annot=True, 
                cmap='coolwarm', 
                vmin=-1, vmax=1,
                center=0,
                linewidths=0.5,
                fmt='.2f',
                annot_kws={"size": 8},
                cbar_kws={'label': 'Correlation'})
    
    # Pearson correlation matrix / heatmap.
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(spearman_df, 
                annot=True, 
                cmap='coolwarm', 
                vmin=-1, vmax=1,
                center=0,
                linewidths=0.5,
                fmt='.2f',
                annot_kws={"size": 8},
                cbar_kws={'label': 'Correlation'})
    # Combine all desired columns.
    selected_cols = short_PFCA + long_PFCA + vocs + water_qual
    selected_cols = list(set(selected_cols))  # Remove duplicates.
    # Filter to columns that exist.
    selected_cols = [col for col in selected_cols if col in df_main.columns]

    if len(selected_cols) < 2:
        return
    df_sub = df_main[selected_cols]

    # Compute correlation.
    pearson_corr = df_sub.corr()
    spearman_corr = df_sub.corr(method='spearman')
    
    # Dendogram to visualize hierarchy.
    corr = df_sub.corr(method='spearman')

    # Dissimilarity (1 - |corr|).
    dissimilarity = 1 - np.abs(corr)
    condensed_dist = squareform(dissimilarity)

    # Hierarchical clustering.
    Z = hierarchy.linkage(condensed_dist, method='complete')

    # Get optimal leaf order.
    leaf_order = hierarchy.leaves_list(Z)
    ordered_labels = corr.columns[leaf_order]

    # Save Dendrogram separately.
    plt.figure(figsize=(12, 6))
    dend = hierarchy.dendrogram(
        Z,
        labels=corr.columns,
        orientation='top',
        leaf_rotation=90,
        leaf_font_size=10
    )
    plt.title("Dendrogram of Spearman Correlations (complete linkage)")
    plt.tight_layout()
    plt.savefig('figs/dendrogram.png', dpi=300, bbox_inches='tight')
    plt.close()  

    # Create clustered heatmap.
    g = sns.clustermap(
        spearman_corr,
        method='complete',         
        metric='correlation',       
        cmap='coolwarm',            
        vmin=-1, vmax=1,
        annot=True,                 
        fmt='.2f',                  
        figsize=(10, 8),
        dendrogram_ratio=0.1,      
        cbar_pos=(0.02, 0.8, 0.05, 0.18),  
        tree_kws={'linewidth': 1.5} 
    )

    g.ax_heatmap.set_title("Hierarchical Clustered Spearman Correlation Heatmap", pad=20)
    plt.tight_layout()
    g.savefig('figs/hierarchical_spearman.png', dpi=300, bbox_inches='tight')

    # Pearson correlation matrix / heatmap.
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(pearson_corr, 
                annot=True, 
                cmap='coolwarm', 
                vmin=-1, vmax=1,
                center=0,
                linewidths=0.5,
                fmt='.2f',
                annot_kws={"size": 8},
                cbar_kws={'label': 'Correlation'})
    
    plt.title("Pearson Correlation Matrix of Selected PFAS, VOCs, and Water Quality")
    plt.tight_layout()

    save_fig = input("Save pearson correlation matrix? ")

    try:
        with open("my_folder/my_file.txt", 'r') as file:
            content = file.read()
            print("Successfully read file.")
    except FileNotFoundError:
        print("File not found. Creating a default...")
    except PermissionError:
        print("Permission denied when accessing the file.")
        if save_fig.lower() == 'yes':
            fig = ax.get_figure()
            fig.savefig('figs/pearson_matrix.png', dpi=300, bbox_inches='tight')

    # Spearman correlation matrix / heatmap.
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(spearman_corr, 
                annot=True, 
                cmap='coolwarm', 
                vmin=-1, vmax=1,
                center=0,
                linewidths=0.5,
                fmt='.2f',
                annot_kws={"size": 8},
                cbar_kws={'label': 'Correlation'})
    plt.title("Spearman Correlation Matrix of Selected PFAS, VOCs, and Water Quality")
    plt.tight_layout()

    save_fig = input("Save spearman correlation matrix? ")
    if save_fig.lower() == 'yes':
        fig = ax.get_figure()
        fig.savefig('figs/spearman_matrix.png', dpi=300,  bbox_inches='tight')

def safe_filename(name: str) -> str:
    """Converts any string into a safe filename by removing invalid characters."""
    name = str(name).lower().strip()
    # Replace common invalid chars with underscore.
    name = re.sub(r'[^a-z0-9_]', '_', name)  # Keep only letters, numbers, _, NOT -
    # Collapse multiple underscores.
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores.
    name = name.strip('_')
    # Fallback if name is empty after cleaning.
    return name or 'column_unknown'
    
def histograms(df):
    """
    Saves a log-scale histogram for every numeric column to figs/histograms/.
 
    Parameters
    df : DataFrame
    """
    os.makedirs('falcon_repo/figs/histograms', exist_ok=True)
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns

    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot histogram with 'sqrt' bins rule (good heuristic).
        df[col].hist(ax=ax, bins=50, edgecolor='black')
        ax.set_yscale('log')  
        ax.set_title(f"Histogram of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        
        filename = f"{safe_filename(col)}_hist.png"
        file_pth = os.path.join('figs', 'histograms', filename)

        fig.savefig(file_pth, dpi=300, bbox_inches='tight')        
        print("Saved to histogram folder")
        plt.close(fig)  

def boxplots(df):
    """
    Saves a boxplot for every numeric column to boxplots/.
 
    Parameters
    df : DataFrame
    """
    os.makedirs('falcon_repo/figs/boxplots', exist_ok=True)
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        df.boxplot(column=col, ax=ax)
        
        ax.set_title(f"Boxplot of {col}")
        fig.tight_layout()
        filename = f"{safe_filename(col)}_box.png"
        file_pth = os.path.join('boxplots', filename)
    
        fig.savefig(file_pth, dpi=300, bbox_inches='tight')
        print("Saved to boxplot folder")
        plt.close(fig) 

def VIF(df, max_cols=20, min_var=0.01):
    """
    Computes Variance Inflation Factor for the top max_cols numeric columns by variance.
    Flags features with VIF > 10 as potential multicollinearity problems.
 
    Parameters
    df       : DataFrame
    max_cols : maximum number of columns to include
    min_var  : minimum variance threshold for column inclusion
    """    
    df_num = df.select_dtypes(include=['number']).fillna(0)
    # Filter: variance --> threshold + top N cols
    variances = df_num.var()
    variable_cols = variances[variances > min_var].index
    top_cols = variances.nlargest(max_cols).index  # Get top 20 BY VARIANCE.
    
    print(f"Using {len(top_cols)} variable columns")
    
    # Add constant + compute VIF.
    X = sm.add_constant(df_num[top_cols])
    vif_data = []
    
    for i, col in enumerate(X.columns):
        try:
            vif = variance_inflation_factor(X.values, i)
            vif_data.append({'feature': col, 'VIF': vif})
        except:
            vif_data.append({'feature': col, 'VIF': np.inf})
    
    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
    print(vif_df.head(10))
    
    # Flag real problems.
    high_vif = vif_df[vif_df['VIF'] > 10]
    if len(high_vif) > 0:
        print(f"{len(high_vif)} features with VIF > 10:")
        print(high_vif[['feature', 'VIF']].head())
    else:
        print("No problematic multicollinearity")

    return high_vif
        

if __name__=="__main__":

    df = preprocess(cleaned_data())    
    print("DF num cols: ", len(df.columns))
    print("DF num rows: ", len(df))
    print("DF summary: ", df.head())

