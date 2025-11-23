import os
import ssl
import urllib.request
import pandas as pd
import io

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

def download_file(url, filename):
    print(f"Downloading {url}...")
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read()
            with open(filename, 'wb') as f:
                f.write(data)
        print(f"Successfully saved to {filename}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

# URLs for raw data
wdbc_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
wpbc_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data"

# Download raw files
if download_file(wdbc_url, 'wdbc.data') and download_file(wpbc_url, 'wpbc.data'):
    print("Downloads complete. Processing into CSVs...")
    
    # Define column names based on UCI documentation
    # 30 features + ID + Diagnosis
    features = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']
    # 1 = mean, 2 = se, 3 = worst
    feature_names = []
    for suffix in ['_mean', '_se', '_worst']:
        for feat in features:
            feature_names.append(feat + suffix)
            
    # WDBC Columns: ID, Diagnosis, 30 features
    wdbc_cols = ['id', 'diagnosis'] + feature_names
    
    # WPBC Columns: ID, Outcome, Time, 30 features, Tumor Size, Lymph Node Status
    # Note: WPBC features are in the same order as WDBC but column names might need mapping if we were using a library.
    # Since we are loading raw, we can assign names directly.
    wpbc_cols = ['id', 'outcome', 'time'] + feature_names + ['tumor_size', 'lymph_node_status']

    # Load WDBC
    try:
        df_wdbc = pd.read_csv('wdbc.data', header=None, names=wdbc_cols)
        print(f"WDBC loaded: {df_wdbc.shape}")
        
        # Load WPBC
        df_wpbc = pd.read_csv('wpbc.data', header=None, names=wpbc_cols)
        print(f"WPBC loaded: {df_wpbc.shape}")
        
        # Process WPBC to match WDBC
        # 1. Select common features
        common_cols = feature_names
        X_wpbc = df_wpbc[common_cols].copy()
        
        # 2. Add Diagnosis column (All WPBC cases are invasive -> Malignant)
        X_wpbc['diagnosis'] = 'M'
        
        # 3. Add ID (optional, but good for tracking)
        X_wpbc['id'] = df_wpbc['id']
        
        # Reorder to match WDBC: id, diagnosis, features
        X_wpbc = X_wpbc[['id', 'diagnosis'] + feature_names]
        
        # Concatenate
        df_final = pd.concat([df_wdbc, X_wpbc], ignore_index=True)
        print(f"Final Enriched Dataset: {df_final.shape}")
        
        # Save to data.csv (overwriting the old one or creating a new one)
        # "enrich" the data source but keep it compatible with data.csv
       
        df_final.to_csv('data_enriched.csv', index=False)
        print("Saved enriched data to 'data_enriched.csv'")
        
    except Exception as e:
        print(f"Error processing dataframes: {e}")

else:
    print("Could not download files. Please check your internet connection.")
