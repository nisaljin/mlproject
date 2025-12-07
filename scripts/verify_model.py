import pandas as pd
import joblib
import numpy as np

# Load Models
clf = joblib.load('backend/models/fault_classifier.pkl')
scaler = joblib.load('backend/models/scaler_clf.pkl')

# Feature Columns expected by Model (36 features)
# [Vpv, Vdc, Ipv, ia, ib, ic, va, vb, vc, Temp] * [mean, std, max, min] ? 
# Check scaler.feature_names_in_ or similar if available, or assume main.py features.
# main.py features: 
# Vpv_mean, Vpv_std, Vpv_max, Vpv_min
# Vdc_mean ...
# Ipv_mean ...
# ia_mean ... 
# ib_mean ...
# ic_mean ...
# va_mean ... 
# vb_mean ...
# vc_mean ...
# Temperature_mean ... (Wait, Temp features?) `main.py` lines 270+

def extract_features_from_df(df_chunk):
    # Simulates the windowed feature extraction
    stats = {}
    # Columns: Vpv, Vdc, Ipv, ia, ib, ic, va, vb, vc, Temperature
    # (Note: CSVs don't have Temp, we simulated it. But for verification, let's see)
    
    # If CSV lacks Temp, add dummy 25.0
    if 'Temperature' not in df_chunk.columns:
        df_chunk['Temperature'] = 25.0

    cols = ['Vpv', 'Vdc', 'Ipv', 'ia', 'ib', 'ic', 'va', 'vb', 'vc', 'Temperature']
    
    features = []
    for col in cols:
        series = df_chunk[col]
        features.append(series.mean())
        features.append(series.std())
        features.append(series.max())
        features.append(series.min())
        
    # Remove last 4? No, user code has 36 features?
    # 10 cols * 4 stats = 40 features.
    # main.py removes Temperature entirely? Or uses it?
    # Let's check main.py feature extraction list.
    return np.array(features).reshape(1, -1)

def test_dataset(name, path):
    try:
        df = pd.read_csv(path)
        if 'Temperature' not in df.columns:
            df['Temperature'] = 25.0 # Dummy
            
        # Take a window of 20 rows AT THE FAILURE POINT
        # Debug scan said failure starts ~34.
        df_window = df.iloc[35:55]
        
        # Extract features (Need to match main.py EXACT order)
        # main.py: Vpv, Vdc, Ipv, ia, ib, ic, va, vb, vc
        # (NO Temperature in extraction loop? Let's verify main.py)
        
        # Assumption based on 36 features: 9 cols * 4 stats = 36.
        # Temp is likely NOT used in the model features, only for rule-based?
        cols_model = ['Vpv', 'Vdc', 'Ipv', 'ia', 'ib', 'ic', 'va', 'vb', 'vc']
        
        feats = []
        for col in cols_model:
            s = df_window[col]
            feats.extend([s.mean(), s.std(), s.max(), s.min()])
            
        X = np.array(feats).reshape(1, -1)
        X_scaled = scaler.transform(X)
        pred = clf.predict(X_scaled)[0]
        
        print(f"Dataset {name}: Prediction = {pred}")
        
    except Exception as e:
        print(f"Dataset {name}: Error {e}")

print("--- Testing Healthy Datasets ---")
test_dataset("F0L (Low)", "dataset/F0L.csv")
test_dataset("F0M (Medium)", "dataset/F0M.csv")

print("\n--- Testing Fault Datasets ---")
test_dataset("F1L", "dataset/F1L.csv")
test_dataset("F2L", "dataset/F2L.csv")
