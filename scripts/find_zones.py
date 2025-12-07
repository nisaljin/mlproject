import pandas as pd
import numpy as np
import joblib
import os
import sys

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# If running in container from /app/scripts, project root is /app
PROJECT_ROOT = "/app"
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Feature Columns (Must match main.py)
SIGNALS = ['Vpv', 'Ipv', 'Vdc', 'ia', 'ib', 'ic', 'va', 'vb', 'vc']
# Map short code to full label if needed, or just check "F5" in string
TARGET_F5 = "Parallel Arc Fault (F5)"

def find_golden_zones(fault_code="F5"):
    print(f"--- Searching for Golden Zones for {fault_code} ---")
    
    # 1. Load Model
    model_path = os.path.join(MODELS_DIR, "stability.pkl")
    if not os.path.exists(model_path):
        # Try alternate name
        model_path = os.path.join(MODELS_DIR, "stability_monitor.pkl")
        
    if not os.path.exists(model_path):
        print("❌ Model not found!")
        return

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # 2. Load Dataset
    fname = f"{fault_code}M.csv" # M for Medium Irradiance
    csv_path = os.path.join(DATASET_DIR, fname)
    if not os.path.exists(csv_path):
        print(f"❌ Dataset {fname} not found!")
        return
        
    print(f"Loading {fname}...")
    df = pd.read_csv(csv_path)
    
    # 3. Sliding Window Scan
    window_size = 20
    valid_start = -1
    longest_run = 0
    best_zone = (0, 0)
    
    current_run_start = -1
    current_run_len = 0
    
    # Only scan first 5000 rows to save time
    limit = min(5000, len(df) - window_size)
    
    print(f"Scanning {limit} windows...")
    
    # Pre-calculate features strictly matching main.py
    # We do row-by-row to simulate stream, but vectorizing a bit for speed?
    # Actually, main.py does: `features = [mean, std(ddof=1), max, min] for col in SIGNALS`
    # We must replicate exactly.
    
    results = []
    
    for i in range(0, limit, 5): # Step 5 to speed up
        window = df.iloc[i : i+window_size]
        
        feats = []
        for col in SIGNALS:
            s = window[col]
            feats.extend([s.mean(), s.std(ddof=1), s.max(), s.min()])
            
        X = np.array([feats])
        pred_label = str(model.predict(X)[0])
        
        # Check if F5
        is_match = (fault_code in pred_label) # e.g. "F5" in "F5"
        
        if is_match:
            if current_run_start == -1:
                current_run_start = i
            current_run_len += 5
        else:
            if current_run_start != -1:
                # End of run
                if current_run_len > longest_run:
                    longest_run = current_run_len
                    best_zone = (current_run_start, i)
                    print(f"Found Zone: {best_zone} (Len: {current_run_len})")
                
                current_run_start = -1
                current_run_len = 0
                
    if current_run_start != -1 and current_run_len > longest_run:
         best_zone = (current_run_start, limit)
         print(f"Found Zone: {best_zone} (Len: {current_run_len})")
         
    print(f"\n🏆 BEST GOLDEN CHANNEL for {fault_code}: {best_zone}")

if __name__ == "__main__":
    find_golden_zones("F5")
    # find_golden_zones("F1") # Can verify F1 too
