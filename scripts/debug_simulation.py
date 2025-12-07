import pandas as pd
import joblib
import numpy as np

# Load Models
clf = joblib.load('backend/models/fault_classifier.pkl')
scaler = joblib.load('backend/models/scaler_clf.pkl')

def scan_dataset(name, target_label):
    path = f"dataset/{name}.csv"
    print(f"\nScanning {name} for Target {target_label}...")
    try:
         df = pd.read_csv(path).head(3000)
    except:
         print("File not found.")
         return

    history = []
    SIGNALS = ['Vpv', 'Vdc', 'Ipv', 'ia', 'ib', 'ic', 'va', 'vb', 'vc']
    
    preds = []
    
    for idx, row in df.iterrows():
        sample = row
        snapshot = {
            'Vpv': sample['Vpv'], 'Vdc': sample['Vdc'], 'Ipv': sample['Ipv'],
            'ia': sample['ia'], 'ib': sample['ib'], 'ic': sample['ic'],
            'va': sample['va'], 'vb': sample['vb'], 'vc': sample['vc']
        }
        
        history.append(snapshot)
        if len(history) > 20: history.pop(0)
            
        if len(history) == 20:
            df_hist = pd.DataFrame(history)
            features = []
            for col in SIGNALS:
                series = df_hist[col]
                features.extend([series.mean(), series.std(ddof=0), series.max(), series.min()])
                
            X_input = np.array([features])
            X_scaled = scaler.transform(X_input)
            pred = clf.predict(X_scaled)[0]
            preds.append(pred)
        else:
            preds.append(-1) 

    # Find longest run of validation
    longest_run = 0
    run_start = 0
    current_run = 0
    best_start = 0
    
    for i, p in enumerate(preds):
        if p == target_label:
            if current_run == 0: run_start = i
            current_run += 1
        else:
            if current_run > longest_run:
                longest_run = current_run
                best_start = run_start
            current_run = 0
            
    if current_run > longest_run:
        longest_run = current_run
        best_start = run_start
        
    print(f"[{name}] Best Run: Start {best_start}, Length {longest_run}")
    mid = best_start + longest_run // 2
    # Define a safe buffer
    safe_end = best_start + longest_run
    print(f"[{name}] Use indices: {best_start} to {safe_end}")

def run_debug():
    targets = {
        'F1M': 1, 'F2M': 2, 'F3M': 3, 'F4M': 4,
        'F5M': 5, 'F6M': 6, 'F7M': 7
    }
    for name, label in targets.items():
        scan_dataset(name, label)

if __name__ == "__main__":
    run_debug()
