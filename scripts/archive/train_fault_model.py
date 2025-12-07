import os
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump

# --- Configuration ---
DATASET_DIR = "../dataset"
MODELS_DIR = "models"
WINDOW_SIZE = 20
STRIDE = 100  # Stride for sliding window to reduce data size & correlation

# Features expected: 36 total
# 9 Signals * 4 Stats (Mean, Std, Max, Min)
SIGNALS = ['Vpv', 'Vdc', 'Ipv', 'ia', 'ib', 'ic', 'va', 'vb', 'vc']

LABELS = {
    'F0': 0, 'F1': 1, 'F2': 2, 'F3': 3,
    'F4': 4, 'F5': 5, 'F6': 6, 'F7': 7
}

def extract_features(window):
    """
    Extracts 36 statistical features from a window of data.
    Order: [Vpv_mean, Vpv_std, ..., vc_min]
    """
    features = []
    feature_names = []
    
    for col in SIGNALS:
        if col not in window.columns:
            # Fallback for missing columns if any
            series = pd.Series(np.zeros(len(window)))
        else:
            series = window[col]
            
        # 1. Mean
        features.append(series.mean())
        # 2. Std (Population std)
        features.append(series.std(ddof=0))
        # 3. Max
        features.append(series.max())
        # 4. Min
        features.append(series.min())
        
    return np.array(features)

def generate_feature_names():
    names = []
    stats = ['mean', 'std', 'max', 'min']
    for sig in SIGNALS:
        for stat in stats:
            names.append(f"{sig}_{stat}")
    return names

def load_and_process_data():
    X = []
    y = []
    
    print("Loading datasets...")
    files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')]
    
    for filename in files:
        # Determine label from filename (e.g. F0L.csv -> F0)
        label_str = filename[:2] 
        if label_str not in LABELS:
            print(f"Skipping {filename} (Unknown Label)")
            continue
            
        label = LABELS[label_str]
        filepath = os.path.join(DATASET_DIR, filename)
        
        try:
            df = pd.read_csv(filepath)
            
            # Sliding Window
            num_windows = (len(df) - WINDOW_SIZE) // STRIDE
            print(f"Processing {filename}: {num_windows} samples...")
            
            for i in range(0, len(df) - WINDOW_SIZE, STRIDE):
                window = df.iloc[i : i + WINDOW_SIZE]
                feats = extract_features(window)
                X.append(feats)
                y.append(label)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return np.array(X), np.array(y)

def main():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    # 1. Prepare Data
    X, y = load_and_process_data()
    print(f"Total Samples: {X.shape[0]}, Features: {X.shape[1]}")
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Scale
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train XGBoost
    print("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softmax',
        num_class=8,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # 5. Evaluate
    acc = model.score(X_test_scaled, y_test)
    print(f"Test Accuracy: {acc*100:.2f}%")
    
    # 6. Save Artifacts
    print("Saving artifacts...")
    dump(model, os.path.join(MODELS_DIR, 'fault_classifier.pkl'))
    dump(scaler, os.path.join(MODELS_DIR, 'scaler_clf.pkl'))
    
    feature_names = generate_feature_names()
    dump(feature_names, os.path.join(MODELS_DIR, 'fault_features.pkl'))
    
    print("Done! Models updated in backend/models/")

if __name__ == "__main__":
    main()
