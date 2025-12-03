import sys
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from data_loader import generate_synthetic_data, get_real_world_stats, load_stability_data
from preprocessing import preprocess_features, prepare_datasets
from model import train_energy_predictor, train_balancing_classifier, train_stability_monitor

def train_and_evaluate():
    print("Loading Real-World Data for Calibration...")
    real_stats = get_real_world_stats('dataset')
    
    print("Generating calibrated synthetic data...")
    # Generate 1 month of data
    n_samples = 1440 * 30 
    data = generate_synthetic_data(n_samples, real_stats=real_stats)
    
    print("Preprocessing data...")
    data = preprocess_features(data)
    
    # 1. Train Energy Predictor (Regression)
    # Target: Net Energy (Generation - Consumption)
    # Actually, let's predict Grid_Consumption for next step as a proxy for demand forecasting
    print("\nTraining Grid Consumption Predictor...")
    
    # Create target: Next step consumption
    data['Target_Consumption'] = data['Grid_Consumption'].shift(-1)
    data_reg = data.dropna()
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg, scaler_reg, feature_cols_reg = prepare_datasets(
        data_reg, 'Target_Consumption'
    )
    
    model_reg = train_energy_predictor(X_train_reg, y_train_reg)
    
    # Evaluate
    y_pred_reg = model_reg.predict(X_test_reg)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    print(f"Grid Consumption MAE: {mae:.4f}")
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_reg, 'models/energy_predictor.pkl')
    joblib.dump(scaler_reg, 'models/scaler_reg.pkl')
    
    # 2. Train Balancing Classifier
    # Target: 0 (Discharge), 1 (Hold), 2 (Charge)
    # Logic: 
    # If SoC < 20 and Generation < Consumption -> Charge (from Grid) -> Label 2
    # If SoC > 80 and Generation > Consumption -> Discharge (to Grid) -> Label 0
    # Else -> Hold/Balance -> Label 1
    # This is a heuristic to create labels for training
    
    print("\nTraining Balancing Signal Classifier...")
    
    def get_balancing_signal(row):
        # Economical Logic:
        # Net Power = Generation - Consumption
        # If Net Power > 0 (Surplus) -> Charge (2)
        # If Net Power < 0 (Deficit) -> Discharge (0)
        # If Net Power ~ 0 -> Hold (1)
        
        # Use the same system size as dashboard (30 panels)
        generation = row['Generated_Power'] * 30 
        consumption = row['Grid_Consumption']
        net_power = generation - consumption
        
        if net_power > 10: # Surplus
            return 2 # Charge
        elif net_power < -10: # Deficit
            return 0 # Discharge
        else:
            return 1 # Hold
            
    data['Signal'] = data.apply(get_balancing_signal, axis=1)
    
    # DEBUG: Check Label Distribution
    print("\n--- DEBUG: Label Distribution ---")
    print("Low SoC (< 25%) Signals:")
    print(data[data['Battery_SoC'] < 25]['Signal'].value_counts())
    print("\nNight Time (Irradiance < 10) Signals:")
    print(data[data['Irradiance'] < 10]['Signal'].value_counts())
    print("---------------------------------")
    
    X_train_clf, X_test_clf, y_train_clf, y_test_clf, scaler_clf, feature_cols_clf = prepare_datasets(
        data, 'Signal', is_classification=True
    )
    
    model_clf = train_balancing_classifier(X_train_clf, y_train_clf)
    
    # Evaluate
    y_pred_clf = model_clf.predict(X_test_clf)
    acc = accuracy_score(y_test_clf, y_pred_clf)
    print(f"Balancing Classifier Accuracy: {acc:.4f}")
    print(classification_report(y_test_clf, y_pred_clf))
    
    # Save model and scaler
    joblib.dump(model_clf, 'models/balancing_classifier.pkl')
    joblib.dump(scaler_clf, 'models/scaler_clf.pkl')
    
    print("Training Complete.")

    # 3. Train Grid Stability Monitor (Real Data)
    print("\nTraining Grid Stability Monitor (Real Data)...")
    stability_data = load_stability_data('dataset')
    
    if not stability_data.empty:
        # Prepare Data
        # Target: Fault_Type (We treat F0 as Stable, F1-F7 as Unstable/Faulty)
        # Features: All columns ending in _mean, _std, _max, _min
        
        feature_cols_stability = [c for c in stability_data.columns if c.endswith(('_mean', '_std', '_max', '_min'))]
        X_stability = stability_data[feature_cols_stability]
        y_stability = stability_data['Fault_Type']
        
        # Split
        from sklearn.model_selection import train_test_split
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_stability, y_stability, test_size=0.2, random_state=42)
        
        # Train
        model_stability = train_stability_monitor(X_train_s, y_train_s)
        
        # Evaluate
        y_pred_s = model_stability.predict(X_test_s)
        acc_s = accuracy_score(y_test_s, y_pred_s)
        print(f"Grid Stability Monitor Accuracy: {acc_s:.4f}")
        print(classification_report(y_test_s, y_pred_s))
        
        # Save
        joblib.dump(model_stability, 'models/stability_monitor.pkl')
        # Save feature names for inference
        joblib.dump(feature_cols_stability, 'models/stability_features.pkl')
    else:
        print("No stability data found. Skipping Stability Monitor training.")

if __name__ == "__main__":
    train_and_evaluate()
