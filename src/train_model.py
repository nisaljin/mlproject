import sys
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from data_loader import generate_synthetic_data, get_real_world_stats
from preprocessing import preprocess_features, prepare_datasets
from model import train_energy_predictor, train_balancing_classifier

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
        generation = row['Irradiance'] * 0.2
        consumption = row['Grid_Consumption'] * 0.1
        soc = row['Battery_SoC']
        
        if soc < 20:
            return 2 # Charge (Priority)
        elif soc > 80:
            return 0 # Discharge (Priority)
        elif abs(generation - consumption) < 20: # Threshold for balance
            return 1 # Hold
        elif generation > consumption:
            return 2 # Charge (Excess energy)
        else:
            return 0 # Discharge (Deficit)
            
    # Simplified logic for 3 classes
    # 0: Discharge, 1: Hold, 2: Charge
    conditions = [
        (data['Battery_SoC'] < 30), # Low battery -> Charge
        (data['Battery_SoC'] > 80), # High battery -> Discharge
        (data['Irradiance'] > 500)  # High generation -> Charge
    ]
    choices = [2, 0, 2]
    # Default to 1 (Hold/Balance) if none match, or maybe just based on net load
    # Let's use a more complex logic
    
    data['Signal'] = data.apply(get_balancing_signal, axis=1)
    
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

if __name__ == "__main__":
    train_and_evaluate()
