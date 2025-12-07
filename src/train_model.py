import sys
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from data_loader import generate_synthetic_data, get_real_world_stats, load_stability_data
from preprocessing import preprocess_features, prepare_datasets
from model import train_energy_predictor, train_balancing_classifier, train_stability_monitor

import argparse

def plot_classification_metrics(model, X_test, y_test, model_name, feature_names=None, output_dir='output'):
    """
    Generates and saves ROC, Precision-Recall, and Feature Importance plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Feature Importance
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        plt.figure(figsize=(10, 6))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.title(f'Feature Importance - {model_name}')
        plt.bar(range(X_test.shape[1]), importances[indices], align='center')
        plt.xticks(range(X_test.shape[1]), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_feature_importance.png'))
        plt.close()

    # 2. ROC and PR Curves (Multiclass)
    # Binarize the output
    classes = np.unique(y_test)
    n_classes = len(classes)
    y_test_bin = label_binarize(y_test, classes=classes)
    
    # Get probabilities
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test)
    else:
        print(f"Model {model_name} does not support predict_proba, skipping ROC/PR curves.")
        return

    # ROC Curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_roc.png'))
    plt.close()

    # Precision-Recall Curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

    plt.figure(figsize=(10, 8))
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'PR curve of class {classes[i]} (AP = {average_precision[i]:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_pr.png'))
    plt.close()
    
    # 3. Confusion Matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'))
    plt.close()

def train_and_evaluate(force=False):
    # Check if models exist
    required_files = [
        'models/energy_predictor.pkl',
        'models/balancing_classifier.pkl',
        'models/scaler_reg.pkl',
        'models/scaler_clf.pkl'
    ]
    
    if not force and all(os.path.exists(f) for f in required_files):
        print("Models already exist. Skipping training. Use --force to override.")
        return

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
    
    # Plot Feature Importance for Regressor
    if hasattr(model_reg, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        importances = model_reg.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.title('Feature Importance - Energy Predictor')
        plt.bar(range(X_test_reg.shape[1]), importances[indices], align='center')
        plt.xticks(range(X_test_reg.shape[1]), [feature_cols_reg[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        os.makedirs('output', exist_ok=True)
        plt.savefig('output/energy_predictor_feature_importance.png')
        plt.close()
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_reg, 'models/energy_predictor.pkl')
    joblib.dump(scaler_reg, 'models/scaler_reg.pkl')
    
    # 2. Train Balancing Classifier
    # Target: 0 (Discharge), 1 (Hold), 2 (Charge)
    
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
    
    # Plot Advanced Metrics
    plot_classification_metrics(model_clf, X_test_clf, y_test_clf, "Balancing Classifier", feature_names=feature_cols_clf)
    
    # Save model and scaler
    joblib.dump(model_clf, 'models/balancing_classifier.pkl')
    joblib.dump(scaler_clf, 'models/scaler_clf.pkl')
    
    print("Training Complete.")

    # 3. Train Grid Stability Monitor (Real Data)
    print("\nTraining Grid Stability Monitor (Real Data)...")
    # MATCH INFERENCE WINDOW: Inference uses a buffer of 20 steps.
    # Training must use the same window size to learn the correct feature distributions (std, min, max).
    stability_data = load_stability_data('dataset', window_size=20)
    
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
        
        # Plot Advanced Metrics
        plot_classification_metrics(model_stability, X_test_s, y_test_s, "Grid Stability Monitor", feature_names=feature_cols_stability)
        
        # Save
        joblib.dump(model_stability, 'models/stability_monitor.pkl')
        # Save feature names for inference
        joblib.dump(feature_cols_stability, 'models/stability_features.pkl')
    else:
        print("No stability data found. Skipping Stability Monitor training.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='Force retraining even if models exist')
    args = parser.parse_args()
    
    train_and_evaluate(force=args.force)
