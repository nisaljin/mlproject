from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
import joblib
import numpy as np

def train_energy_predictor(X_train, y_train):
    """
    Train a Random Forest Regressor to predict energy generation/consumption.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_balancing_classifier(X_train, y_train):
    """
    Train an XGBoost Classifier to predict battery balancing signals.
    Target: 0 (Discharge), 1 (Hold), 2 (Charge)
    """
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)
    return model

def train_stability_monitor(X_train, y_train):
    """
    Train a Random Forest Classifier to monitor Grid Stability (Stable vs Unstable).
    """
    # Using Random Forest as it handles tabular data well and is robust
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath):
    joblib.dump(model, filepath)

def load_model(filepath):
    return joblib.load(filepath)
