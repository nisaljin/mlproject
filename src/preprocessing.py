import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features: Hour, Minute, DayOfWeek.
    """
    df['Hour'] = df['Timestamp'].dt.hour
    df['Minute'] = df['Timestamp'].dt.minute
    # df['DayOfWeek'] = df['Timestamp'].dt.dayofweek # Not needed for synthetic data which is just 1 week
    return df

def add_lag_features(df: pd.DataFrame, columns: list, lags: list) -> pd.DataFrame:
    """
    Add lag features for specified columns.
    """
    for col in columns:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess features: handling missing values, scaling, etc.
    """
    # Add time features
    df = add_time_features(df)
    
    # Add lag features (e.g., previous 15 minutes)
    # We want to predict future state based on past
    # For simplicity, let's just use current state to predict "next step" or "optimal action"
    # But for a realistic model, lags are good.
    # Let's add lags for Irradiance and Consumption
    df = add_lag_features(df, ['Irradiance', 'Grid_Consumption'], [1, 5, 15])
    
    # Drop rows with missing values (due to lags)
    df = df.dropna()
    
    return df

def prepare_datasets(df: pd.DataFrame, target_col: str, test_size: float = 0.2, is_classification: bool = False):
    """
    Split data into training and testing sets.
    """
    # Features: Irradiance, Temperature, Grid_Consumption, Battery_SoC, Time features, Lags
    # Target: Depends on the model (Energy Prediction or Balancing Signal)
    
    feature_cols = [c for c in df.columns if c not in ['Timestamp', target_col]]
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols
