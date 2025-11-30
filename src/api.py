from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import uvicorn
from typing import List

# Initialize App
app = FastAPI(
    title="Helios AI Energy API",
    description="API for predicting energy consumption and battery balancing signals.",
    version="1.0.0"
)

# Load Models (Global State)
models = {}

@app.on_event("startup")
def load_models():
    model_path = "models"
    # Check if running in Docker (paths might differ)
    if not os.path.exists(model_path):
        model_path = "../models" # Fallback
        
    try:
        models['energy_model'] = joblib.load(os.path.join(model_path, 'energy_predictor.pkl'))
        models['balancing_model'] = joblib.load(os.path.join(model_path, 'balancing_classifier.pkl'))
        models['scaler_reg'] = joblib.load(os.path.join(model_path, 'scaler_reg.pkl'))
        models['scaler_clf'] = joblib.load(os.path.join(model_path, 'scaler_clf.pkl'))
        print("✅ Models loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

# Input Schemas
class PredictionInput(BaseModel):
    Irradiance: float
    Temperature: float
    Grid_Consumption: float
    Battery_SoC: float
    Generated_Power: float
    Hour: int
    Minute: int
    # Lags (Optional - in a real system these would be fetched from a DB)
    # For simplicity, we'll accept them or default to current values if missing
    Irradiance_lag_1: float = 0.0
    Irradiance_lag_5: float = 0.0
    Irradiance_lag_15: float = 0.0
    Grid_Consumption_lag_1: float = 0.0
    Grid_Consumption_lag_5: float = 0.0
    Grid_Consumption_lag_15: float = 0.0

class PredictionOutput(BaseModel):
    predicted_consumption: float
    recommended_action: str
    action_code: int

@app.get("/")
def health_check():
    return {"status": "online", "models_loaded": len(models) == 4}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Convert input to DataFrame
    data = pd.DataFrame([input_data.dict()])
    
    # 1. Energy Prediction
    # Select features for regression
    # Note: The order must match training. 
    # We rely on the scaler to handle column ordering if we pass a DataFrame with correct names
    # But scaler expects specific columns.
    
    # Let's assume the input provides all necessary features for simplicity
    # In a robust production system, we would have a Feature Store.
    
    # Construct feature vector for Regression
    # We need to ensure columns match exactly what scaler expects
    # This is tricky without the original training column list.
    # For this demo, we'll assume the input schema matches the scaler's expectation
    # (Irradiance, Temp, Grid, SoC, Hour, Minute, Lags...)
    
    try:
        # Prepare Regression Input
        # We need to drop 'Target_Consumption' and 'Signal' and 'Timestamp' which are not in input
        X_reg = data.copy()
        X_reg_scaled = models['scaler_reg'].transform(X_reg)
        pred_consumption = models['energy_model'].predict(X_reg_scaled)[0]
        
        # Prepare Classification Input
        # Add the prediction as a feature
        X_clf = data.copy()
        X_clf['Target_Consumption'] = pred_consumption
        X_clf_scaled = models['scaler_clf'].transform(X_clf)
        pred_signal = models['balancing_model'].predict(X_clf_scaled)[0]
        
        # Map Signal
        signal_map = {0: 'Discharge', 1: 'Hold', 2: 'Charge'}
        action = signal_map.get(int(pred_signal), "Unknown")
        
        # Log Prediction Details
        print(f"🔮 Input: SoC={input_data.Battery_SoC:.1f}%, Gen={input_data.Generated_Power:.0f}W, Load={input_data.Grid_Consumption:.0f}W")
        print(f"🤖 Prediction: Action={action}, Forecast Load={pred_consumption:.0f}W")
        
        return {
            "predicted_consumption": float(pred_consumption),
            "recommended_action": action,
            "action_code": int(pred_signal)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
