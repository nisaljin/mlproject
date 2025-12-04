from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys
import os
import numpy as np
import pandas as pd
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulation_logic import simulate_step
from src.data_loader import generate_synthetic_data, get_real_world_stats
from src.simulation_logic import simulate_step
from src.data_loader import generate_synthetic_data, get_real_world_stats
import joblib

def load_models():
    if os.path.exists('models/energy_predictor.pkl') and os.path.exists('models/balancing_classifier.pkl'):
        energy_model = joblib.load('models/energy_predictor.pkl')
        balancing_model = joblib.load('models/balancing_classifier.pkl')
        scaler_reg = joblib.load('models/scaler_reg.pkl')
        scaler_clf = joblib.load('models/scaler_clf.pkl')
        
        stability_model = None
        if os.path.exists('models/stability_monitor.pkl'):
            stability_model = joblib.load('models/stability_monitor.pkl')
            
        return energy_model, balancing_model, scaler_reg, scaler_clf, stability_model
    return None, None, None, None, None

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- State Management ---
class SimulationState:
    def __init__(self):
        self.reset()
        self.real_stats = get_real_world_stats('dataset')
        self.sim_data = generate_synthetic_data(1440 * 30, real_stats=self.real_stats)
        self.models = load_models() # (energy, balancing, scaler_reg, scaler_clf, stability)

    def reset(self):
        self.sim_index = 0
        self.battery_soc = 50.0
        self.string_status = [True, True, True]
        self.weather_factor = 1.0
        self.fault_detected = False
        self.logs_guardian = [{
            "timestamp": "00:00:00",
            "module": "Guardian",
            "input": {},
            "output": "System Initialized. Monitoring active."
        }]
        self.logs_planner = [{
            "timestamp": "00:00:00",
            "module": "Planner",
            "input": {},
            "output": "Planner Online. Waiting for data buffer..."
        }]
        self.history_soc = []
        self.history_solar = []
        self.history_load = []
        self.history_timestamp = []
        self.total_savings = 0.0
        self.co2_saved = 0.0
        self.stability_history = []

sim_state = SimulationState()

# --- Pydantic Models ---
class ConfigRequest(BaseModel):
    weather_factor: Optional[float] = None
    string_status: Optional[List[bool]] = None
    sim_speed: Optional[int] = None

class StepResponse(BaseModel):
    timestamp: str
    generated_power: float
    grid_consumption: float
    predicted_load: float
    net_power: float
    battery_soc: float
    energy_flow: float
    grid_net: float
    action: str
    fault_detected: bool
    status_message: str
    logs_guardian: List[Dict]
    logs_planner: List[Dict]
    history_soc: List[float]
    history_solar: List[float]
    history_load: List[float]
    history_timestamp: List[str]

# --- Endpoints ---

@app.get("/state")
def get_state():
    return {
        "sim_index": sim_state.sim_index,
        "battery_soc": sim_state.battery_soc,
        "string_status": sim_state.string_status,
        "weather_factor": sim_state.weather_factor,
        "fault_detected": sim_state.fault_detected
    }

@app.post("/reset")
def reset_simulation():
    sim_state.reset()
    return {"status": "reset"}

@app.post("/config")
def update_config(config: ConfigRequest):
    if config.weather_factor is not None:
        sim_state.weather_factor = config.weather_factor
    if config.string_status is not None:
        sim_state.string_status = config.string_status
    return {"status": "updated"}

class StepRequest(BaseModel):
    action: str = "Hold"
    steps: int = 1

@app.post("/step")
def step_simulation(req: StepRequest):
    global sim_state
    
    last_result = None
    # Run simulation for 'steps' times
    for _ in range(req.steps):
        if sim_state.sim_index >= len(sim_state.sim_data) - 1:
            # If end of data, break and return current state
            break
        last_result = run_single_step(req.action)
        
    if last_result is None:
        # Should not happen unless steps=0 or end of data immediately
        # Return current state manually if needed, or just run one step to be safe
        last_result = run_single_step(req.action)
        
    return last_result

def run_single_step(action_override: str):
    global sim_state
    
    # 1. Get Current Data
    current_row = sim_state.sim_data.iloc[sim_state.sim_index]
    
    # 2. AI Prediction & Logging
    energy_model, balancing_model, scaler_reg, scaler_clf, stability_model = sim_state.models
    
    # Prepare Data for Inference
    # We need a window of data
    window_start = max(0, sim_state.sim_index - 20)
    window_data = sim_state.sim_data.iloc[window_start : sim_state.sim_index + 1].copy()
    
    # Update SoC in the window data to match current state
    window_data.loc[window_data.index[-1], 'Battery_SoC'] = sim_state.battery_soc
    
    pred_consumption = current_row['Grid_Consumption'] # Default
    action = "Hold" # Default

    # --- Planner (Energy) Inference ---
    if len(window_data) > 15 and energy_model and balancing_model:
        try:
            from src.preprocessing import preprocess_features
            processed_window = preprocess_features(window_data)
            
            if not processed_window.empty:
                recent_data = processed_window.tail(1).copy()
                if 'Generated_Power' not in recent_data.columns:
                     recent_data['Generated_Power'] = current_row['Generated_Power']

                # 1. Predict Consumption
                X_reg = recent_data[[c for c in recent_data.columns if c not in ['Target_Consumption', 'Signal', 'Timestamp']]]
                X_reg_scaled = scaler_reg.transform(X_reg)
                pred_consumption = energy_model.predict(X_reg_scaled)[0]
                
                # 2. Decide Action
                recent_data['Target_Consumption'] = pred_consumption
                X_clf = recent_data[[c for c in recent_data.columns if c not in ['Signal', 'Timestamp']]]
                X_clf_scaled = scaler_clf.transform(X_clf)
                pred_signal = balancing_model.predict(X_clf_scaled)[0]
                
                signal_map = {0: 'Discharge', 1: 'Hold', 2: 'Charge'}
                action = signal_map.get(pred_signal, "Hold")
                
                # Log Planner (Every 10 steps)
                if sim_state.sim_index % 10 == 0:
                    log_entry = {
                        "timestamp": str(current_row['Timestamp'].strftime("%H:%M:%S")),
                        "module": "Planner",
                        "input": X_clf.iloc[0].to_dict(),
                        "output": f"Action: {action}, Load Pred: {int(pred_consumption)}W"
                    }
                    sim_state.logs_planner.insert(0, log_entry)
                    if len(sim_state.logs_planner) > 50: sim_state.logs_planner.pop()
                    
        except Exception as e:
            print(f"Inference Error: {e}")
            
    elif len(window_data) <= 15 and sim_state.sim_index % 5 == 0:
         # Log that we are buffering
         log_entry = {
            "timestamp": str(current_row['Timestamp'].strftime("%H:%M:%S")),
            "module": "Planner",
            "input": {},
            "output": f"Buffering data... ({len(window_data)}/15)"
        }
         # Avoid duplicate
         if not sim_state.logs_planner or sim_state.logs_planner[0]['output'] != log_entry['output']:
            sim_state.logs_planner.insert(0, log_entry)

    if not (energy_model and balancing_model) and sim_state.sim_index % 10 == 0:
         log_entry = {
            "timestamp": str(current_row['Timestamp'].strftime("%H:%M:%S")),
            "module": "Planner",
            "input": {},
            "output": "⚠️ Models not loaded. Using fallback logic."
        }
         if not sim_state.logs_planner or sim_state.logs_planner[0]['output'] != log_entry['output']:
            sim_state.logs_planner.insert(0, log_entry)

    # --- Guardian (Stability) Inference ---
    # Check for faults if we are in a fault injection scenario (simulated by checking if string status changed)
    # Or just run it periodically. For demo, let's run it if we are injecting a fault or randomly.
    # Actually, the fault injection in Sidebar trips the string directly. 
    # But to show the "Guardian" log, we should simulate the detection of that state.
    

    
    # 3. Run Physics

    # 3. Run Physics
    result = simulate_step(
        current_row,
        sim_state.battery_soc,
        sim_state.string_status,
        sim_state.weather_factor,
        action,
        pred_consumption
    )

    # --- Guardian (Stability) Inference ---
    # Moved after physics to access 'result'
    if not all(sim_state.string_status): # If any string is down
         if sim_state.sim_index % 5 == 0: # Log occasionally during fault
            # Calculate realistic electrical values
            # Assume 400V nominal bus voltage
            v_pv = 400.0 + np.random.uniform(-10, 10) 
            # Current = Power / Voltage. If fault, current is erratic or high before isolation.
            # Let's show the "Fault" condition (e.g. high current or voltage dip)
            i_pv = (result['generated_power'] / v_pv) if v_pv > 0 else 0
            
            # Identify which string is down
            down_idx = [i+1 for i, s in enumerate(sim_state.string_status) if not s]
            
            log_entry = {
                "timestamp": str(current_row['Timestamp'].strftime("%H:%M:%S")),
                "module": "Guardian",
                "input": {
                    "V_bus": f"{v_pv:.1f} V",
                    "I_total": f"{i_pv:.1f} A",
                    "Fault_Type": "Line-Line (Simulated)",
                    "Strings_Status": sim_state.string_status
                },
                "output": f"CRITICAL: Isolating String {down_idx}. Protection Relay OPEN."
            }
            # Avoid duplicate top log
            if not sim_state.logs_guardian or sim_state.logs_guardian[0]['output'] != log_entry['output']:
                 sim_state.logs_guardian.insert(0, log_entry)
    
    # 4. Update State
    sim_state.battery_soc = result['next_soc']
    sim_state.sim_index += 1
    
    # History
    sim_state.history_soc.append(sim_state.battery_soc)
    sim_state.history_solar.append(result['generated_power'])
    sim_state.history_load.append(result['grid_consumption'])
    sim_state.history_timestamp.append(current_row['Timestamp'].strftime("%H:%M"))
    if len(sim_state.history_soc) > 100:
        sim_state.history_soc.pop(0)
        sim_state.history_solar.pop(0)
        sim_state.history_load.pop(0)
        sim_state.history_timestamp.pop(0)

    return {
        "timestamp": str(current_row['Timestamp']),
        "generated_power": result['generated_power'],
        "grid_consumption": result['grid_consumption'],
        "predicted_load": float(pred_consumption),
        "net_power": result['net_power'],
        "battery_soc": sim_state.battery_soc,
        "energy_flow": result['energy_flow'],
        "grid_net": result['grid_net'],
        "action": result['action_taken'],
        "fault_detected": result['fault_detected'],
        "status_message": result['status_message'],
        "logs_guardian": sim_state.logs_guardian, # TODO: Populate real logs
        "logs_planner": sim_state.logs_planner,
        "history_soc": sim_state.history_soc,
        "history_solar": sim_state.history_solar,
        "history_load": sim_state.history_load,
        "history_timestamp": sim_state.history_timestamp
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
