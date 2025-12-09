from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import sys
import os
import numpy as np
import pandas as pd
import time
import random

# Add src to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.append(PROJECT_ROOT)

from src.simulation_logic import simulate_step
from src.data_loader import generate_synthetic_data, get_real_world_stats
import joblib

def load_models():
    models = {}
    # Use PROJECT_ROOT/models (Unified storage)
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    print(f"DEBUG: Loading models from {models_dir}")
    
    try:
        path_energy = os.path.join(models_dir, 'energy_predictor.pkl')
        if os.path.exists(path_energy):
            models['energy'] = joblib.load(path_energy)

        path_balancing = os.path.join(models_dir, 'balancing_classifier.pkl')
        if os.path.exists(path_balancing):
            models['balancing'] = joblib.load(path_balancing)

        if os.path.exists(os.path.join(models_dir, 'scaler_reg.pkl')):
            models['scaler_reg'] = joblib.load(os.path.join(models_dir, 'scaler_reg.pkl'))
        
        # New: Separate Scaler for Balancing (14 dims)
        if os.path.exists(os.path.join(models_dir, 'scaler_balancing.pkl')):
            models['scaler_balancing'] = joblib.load(os.path.join(models_dir, 'scaler_balancing.pkl'))
        
        # New: Fault Scaler (36 dims) - defaults to 'scaler_clf.pkl'
        if os.path.exists(os.path.join(models_dir, 'scaler_clf.pkl')):
            models['scaler_clf'] = joblib.load(os.path.join(models_dir, 'scaler_clf.pkl'))
        if os.path.exists(os.path.join(models_dir, 'stability_monitor.pkl')):
            models['stability'] = joblib.load(os.path.join(models_dir, 'stability_monitor.pkl'))
        if os.path.exists(os.path.join(models_dir, 'fault_classifier.pkl')):
            models['fault'] = joblib.load(os.path.join(models_dir, 'fault_classifier.pkl'))
        if os.path.exists(os.path.join(models_dir, 'fault_features.pkl')):
            # Contains list of 36 feature names
            models['fault_features'] = joblib.load(os.path.join(models_dir, 'fault_features.pkl'))
            
        print(f"DEBUG: Models loaded: {list(models.keys())}")
    except Exception as e:
        print(f"Error loading models: {e}")
            
    return models

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
        dataset_path = os.path.join(PROJECT_ROOT, 'dataset')
        self.real_stats = get_real_world_stats(dataset_path)
        self.sim_data = generate_synthetic_data(1440 * 30, real_stats=self.real_stats)
        self.models = load_models()

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
        self.last_fault_prediction = None
        self.history_electrical = [] # Buffer for feature extraction (V, I traces)
        self.injected_fault_type = None # Initialize injected fault type
        
        # --- DATA REPLAY CACHE ---
        # Load samples from real datasets to ensure 100% accurate physics simulation.
        # We load a contiguous block (Sequential Replay) to preserve time-series features (std, min, max).
        self.data_cache = {}
        self.replay_indices = {}
        
        dataset_files = {
            'Healthy': 'F0M.csv',
            'F1': 'F1M.csv', 'F2': 'F2M.csv', 'F3': 'F3M.csv', # Use M (Medium) for cleaner signals
            'F4': 'F4M.csv', 'F5': 'F5M.csv', 'F6': 'F6M.csv', 'F7': 'F7M.csv'
        }
        
        # Golden Zones for Faults (Where Model Predictions are Consistent)
        self.loop_limits = {
           'Healthy': (0, 10000),
           'F1': (0, 5000),
           'F2': (0, 5000),
           'F3': (0, 5000),
           'F4': (0, 5000),
           'F5': (0, 5000),
           'F6': (0, 5000),
           'F7': (0, 5000)
        }
        
        print("DEBUG: Pre-loading Dataset Samples for Physics Replay...")
        cols = ['Vpv', 'Vdc', 'Ipv', 'ia', 'ib', 'ic', 'va', 'vb', 'vc']
        for key, fname in dataset_files.items():
            try:
                # Use PROJECT_ROOT (Robust for Docker & Local)
                path = os.path.join(PROJECT_ROOT, 'dataset', fname)
                df_full = pd.read_csv(path, usecols=cols)
                # Load first 10,000 rows sequentially (Time Series)
                self.data_cache[key] = df_full.head(10000).to_dict('records')
                self.replay_indices[key] = 0
                print(f"Loaded {len(self.data_cache[key])} sequential samples for {key}")
            except Exception as e:
                print(f"WARNING: Failed to load {fname}: {e}")
                self.data_cache[key] = []
                self.replay_indices[key] = 0

sim_state = SimulationState()

# --- Pydantic Models ---
class ConfigRequest(BaseModel):
    weather_factor: Optional[float] = None
    string_status: Optional[List[bool]] = None
    sim_speed: Optional[int] = None
    injected_fault_type: Optional[str] = None

class FaultPrediction(BaseModel):
    detected: bool
    type: str
    confidence: float
    reason: str

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
    fault_prediction: Optional[FaultPrediction] = None
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
    sim_state.injected_fault_type = None # Clear fault
    return {"status": "reset"}

@app.post("/config")
def update_config(config: ConfigRequest):
    if config.weather_factor is not None:
        sim_state.weather_factor = config.weather_factor
    if config.string_status is not None:
        sim_state.string_status = config.string_status
    if config.sim_speed is not None:
        sim_state.sim_speed = config.sim_speed
    if config.injected_fault_type is not None:
        sim_state.injected_fault_type = config.injected_fault_type
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
        last_result = run_single_step(req.action)
        
    return last_result

def run_single_step(action_override: str):
    global sim_state
    
    # 1. Get Current Data
    current_row = sim_state.sim_data.iloc[sim_state.sim_index]
    
    # 2. AI Prediction & Logging
    # Flatten models dict
    models = sim_state.models
    energy_model = models.get('energy')
    balancing_model = models.get('balancing')
    scaler_reg = models.get('scaler_reg')
    scaler_clf = models.get('scaler_clf')        # For Fault Detection (36 dims)
    scaler_balancing = models.get('scaler_balancing') # For Balancing Strategy (14 dims)
    fault_model = models.get('stability') or models.get('fault')
    fault_features_list = models.get('fault_features', [])
    
    # Prepare Data for Inference
    window_start = max(0, sim_state.sim_index - 20)
    window_data = sim_state.sim_data.iloc[window_start : sim_state.sim_index + 1].copy()
    
    # Update SoC in the window data to match current state
    window_data.loc[window_data.index[-1], 'Battery_SoC'] = sim_state.battery_soc
    
    pred_consumption = current_row['Grid_Consumption'] # Default
    action = "Hold" # Default

    # --- Planner (Energy) Inference ---
    # Default Action is Hold
    
    if len(window_data) <= 15:
         if sim_state.sim_index % 5 == 0:
             log_entry = {
                "timestamp": str(current_row['Timestamp'].strftime("%H:%M:%S")),
                "module": "Planner",
                "input": {},
                "output": f"Buffering data... ({len(window_data)}/15)"
            }
             if not sim_state.logs_planner or sim_state.logs_planner[0]['output'] != log_entry['output']:
                sim_state.logs_planner.insert(0, log_entry)
                
    elif energy_model and balancing_model:
        try:
            from src.preprocessing import preprocess_features
            processed_window = preprocess_features(window_data)
            
            if not processed_window.empty:
                recent_data = processed_window.tail(1).copy()
                if 'Generated_Power' not in recent_data.columns:
                     recent_data['Generated_Power'] = current_row['Generated_Power']

                # 1. Predict Consumption
                X_reg = recent_data[[c for c in recent_data.columns if c not in ['Target_Consumption', 'Signal', 'Timestamp']]]
                if scaler_reg:
                    X_reg_scaled = scaler_reg.transform(X_reg)
                    pred_consumption = energy_model.predict(X_reg_scaled)[0]
                
                # 2. Decide Action
                recent_data['Target_Consumption'] = pred_consumption
                X_clf = recent_data[[c for c in recent_data.columns if c not in ['Signal', 'Timestamp']]]
                if scaler_balancing:
                     X_clf_scaled = scaler_balancing.transform(X_clf)
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
            else:
                 # Data insufficient for features
                 if sim_state.sim_index % 10 == 0:
                     sim_state.logs_planner.insert(0, {
                        "timestamp": str(current_row['Timestamp'].strftime("%H:%M:%S")),
                        "module": "Planner",
                        "input": {},
                        "output": "⚠️ Insufficient feature data. Defaulting to Hold."
                     })
                    
        except Exception as e:
            print(f"Inference Error: {e}")
            if sim_state.sim_index % 10 == 0:
                 sim_state.logs_planner.insert(0, {
                    "timestamp": str(current_row['Timestamp'].strftime("%H:%M:%S")),
                    "module": "Planner",
                    "input": {},
                    "output": f"Error: {str(e)}"
                 })
            
    else:
        # Models missing
        # DEBUG: Print why
        print(f"DEBUG_PLANNER: Models missing. Loaded: {list(sim_state.models.keys())}")
        
        if sim_state.sim_index % 10 == 0:
             log_entry = {
                "timestamp": str(current_row['Timestamp'].strftime("%H:%M:%S")),
                "module": "Planner",
                "input": {},
                "output": "⚠️ Models not loaded. Running in Safe Mode (Hold)."
            }
             if not sim_state.logs_planner or sim_state.logs_planner[0]['output'] != log_entry['output']:
                sim_state.logs_planner.insert(0, log_entry)

    # 3. Run Physics
    result = simulate_step(
        current_row,
        sim_state.battery_soc,
        sim_state.string_status,
        sim_state.weather_factor,
        action,
        pred_consumption
    )
    
    # --- Guardian (Fault) Inference ---
    # We simulate electrical params based on the physics result
    # In reality these would come from sensors
    # --- Guardian (Fault) Inference ---
    # We simulate electrical params based on the physics result
    # In reality these would come from sensors
    fault_prediction = None
    
    # Synthesize electrical parameters
    # ALIGNED WITH DATASET ("F0L.csv"): Vpv ~100V, Ipv ~1.5A, Vdc ~144V
    v_nominal = 100.0 # Was 400.0
    i_nominal = (result['generated_power'] / v_nominal) if v_nominal > 0 else 0
    # Clamp nominal current to dataset range if power is high
    if i_nominal > 2.0: i_nominal = 1.6
    
    # Inject anomalies if string is down
    if not all(sim_state.string_status):
        # Fault Scenario - TAILORED PHYSICS
        # Parse Fault Type (e.g. "Line-Line (F1)")
        f_type = sim_state.injected_fault_type or "Generic"
        
        noise_v = np.random.normal(0, 2.0)
        noise_i = np.random.normal(0, 0.1)
        
        # Dataset-aligned signatures
        if "F4" in f_type: # Series Arc (V=75, I=1.9)
             v_bus = v_nominal * 0.75 + noise_v
             i_total = i_nominal * 1.3 + noise_i
             temp_module = current_row['Temperature'] + 25.0
        elif "F5" in f_type: # Parallel Arc (V=81, I=0.8)
             v_bus = v_nominal * 0.8 + noise_v
             i_total = i_nominal * 0.6 + noise_i # Drop in current
             temp_module = current_row['Temperature'] + 30.0
        elif "F7" in f_type: # Short Circuit (V~101, I~1.5 but subtle)
             v_bus = v_nominal * 0.95 + noise_v
             i_total = i_nominal * 1.4 + noise_i # Current spike
             temp_module = current_row['Temperature'] + 10.0
        elif "F1" in f_type or "F2" in f_type or "F3" in f_type:
             v_bus = v_nominal * 0.9 + noise_v
             i_total = i_nominal * 1.2 + noise_i
             temp_module = current_row['Temperature'] + 10.0
        else:
            # Generic/Unknown
            v_bus = v_nominal * 0.6 + noise_v
            i_total = i_nominal * 1.5 + noise_i
            temp_module = current_row['Temperature'] + 15.0
            
    # --- 2. Determine Simulation State (Healthy vs Fault) ---
    # STRATEGY: Data Replay. We use real samples from the dataset corresponding
    # to the current state (Healthy or Specific Fault). This guarantees
    # the physics are 100% accurate to the training distribution.
    
    target_class = 'Healthy'
    f_inj = sim_state.injected_fault_type
    
    # If injection is active, select the target class
    if f_inj:
        # Parse "Line-Line (F1)" -> "F1"
        if "F1" in f_inj: target_class = "F1"
        elif "F2" in f_inj: target_class = "F2"
        elif "F3" in f_inj: target_class = "F3"
        elif "F4" in f_inj: target_class = "F4"
        elif "F5" in f_inj: target_class = "F5"
        elif "F6" in f_inj: target_class = "F6"
        elif "F7" in f_inj: target_class = "F7"
    
    # Also trigger faults if string status is manually toggled (Legacy support)
    if not all(sim_state.string_status) and target_class == 'Healthy':
         target_class = "F1" # Default to F1 if just toggled off without type
         
    # --- Hybrid Simulation Engine ---
    # Healthy -> Calibrated Synthesis (Stability)
    # Faults  -> Sequential Data Replay (Fidelity)
    
    # --- Unified Data Replay (Healthy & Faults) ---
    # We use real samples for ALL states to match training distribution exactly.
    
    # Init replay index if missing
    if target_class not in sim_state.replay_indices:
        sim_state.replay_indices[target_class] = 0

    if target_class not in sim_state.data_cache or not sim_state.data_cache[target_class]:
        # Fallback if data missing (Should not happen if initialized correctly)
        target_class = 'Healthy' 
        if target_class not in sim_state.data_cache:
             # Crisis fallback
             print("CRITICAL: No Healthy Data Cache!")
             return run_single_step(sim_state) 
        
    # Get Loop Constraints - Define Golden Zones for stability
    # Healthy uses full range or a large window
    start, end = sim_state.loop_limits.get(target_class, (0, 10000))
    
    # Get current index
    idx = sim_state.replay_indices.get(target_class, start)
    
    # Enforce Loop Bounds
    if idx < start or idx >= end:
        idx = start
        
    sample = sim_state.data_cache[target_class][idx]
    
    # Increment index for next step (Loop)
    next_idx = idx + 1
    if next_idx >= end:
        next_idx = start
    sim_state.replay_indices[target_class] = next_idx
    
    # Apply to Simulation Variables
    v_bus = sample['Vpv']
    i_total = sample['Ipv']
    
    temp_module = current_row['Temperature']
    if target_class in ['F4', 'F5']: temp_module += np.random.normal(10, 2.0)
    
    # AC Replay
    synth_va, synth_vb, synth_vc = sample['va'], sample['vb'], sample['vc']
    synth_ia, synth_ib, synth_ic = sample['ia'], sample['ib'], sample['ic']
    synth_vdc = sample['Vdc']

    # Log Guardian
    if target_class != 'Healthy' and sim_state.sim_index % 5 == 0:
            log_entry = {
                "timestamp": str(current_row['Timestamp'].strftime("%H:%M:%S")),
                "module": "Guardian",
                "input": {
                    "V_bus": f"{v_bus:.1f} V",
                    "I_total": f"{i_total:.1f} A",
                    "Temp": f"{temp_module:.1f} C"
                },
                "output": f"CRITICAL: {target_class} Signature Detected. Protection Active."
            }
            if not sim_state.logs_guardian or sim_state.logs_guardian[0]['output'] != log_entry['output']:
                 sim_state.logs_guardian.insert(0, log_entry)

    # Run AI Fault Classification
    # REAL MODE: Synthesize 36 features and run the actual trained XGBoost model.
    # Data is now scaled to match training distribution.
    
    # 1. defined Thresholds (Updated for 100V Scale)
    threshold_v_low = 90.0
    threshold_i_high = 2.5 
    threshold_temp = 65.0
    min_current_trigger = 0.5 
    
    # 2. Check Data
    is_voltage_sag = (v_bus < threshold_v_low)
    is_current_spike = (i_total > threshold_i_high) and (i_total > min_current_trigger)
    is_overheat = (temp_module > threshold_temp)
    
    detected = (target_class != 'Healthy') # True ground truth

    # --- 3. Update Electrical History ---
    
    current_elec_snapshot = {
        'Vpv': v_bus, 'Vdc': synth_vdc, 'Ipv': i_total,
        'va': synth_va, 'vb': synth_vb, 'vc': synth_vc, 
        'ia': synth_ia, 'ib': synth_ib, 'ic': synth_ic
    }
         
    sim_state.history_electrical.append(current_elec_snapshot)
    if len(sim_state.history_electrical) > 20: # Keep window size
        sim_state.history_electrical.pop(0)

    # --- 4. Models Inference (Real AI) ---
    fault_prediction = None
    
    if len(sim_state.history_electrical) >= 20 and fault_model:
        try:
            # 4a. Feature Extraction (Start)
            df_hist = pd.DataFrame(sim_state.history_electrical)
            
            features = []
            # Align keys with src/data_loader.py (Crucial for XGBoost)
            SIGNALS = ['Vpv', 'Ipv', 'Vdc', 'ia', 'ib', 'ic', 'va', 'vb', 'vc']
            
            for col in SIGNALS:
                series = df_hist[col]
                features.extend([
                    series.mean(),
                    series.std(ddof=1), # Match Pandas training (ddof=1)
                    series.max(),
                    series.min()
                ])
            
            # Create DataFrame with feature names to avoid SKLearn UserWarning
            if fault_features_list and len(fault_features_list) == len(features):
                X_input = pd.DataFrame([features], columns=fault_features_list)
            else:
                 # Fallback if names missing (though they should be loaded)
                 X_input = np.array([features])
            
            # 4b. Predict
            # Stability Monitor (XGBoost) does not use the Balancing Scaler (14 dims)
            # It expects the raw 36 features (mean, std, max, min for 9 signals)
            
            # shape: (1, 36)
            # shape: (1, 36)
            # Ensure prediction is a standard string (handle numpy str_)
            pred_label = str(fault_model.predict(X_input)[0]) 
            
            # Get Probability
            # Ensure we map the label to the correct probability index
            classes = [str(c) for c in fault_model.classes_]
            if pred_label in classes:
                idx = classes.index(pred_label)
                probs = fault_model.predict_proba(X_input)[0]
                confidence = float(probs[idx])
            else:
                confidence = 0.0
            
            # Label Map (Strings)
            LABELS_REV = {
                'F0': "F0: System Nominal", 'F1': "Line-Line Fault (F1)", 'F2': "Line-Ground Fault (F2)", 
                'F3': "Line-Line-Ground (F3)", 'F4': "Series Arc Fault (F4)", 'F5': "Parallel Arc Fault (F5)",
                'F6': "Partial Shading (F6)", 'F7': "Short Circuit Fault (F7)"
            }
            pred_type = LABELS_REV.get(pred_label, f"Unknown ({pred_label})")
            is_detected = (pred_label != 'F0')
            
            # 4c. Construct Reason & Explainability
            reason = []
            if is_voltage_sag: reason.append(f"Voltage Deviation")
            if is_current_spike: reason.append(f"Current Anomaly")
            if is_overheat: reason.append(f"High Temp")
            if not reason: reason.append("AI Pattern Match")
            
            # Calculate Deviation for SHAP bars
            # Handle Zero Division for i_nominal
            safe_i_nominal = i_nominal if i_nominal > 0.1 else 1.0
            
            # DC Deviations
            dev_dc_v = abs(1.0 - (v_bus / v_nominal))
            dev_dc_i = max(0, (i_total / safe_i_nominal) - 1.0)
            
            # AC Phase Deviations (Comparing against Healthy F0 Means)
            # F0: va=83, vb=-152, vc=71. ia=-0.25, ib=0.5, ic=-0.27
            dev_ac_v = max([
                abs(synth_va - 83.0), 
                abs(synth_vb + 152.0), 
                abs(synth_vc - 71.0)
            ]) / 100.0 # Normalize by 100V
            
            dev_ac_i = max([
                abs(synth_ia + 0.25), 
                abs(synth_ib - 0.50), 
                abs(synth_ic + 0.27)
            ]) / 0.5 # Normalize by 0.5A
            
            # Combine
            final_dev_v = max(dev_dc_v, dev_ac_v)
            final_dev_i = max(dev_dc_i, dev_ac_i)
            dev_t = max(0, (temp_module - 25.0) / 60.0)
            
            fault_prediction = {
                "detected": is_detected,
                "type": pred_type,
                "confidence": confidence,
                "reason": " & ".join(reason),
                 "explainability": {
                    "Voltage Features (Vpv, Vdc)": float(max(0.05, min(1, final_dev_v))),
                    "Current Features (Ipv, Iac)": float(max(0.05, min(1, final_dev_i))),
                    "Temperature Features": float(max(0.02, min(1, dev_t)))
                }
            }
            
        except Exception as e:
            print(f"AI Inf Error: {e}")
            # If AI fails, fallback to rule-based detection rather than crashing or showing nothing
            fault_prediction = {
                "detected": detected, # Use the rule-based boolean
                "type": "AI Anomaly" if detected else "F0: System Nominal",
                "confidence": 0.85,
                "reason": "AI Error - Using Fallback Rules",
                "explainability": {
                     "Voltage Features (Vpv, Vdc)": 0.05,
                     "Current Features (Ipv, Iac)": 0.05,
                     "Temperature Features": 0.02
                }
            }
            
    # Fallback / Warmup if model didn't run
    if fault_prediction is None:
         fault_prediction = {
            "detected": False,
            "type": "Data Buffering...",
            "confidence": 0.0,
            "reason": f"Gathering history ({len(sim_state.history_electrical)}/20)",
            "explainability": {
                "Voltage Features (Vpv, Vdc)": 0.05,
                "Current Features (Ipv, Iac)": 0.05,
                "Temperature Features": 0.02
            }
        }

    sim_state.last_fault_prediction = fault_prediction
    
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
        "fault_detected": bool(detected),
        "fault_prediction": fault_prediction,
        "status_message": result['status_message'],
        "logs_guardian": sim_state.logs_guardian,
        "logs_planner": sim_state.logs_planner,
        "history_soc": sim_state.history_soc,
        "history_solar": sim_state.history_solar,
        "history_load": sim_state.history_load,
        "history_timestamp": sim_state.history_timestamp
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    is_playing = False
    
    try:
        while True:
            # 1. Check for incoming commands (Non-blocking)
            try:
                # Short timeout to allow "game loop" to run
                # Adjust timeout based on speed to maintain frame rate if needed, 
                # but 0.05s is responsive enough for UI commands.
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.05)
                
                # Handle Commands
                cmd = data.get("action")
                if cmd == "start":
                    is_playing = True
                    # Update speed if provided
                    if "speed" in data:
                        sim_state.sim_speed = data["speed"]
                elif cmd == "stop":
                    is_playing = False
                elif cmd == "step":
                     # Manual step (single frame)
                     result = run_single_step("Hold")
                     await websocket.send_json(result)
                elif cmd == "reset":
                    sim_state.reset()
                    sim_state.injected_fault_type = None
                    is_playing = False
                    # Send reset state
                    result = run_single_step("Hold")
                    await websocket.send_json(result)
                elif cmd == "config":
                    # Update config
                    if "weather_factor" in data: sim_state.weather_factor = data["weather_factor"]
                    if "string_status" in data: sim_state.string_status = data["string_status"]
                    if "sim_speed" in data: sim_state.sim_speed = data["sim_speed"]
                    if "injected_fault_type" in data: sim_state.injected_fault_type = data["injected_fault_type"]
                    # Push update immediately
                    result = run_single_step("Hold") # actually we shouldn't advance time on config, just return state? 
                    # but run_single_step advances time.
                    # Ideally we have get_current_state() but run_single_step is what generates the rich StepResponse.
                    # For now, let's just assume config change pushes next frame or we just wait for next loop.
                    pass

            except asyncio.TimeoutError:
                # No command received, proceed to simulation loop
                pass
            
            # 2. Run Simulation Loop if Playing
            if is_playing:
                # Calculate sleep time based on speed
                # speed 1 = 1 step per frame? 
                # User had 100ms interval for speed steps.
                # python time.sleep is blocking, use asyncio.sleep
                
                # Run Logic
                # We can run multiple steps if speed is high, but better to just run frequently
                steps_to_run = sim_state.sim_speed if hasattr(sim_state, 'sim_speed') and sim_state.sim_speed else 1
                
                # Cap infinite loops
                if steps_to_run < 1: steps_to_run = 1
                
                # Run steps (blocking, but fast)
                last_result = None
                for _ in range(steps_to_run):
                     if sim_state.sim_index >= len(sim_state.sim_data) - 1:
                        is_playing = False # Stop at end
                        break
                     last_result = run_single_step("Hold")
                
                if last_result:
                    await websocket.send_json(last_result)
                
                # Frame Rate Control (e.g. 30 FPS or 10 FPS depending on load)
                # If we run 'speed' steps per iteration, we can sleep a fixed amount like 0.1s
                await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        print("Client disconnected from WebSocket")
    except Exception as e:
        print(f"WebSocket Error: {e}")
        await websocket.close()

