import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
from datetime import timedelta
from data_loader import generate_synthetic_data, get_real_world_stats
from preprocessing import preprocess_features

# Page config
st.set_page_config(
    page_title="GCPBBB Smart Grid Optimization",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Setup & Loading ---
@st.cache_resource
def load_data_stats():
    return get_real_world_stats('dataset')

@st.cache_resource
def load_models():
    if os.path.exists('models/energy_predictor.pkl') and os.path.exists('models/balancing_classifier.pkl'):
        energy_model = joblib.load('models/energy_predictor.pkl')
        balancing_model = joblib.load('models/balancing_classifier.pkl')
        scaler_reg = joblib.load('models/scaler_reg.pkl')
        scaler_clf = joblib.load('models/scaler_clf.pkl')
        return energy_model, balancing_model, scaler_reg, scaler_clf
    return None, None, None, None

real_stats = load_data_stats()
energy_model, balancing_model, scaler_reg, scaler_clf = load_models()

# --- 2. Session State Initialization ---
if 'sim_index' not in st.session_state:
    st.session_state.sim_index = 0
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'total_savings' not in st.session_state:
    st.session_state.total_savings = 0.0
if 'co2_saved' not in st.session_state:
    st.session_state.co2_saved = 0.0
if 'stability_history' not in st.session_state:
    st.session_state.stability_history = []

# --- 3. Data Generation (The "World") ---
@st.cache_data
def get_simulation_data():
    return generate_synthetic_data(1440, real_stats=real_stats)

sim_data = get_simulation_data()

# --- 4. Sidebar Controls ---
st.sidebar.title("Simulation Controls")

# Playback Controls
col_play, col_reset = st.sidebar.columns(2)
with col_play:
    if st.button("Play / Pause"):
        st.session_state.is_playing = not st.session_state.is_playing
with col_reset:
    if st.button("Reset"):
        st.session_state.sim_index = 0
        st.session_state.total_savings = 0.0
        st.session_state.co2_saved = 0.0
        st.session_state.stability_history = []
        st.session_state.is_playing = False

# Speed Control
step_size = st.sidebar.slider("Simulation Speed", 1, 100, 10)

# Progress Bar (Time of Day)
current_time = sim_data.iloc[st.session_state.sim_index]['Timestamp']
progress = st.session_state.sim_index / 1440
st.sidebar.progress(progress)
st.sidebar.caption(f"Time: {current_time.strftime('%H:%M')}")

st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")
if energy_model:
    st.sidebar.success("AI Model Active")
else:
    st.sidebar.error("AI Model Inactive")

# --- 5. Main Dashboard Logic ---

# Title
st.title("GCPBBB Smart Grid Optimization")
st.markdown("Real-time energy balancing and grid stabilization.")

# Get Current State
current_row = sim_data.iloc[st.session_state.sim_index]
current_idx = st.session_state.sim_index

# Calculate Real Net Power (Hidden logic from data_loader exposed here)
# In data_loader: n_panels = 5 (if peak < 500), consumption scaled by 0.1
# We need to replicate this to show the user WHY it is charging
# But wait, the data_loader used consumption * 0.1 for SoC calculation.
# Let's show the "Effective Balance"
n_panels = 5
generated_power_total = current_row['Generated_Power'] * n_panels
# The simulation used a scaled consumption for SoC, but let's show the raw relationship
# to be honest to the user, or explain the scaling.
# Let's just show Solar vs Load.
net_power = generated_power_total - current_row['Grid_Consumption']

# --- AI Inference ---
action = "Hold"
pred_consumption = 0.0
import requests

# ... (imports)

# API Configuration
API_URL = os.getenv("API_URL", None) # Default to None (Local Mode) unless specified

# ... (setup)

# --- AI Inference ---
action = "Hold"
pred_consumption = 0.0

# Prepare Features (Common for both Local and API)
window_start = max(0, current_idx - 20)
window_data = sim_data.iloc[window_start : current_idx + 1].copy()

if len(window_data) > 15:
    processed_window = preprocess_features(window_data)
    if not processed_window.empty:
        recent_data = processed_window.tail(1).copy()
        
        # Mode 1: API Inference (Production)
        if API_URL:
            try:
                # Construct Payload from recent_data
                # We need to pass all features expected by the API
                # Convert row to dict
                payload = recent_data.iloc[0].to_dict()
                # Remove non-serializable or unnecessary fields
                if 'Timestamp' in payload: del payload['Timestamp']
                if 'Target_Consumption' in payload: del payload['Target_Consumption']
                if 'Signal' in payload: del payload['Signal']
                
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=1)
                if response.status_code == 200:
                    result = response.json()
                    pred_consumption = result['predicted_consumption']
                    action = result['recommended_action']
                else:
                    st.error(f"API Error: {response.status_code}")
            except Exception as e:
                st.error(f"API Connection Failed: {e}")

        # Mode 2: Local Inference (Fallback/Dev)
        elif energy_model and balancing_model:
            # 1. Predict
            X_reg = recent_data[[c for c in recent_data.columns if c not in ['Target_Consumption', 'Signal', 'Timestamp']]]
            X_reg_scaled = scaler_reg.transform(X_reg)
            pred_consumption = energy_model.predict(X_reg_scaled)[0]
            
            # 2. Decide
            recent_data['Target_Consumption'] = pred_consumption
            X_clf = recent_data[[c for c in recent_data.columns if c not in ['Signal', 'Timestamp']]]
            X_clf_scaled = scaler_clf.transform(X_clf)
            pred_signal = balancing_model.predict(X_clf_scaled)[0]
            
            signal_map = {0: 'Discharge', 1: 'Hold', 2: 'Charge'}
            action = signal_map.get(pred_signal, "Hold")

# --- Impact Calculation ---
grid_price = 0.30 
if 17 <= current_time.hour <= 21: 
    grid_price = 0.50

energy_moved = (current_row['Grid_Consumption'] / 1000) * (1/60) 
money_saved_now = 0.0
co2_saved_now = 0.0

if action == "Discharge":
    money_saved_now = energy_moved * grid_price
    co2_saved_now = energy_moved * 0.4 
elif action == "Charge":
    pass

st.session_state.total_savings += money_saved_now
st.session_state.co2_saved += co2_saved_now

# Stability Score
stability_impact = 0
if current_row['Grid_Consumption'] > 800 and action == "Discharge":
    stability_impact = 1 
elif current_row['Grid_Consumption'] < 300 and action == "Charge":
    stability_impact = 1 
    
st.session_state.stability_history.append(stability_impact)
if len(st.session_state.stability_history) > 100:
    st.session_state.stability_history.pop(0)
    
avg_stability = np.mean(st.session_state.stability_history) * 100 if st.session_state.stability_history else 100

# --- Visual Layout ---

# Top Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Est. Savings", f"${st.session_state.total_savings:.2f}")
m2.metric("CO2 Avoided", f"{st.session_state.co2_saved:.3f} kg")
m3.metric("Grid Stability", f"{avg_stability:.1f}%")
m4.metric("Forecast Load", f"{pred_consumption:.0f} W")

st.markdown("---")

# Energy Flow Diagram
c_solar, c_mid, c_home, c_grid = st.columns([1, 2, 1, 1])

with c_solar:
    st.markdown("### Solar Array")
    st.metric("Generation", f"{generated_power_total:.0f} W")
    if generated_power_total > 50:
        st.caption(f"Output from {n_panels} Panels")

with c_mid:
    st.markdown("### AI Controller")
    
    # Clean Action Card
    bg_color = "#f8f9fa"
    border_color = "#dee2e6"
    text_color = "#212529"
    
    if action == "Charge": 
        border_color = "#28a745"
        status_text = "Storing Excess Energy"
    elif action == "Discharge": 
        border_color = "#dc3545"
        status_text = "Powering Home"
    else: 
        border_color = "#17a2b8"
        status_text = "System Balanced"
    
    st.markdown(f"""
    <div style="background-color: {bg_color}; padding: 15px; border: 2px solid {border_color}; border-radius: 8px; text-align: center; color: {text_color};">
        <h3 style="margin:0;">{action.upper()}</h3>
        <p style="margin:5px 0 0 0; font-size: 0.9em;">{status_text}</p>
        <p style="margin:5px 0 0 0; font-weight: bold;">Battery: {current_row['Battery_SoC']:.1f}%</p>
        <p style="margin:5px 0 0 0; font-size: 0.8em; color: #666;">Net Power: {net_power:.0f} W</p>
    </div>
    """, unsafe_allow_html=True)

with c_home:
    st.markdown("### Home")
    st.metric("Consumption", f"{current_row['Grid_Consumption']:.0f} W")

with c_grid:
    st.markdown("### Grid")
    if action == "Discharge":
        st.caption("Idle (Saved!)")
    else:
        st.caption("Supplying Deficit")

st.markdown("---")

# Charts
st.subheader("Live Telemetry")
chart_col1, chart_col2 = st.columns(2)

history_df = sim_data.iloc[:current_idx+1].tail(200)

with chart_col1:
    st.markdown("**Solar vs. Load**")
    # Show Total Generation vs Load
    chart_data = history_df[['Grid_Consumption']].copy()
    chart_data['Solar_Generation'] = history_df['Generated_Power'] * n_panels
    st.line_chart(chart_data)

with chart_col2:
    st.markdown("**Battery Level**")
    st.line_chart(history_df[['Battery_SoC']])

# --- Auto-Play Logic ---
if st.session_state.is_playing:
    time.sleep(0.1) 
    st.session_state.sim_index += step_size
    
    if st.session_state.sim_index >= 1439:
        st.session_state.sim_index = 0
        st.session_state.is_playing = False 
    
    st.rerun()
