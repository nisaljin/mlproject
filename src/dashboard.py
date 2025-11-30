import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import time
from datetime import timedelta

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Add current directory to path to allow imports when running from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import generate_synthetic_data, get_real_world_stats
from preprocessing import preprocess_features

# Page config
st.set_page_config(
    page_title="Helios AI: Smart Grid Optimization",
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
    # Generate 30 days of data (1440 mins/day * 30 days)
    return generate_synthetic_data(1440 * 30, real_stats=real_stats)

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
        st.session_state.battery_soc = 50.0

# Speed Control
step_size = st.sidebar.slider("Simulation Speed", 1, 100, 10)

# Auto-Stop Control
st.session_state.limit_days = st.sidebar.slider("Auto-Stop (Days)", 1, 30, 20, help="Simulation will pause after this many days.")

# (Sidebar Progress moved to fragment)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")
if energy_model:
    st.sidebar.success("AI Model Active")
else:
    st.sidebar.error("AI Model Inactive")

# ... (rest of the file) ...

# (Auto-Play Logic moved to end of file)

# --- 5. Main Dashboard Logic (Fragment) ---

@st.fragment
def run_simulation_fragment():
    # Title (Inside fragment to allow reruns to catch it? No, title should be static)
    # Actually, if we rerun the fragment, we need everything that updates to be inside.
    
    # Title (Inside fragment to prevent duplication/ghosting)
    st.title("Helios AI: Smart Grid Optimization")
    st.markdown("Real-time energy balancing and grid stabilization.")
    
    # Initialize Dynamic State (Safety check inside fragment)
    if 'battery_soc' not in st.session_state:
        st.session_state.battery_soc = 50.0

    # Get Current Environment (Solar/Load)
    current_row = sim_data.iloc[st.session_state.sim_index]
    current_idx = st.session_state.sim_index
    
    # --- Update Status (Main Area) ---
    current_time = current_row['Timestamp']
    day_progress = (st.session_state.sim_index % 1440) / 1440
    
    # Display Time and Progress at the top of the dashboard
    status_col1, status_col2 = st.columns([1, 3])
    with status_col1:
        st.markdown(f"### {current_time.strftime('%Y-%m-%d')}")
        st.markdown(f"### {current_time.strftime('%H:%M')}")
    with status_col2:
        st.caption("Daily Progress")
        st.progress(day_progress)

    # Calculate Real Net Power
    n_panels = 30
    generated_power_total = current_row['Generated_Power'] * n_panels
    grid_consumption = current_row['Grid_Consumption']
    net_power = generated_power_total - grid_consumption

    # --- AI Inference ---
    action = "Hold"
    pred_consumption = 0.0
    import requests

    # API Configuration
    API_URL = os.getenv("API_URL", None)

    # Prepare Features
    window_start = max(0, current_idx - 20)
    window_data = sim_data.iloc[window_start : current_idx + 1].copy()

    # Overwrite SoC in window_data with our dynamic state for the last row
    window_data.loc[window_data.index[-1], 'Battery_SoC'] = st.session_state.battery_soc

    # print(f"DEBUG: Index={current_idx}, WindowLen={len(window_data)}, API_URL={API_URL}")

    if len(window_data) > 15:
        processed_window = preprocess_features(window_data)
        if not processed_window.empty:
            recent_data = processed_window.tail(1).copy()
            
            # Ensure 'Generated_Power' is present
            if 'Generated_Power' not in recent_data.columns:
                 recent_data['Generated_Power'] = current_row['Generated_Power']

            # Mode 1: API Inference (Production)
            if API_URL:
                # print("DEBUG: Attempting API Call...")
                try:
                    payload = recent_data.iloc[0].to_dict()
                    # Cleanup payload
                    for key in ['Timestamp', 'Target_Consumption', 'Signal']:
                        if key in payload: del payload[key]
                    
                    response = requests.post(f"{API_URL}/predict", json=payload, timeout=1)
                    # print(f"DEBUG: API Response Status: {response.status_code}")
                    if response.status_code == 200:
                        result = response.json()
                        pred_consumption = result['predicted_consumption']
                        action = result['recommended_action']
                    else:
                        st.error(f"API Error: {response.status_code}")
                except Exception as e:
                    # print(f"DEBUG: API Exception: {e}")
                    st.error(f"API Connection Failed: {e}")

            # Mode 2: Local Inference (Fallback)
            elif energy_model and balancing_model:
                try:
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
                except Exception as e:
                    st.error(f"Local Inference Error: {e}")

    # --- Physics Engine (Smart Self-Consumption) ---
    battery_capacity_wh = 10000
    max_power = 5000
    dt_hours = 1/60 

    # 1. Calculate Physical Limits (What the battery CAN do right now)
    current_wh = (st.session_state.battery_soc / 100.0) * battery_capacity_wh
    max_charge_wh = battery_capacity_wh - current_wh
    max_discharge_wh = current_wh
    
    # Convert Wh limits to Power (W) limits for this time step
    max_charge_power_limit = max_charge_wh / dt_hours
    max_discharge_power_limit = max_discharge_wh / dt_hours
    
    # Effective Power Limits (Hardware + State)
    eff_max_charge = min(max_power, max_charge_power_limit)
    eff_max_discharge = min(max_power, max_discharge_power_limit)

    target_flow = 0.0

    # 2. Determine Target Flow (Hierarchy of Control)
    
    # Priority 1: AI "Charge" (Arbitrage / Pre-charging)
    if action == "Charge":
        # Smart Charging: Prioritize Solar
        if net_power > 0:
            target_flow = net_power # Charge only with surplus solar (No Grid Import)
        else:
            target_flow = max_power # Charge from Grid (Arbitrage)
        
    # Priority 3: AI "Discharge" (Peak Selling / Cost Saving)
    elif action == "Discharge":
        # Load Following Logic: Only discharge to cover deficit
        if net_power < 0:
            target_flow = net_power # Match the deficit exactly (e.g., -1000W)
        else:
            target_flow = 0.0 # Do not discharge if there is surplus solar
        
    # Priority 4: Baseline Self-Consumption (Hold / Default)
    else:
        # If Net Power > 0 (Excess Solar): Charge Battery
        if net_power > 0:
            target_flow = net_power
        # If Net Power < 0 (Deficit): Discharge Battery
        else:
            target_flow = net_power # This is negative
            
    # 3. Apply Physical Constraints
    # Clamp target_flow between max discharge (negative) and max charge (positive)
    energy_flow = max(-eff_max_discharge, min(eff_max_charge, target_flow))

    # 4. Update SoC
    delta_wh = energy_flow * dt_hours
    st.session_state.battery_soc += (delta_wh / battery_capacity_wh) * 100.0
    st.session_state.battery_soc = np.clip(st.session_state.battery_soc, 0.0, 100.0)

    # --- Impact Calculation ---
    grid_price = 0.30 
    if 17 <= current_time.hour <= 21: 
        grid_price = 0.50

    # Calculate Financial Impact
    grid_interaction = 0.0
    if energy_flow > 0: # Battery Charging
        if net_power < 0:
            grid_interaction = -energy_flow # Import
        else:
            grid_interaction = 0
    elif energy_flow < 0: # Battery Discharging
        grid_interaction = -energy_flow # "Generation" from battery
        
    money_saved_now = 0.0
    if action == "Discharge" and grid_price == 0.50:
        money_saved_now = 0.05 # Bonus for discharging at peak
    elif action == "Charge" and grid_price == 0.30:
        money_saved_now = 0.01 # Small bonus for charging off-peak

    st.session_state.total_savings += money_saved_now
    st.session_state.co2_saved += (generated_power_total * dt_hours / 1000) * 0.4 # Solar always saves CO2

    # Stability Score
    stability_impact = 0
    if grid_consumption > 800 and action == "Discharge":
        stability_impact = 1 
    elif grid_consumption < 300 and action == "Charge":
        stability_impact = 1 
        
    st.session_state.stability_history.append(stability_impact)
    if len(st.session_state.stability_history) > 100:
        st.session_state.stability_history.pop(0)
        
    avg_stability = np.mean(st.session_state.stability_history) * 100 if st.session_state.stability_history else 100

    # --- Auto-Stop Logic ---
    # Get max days from sidebar (we need to pass this in or read from session state if we move slider to fragment)
    # Since slider is in sidebar (outside fragment), we can read it if we put it in session state or just read the widget value if accessible.
    # Streamlit widgets in sidebar are accessible globally.
    
    # --- Visual Layout ---

    # Top Metrics (Compact)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Est. Savings", f"${st.session_state.total_savings:.2f}")
    m2.metric("CO2 Avoided", f"{st.session_state.co2_saved:.3f} kg")
    m3.metric("Grid Stability", f"{avg_stability:.1f}%")
    m4.metric("Forecast Load", f"{pred_consumption:.0f} W")

    st.markdown("---")

    # Main Layout: Left (Flow) | Right (Charts)
    col_left, col_right = st.columns([0.8, 1.2])

    with col_left:
        st.subheader("Energy Flow")
        
        # 2x2 Grid for Flow Cards
        f1, f2 = st.columns(2)
        f3, f4 = st.columns(2)
        
        with f1:
            st.markdown("**☀️ Solar**")
            st.metric("Generation", f"{generated_power_total:.0f} W")
        
        with f2:
            st.markdown("**🏠 Home**")
            st.metric("Consumption", f"{grid_consumption:.0f} W")
            
        with f3:
            st.markdown("**🤖 AI Controller**")
            # Compact Action Status
            color = "#6c757d"
            if action == "Charge": color = "#28a745"
            elif action == "Discharge": color = "#dc3545"
            elif action == "Hold": color = "#17a2b8"
            
            st.markdown(f"<h3 style='color: {color}; margin:0;'>{action}</h3>", unsafe_allow_html=True)
            st.caption(f"Bat: {st.session_state.battery_soc:.1f}%")
            
        with f4:
            st.markdown("**⚡ Grid**")
            # Calculate True Grid Interaction
            grid_net = net_power - energy_flow
            
            status = "Balanced"
            val = "0 W"
            if grid_net > 10: 
                status = "Exporting"
                val = f"{grid_net:.0f} W"
            elif grid_net < -10: 
                status = "Importing"
                val = f"{grid_net:.0f} W"
                
            st.metric(status, val)
            
        # Details Expander
        with st.expander("View Math Details"):
            st.text(f"  Solar:   +{generated_power_total:5.0f} W")
            st.text(f"- Load:    -{grid_consumption:5.0f} W")
            st.text(f"- Battery: -{energy_flow:5.0f} W")
            st.text(f"---------------------")
            st.text(f"= Net:     {grid_net:5.0f} W")

    with col_right:
        st.subheader("Live Telemetry")
        
        # Update History
        if 'history_soc' not in st.session_state:
            st.session_state.history_soc = []
            st.session_state.history_solar = []
            st.session_state.history_load = []

        st.session_state.history_soc.append(st.session_state.battery_soc)
        st.session_state.history_solar.append(generated_power_total)
        st.session_state.history_load.append(grid_consumption)

        # Limit history (Memory Safety)
        if len(st.session_state.history_soc) > 200:
            st.session_state.history_soc.pop(0)
            st.session_state.history_solar.pop(0)
            st.session_state.history_load.pop(0)
            
        # Charts Stacked (Visible at once)
        
        st.markdown("**Solar vs. Load**")
        chart_df = pd.DataFrame({
            'Solar': st.session_state.history_solar,
            'Load': st.session_state.history_load
        })
        st.line_chart(chart_df, height=200)
        
        st.markdown("**Battery Level**")
        st.line_chart(st.session_state.history_soc, height=200)

    # --- Auto-Play Logic (Inside Fragment) ---
    if st.session_state.is_playing:
        time.sleep(0.1) 
        st.session_state.sim_index += step_size
        
        # Auto-Stop Limit
        max_sim_steps = 20 * 1440 # Default 20 days hardcoded if slider not passed, but let's try to read slider
        # We can't easily read the slider value here if it's not in session state.
        # Let's assume the user sets it in sidebar and we access it via st.session_state if we bound it, 
        # OR we just rely on the global variable 'max_days' if we move the slider definition.
        # Better: Check session state for a limit.
        
        limit_days = st.session_state.get('limit_days', 20)
        if st.session_state.sim_index >= limit_days * 1440:
            st.session_state.is_playing = False
            st.toast(f"Simulation stopped after {limit_days} days.")
        
        # Loop back check (Safety)
        if st.session_state.sim_index >= len(sim_data) - 1:
            st.session_state.sim_index = 0
            st.session_state.is_playing = False # Stop at end of data
        
        st.rerun()

# --- Main Execution ---

# Run the fragment
run_simulation_fragment()
