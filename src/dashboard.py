import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from data_loader import generate_synthetic_data
from preprocessing import preprocess_features

# Page config
st.set_page_config(
    page_title="GCPBBB Energy Optimizer",
    page_icon="⚡",
    layout="wide"
)

# --- 1. Setup & Loading ---
@st.cache_resource
def load_models():
    if os.path.exists('models/energy_predictor.pkl') and os.path.exists('models/balancing_classifier.pkl'):
        energy_model = joblib.load('models/energy_predictor.pkl')
        balancing_model = joblib.load('models/balancing_classifier.pkl')
        scaler_reg = joblib.load('models/scaler_reg.pkl')
        scaler_clf = joblib.load('models/scaler_clf.pkl')
        return energy_model, balancing_model, scaler_reg, scaler_clf
    return None, None, None, None

energy_model, balancing_model, scaler_reg, scaler_clf = load_models()

# Sidebar
st.sidebar.title("⚙️ Simulation Controls")
n_samples = st.sidebar.slider("Simulation Duration (Minutes)", 100, 5000, 1000)
st.sidebar.markdown("---")
st.sidebar.info(
    "**System Status:**\n\n"
    f"{'🟢 Models Loaded' if energy_model else '🔴 Models Missing'}\n\n"
    "🟢 Pipeline Active"
)

# --- 2. Header & Rationale ---
st.title("⚡ Renewable Energy GCPBBB Optimization System")

with st.expander("ℹ️ **About the Data & Rationale (Click to Expand)**", expanded=True):
    st.markdown("""
    **Data Source: Hybrid Approach**
    *   **Real-World:** Electrical characteristics (Voltage, Current) are derived from the **Mendeley GPVS-Faults dataset**.
    *   **Synthetic:** Weather (Irradiance), Grid Consumption, and Battery State of Charge (SoC) are **generated synthetically**.
    
    **Why Synthetic Data?**
    > The GPVS dataset is designed for *micro-level fault detection* and lacks the *macro-level* variables (Grid Demand, Weather Forecasts) required for energy optimization. 
    > Synthetic data allows us to **simulate specific edge cases** (e.g., Critical Battery Low, Peak Grid Demand) to validate the robustness of our balancing logic.
    """)

# --- 3. Data Generation (Simulating Real-Time Feed) ---
# In a real app, this would be an API call. Here we generate fresh data.

if 'data_offset' not in st.session_state:
    st.session_state.data_offset = 0

# Checkbox to toggle real-time (logic handled at end of script)
st.checkbox("🔄 Enable Real-Time Simulation", key="real_time")

# Generate base data and shift it to simulate time passing
base_data = generate_synthetic_data(n_samples + st.session_state.data_offset)
data = base_data.tail(n_samples) # Always show the "latest" n_samples

# Get the "Current" state (last row)
current_state = data.iloc[-1]

# --- 4. Inputs Section ---
st.subheader("1. Real-Time System Inputs")
st.markdown("Live sensor readings feeding into the ML Pipeline:")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("☀️ Solar Irradiance", f"{current_state['Irradiance']:.1f} W/m²", delta="Input")
with col2:
    st.metric("🌡️ Temperature", f"{current_state['Temperature']:.1f} °C", delta="Input")
with col3:
    st.metric("🏙️ Grid Load", f"{current_state['Grid_Consumption']:.1f} W", delta="-Demand", delta_color="inverse")
with col4:
    st.metric("🔋 Battery SoC", f"{current_state['Battery_SoC']:.1f} %", delta="State")

# --- 5. Visualization Section ---
st.subheader("2. Energy Flow & Battery Status")
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.markdown("**Energy Balance (Generation vs Consumption)**")
    st.caption("Orange: Solar Generation | Blue: Grid Consumption")
    # Plot last 200 points
    chart_data = data[['Timestamp', 'Irradiance', 'Grid_Consumption']].tail(200).set_index('Timestamp')
    st.line_chart(chart_data, color=["#FFA500", "#0000FF"]) 

with col_g2:
    st.markdown("**Battery State of Charge (SoC)**")
    st.caption("Green: Battery Level (%)")
    soc_data = data[['Timestamp', 'Battery_SoC']].tail(200).set_index('Timestamp')
    st.line_chart(soc_data, color=["#00FF00"]) 

# --- 6. AI Inference & Outputs ---
st.subheader("3. AI Optimization Outputs")

if energy_model and balancing_model:
    # Preprocess for prediction
    processed_data = preprocess_features(data.copy())
    recent_data = processed_data.tail(1).copy() # Just predict for the "current" moment
    
    # --- Prediction Logic ---
    try:
        # 1. Energy Prediction
        # Reconstruct features expected by model
        # Explicitly drop 'Timestamp' as it was not used in training
        X_reg = recent_data[[c for c in recent_data.columns if c not in ['Target_Consumption', 'Signal', 'Timestamp']]]
        
        # Handle potential missing columns if any (robustness)
        # For now assuming exact match from training
        X_reg_scaled = scaler_reg.transform(X_reg)
        pred_consumption = energy_model.predict(X_reg_scaled)[0]
        
        # 2. Balancing Decision
        # Need 'Target_Consumption' feature for classifier
        # We use the PREDICTED consumption as the input for the classifier!
        # This is the "Pipeline" effect: Model 1 output -> Model 2 input
        recent_data['Target_Consumption'] = pred_consumption
        
        X_clf = recent_data[[c for c in recent_data.columns if c not in ['Signal', 'Timestamp']]]
        X_clf_scaled = scaler_clf.transform(X_clf)
        pred_signal = balancing_model.predict(X_clf_scaled)[0]
        
        # --- Display Results ---
        
        # Map signal
        signal_map = {0: 'Discharge', 1: 'Hold', 2: 'Charge'}
        signal_text = signal_map.get(pred_signal, "Unknown")
        
        # Determine Color
        if signal_text == "Charge": 
            action_color = "#28a745" # Green
            icon = "⚡"
        elif signal_text == "Discharge": 
            action_color = "#dc3545" # Red
            icon = "🔋"
        else: 
            action_color = "#17a2b8" # Blue
            icon = "⚖️"
            
        # Layout
        col_out1, col_out2 = st.columns([1, 2])
        
        with col_out1:
            st.markdown("### 🤖 Recommended Action")
            st.markdown(f"""
            <div style="padding: 20px; background-color: {action_color}; color: white; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h1 style="margin:0; font-size: 3em;">{icon}</h1>
                <h2 style="margin:0;">{signal_text.upper()}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col_out2:
            st.markdown("### 🧠 Decision Logic")
            
            # Generate dynamic explanation
            soc = current_state['Battery_SoC']
            gen = current_state['Irradiance']
            load = current_state['Grid_Consumption']
            
            reason = []
            if soc < 20: reason.append("Battery is Critically Low (<20%).")
            elif soc > 80: reason.append("Battery is Near Full (>80%).")
            
            if gen > load: reason.append("Solar Generation exceeds Grid Demand.")
            else: reason.append("Grid Demand exceeds Solar Generation.")
            
            if pred_consumption > load: reason.append("⚠️ Forecast predicts RISING demand.")
            
            st.info(f"**Why this action?**\n\n" + "\n".join([f"- {r}" for r in reason]))
            
            st.metric("🔮 Predicted Next-Step Load", f"{pred_consumption:.1f} W", delta=f"{pred_consumption - load:.1f} W")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.caption("Check if feature columns match the trained model.")

else:
    st.warning("⚠️ Models not found. Please run `python src/train_model.py` first.")

# --- 7. Auto-Refresh Logic (At the end to prevent blocking render) ---
if st.session_state.get('real_time', False):
    st.session_state.data_offset += 1
    import time
    time.sleep(2) # Update every 2 seconds
    st.rerun()
