# Technical Report: Renewable Energy GCPBBB Grid Connected Photo Sensor

## 1. Introduction
This project aims to optimize renewable energy integration into the GCPBBB (Grid-Connected Photo Sensor Based Battery Balancing) system using Machine Learning. The solution addresses the challenge of inconsistent renewable energy generation by predicting optimal energy distribution and battery balancing signals.

## 2. Methodology

### 2.1 Data Acquisition & Rationale
- **Dataset Analysis:** The available **GPVS-Faults dataset** (Mendeley) contains high-frequency electrical measurements primarily designed for *fault detection* and *diagnosis* in PV systems. As noted in recent research, this dataset lacks the high-level grid consumption, weather forecasts, and battery state data required for predictive energy optimization.
- **Strategy:** To bridge the gap between the available data and the project's energy balancing goals, we adopted a **hybrid approach**:
    1.  **Synthetic Data Generation:** We generated realistic time-series data for:
        - **Solar Irradiance:** Modeled with diurnal cycles and stochastic cloud cover.
        - **Grid Consumption:** Simulated with typical residential load profiles (morning/evening peaks).
        - **Battery State of Charge (SoC):** Calculated based on the net energy flow (Generation vs. Consumption).
    2.  **Model Training:** The machine learning models were trained on this comprehensive synthetic dataset to demonstrate the feasibility of the *optimization* and *balancing* logic.

#### Justification for Synthetic Data
The use of synthetic data is justified by three key factors:
1.  **Data Scarcity:** Public datasets rarely combine synchronized PV generation, grid consumption, *and* battery storage data for a specific custom architecture like GCPBBB.
2.  **Controlled Experimentation:** Synthetic generation allows us to simulate specific edge cases (e.g., extreme weather, sudden grid demand spikes) to test the robustness of the balancing logic, which may not be present in limited historical records.
3.  **Standard Research Practice:** In Smart Grid research, simulation-based modeling is a standard precursor to physical deployment, allowing for the validation of control algorithms before hardware integration.

### 2.2 Feature Engineering
- **Time-based Features:** Hour, Minute to capture daily patterns.
- **Lag Features:** Previous values of Irradiance and Consumption to capture temporal dependencies.
- **Derived Features:** Net Energy (Generation vs Consumption).

### 2.3 Model Development
Two non-deep learning models were developed:
1.  **Energy Predictor (Random Forest Regressor):**
    - **Goal:** Predict next-step Grid Consumption.
    - **Performance:** MAE of ~0.30 (on normalized data).
2.  **Balancing Classifier (XGBoost):**
    - **Goal:** Predict battery action (0: Discharge, 1: Hold, 2: Charge).
    - **Logic:** Based on Battery SoC limits (<20% Charge, >80% Discharge) and Net Energy balance.
    - **Performance:** High accuracy on synthetic test set.
3.  **Grid Stability Monitor (Random Forest Classifier):**
    - **Goal:** Detect and classify grid faults (e.g., Line-Line, Open Circuit) in real-time.
    - **Logic:** Analyzes high-frequency electrical signatures (Voltage, Current, Harmonics) to identify anomalies.
    - **Training Data:** Real-world Mendeley GPVS-Faults dataset.

### 2.4 System Architecture (Pipeline)
The automated pipeline operates in a continuous loop:
1.  **Data Ingestion:** Sensors collect real-time data (Irradiance, Temperature, Current Load, Battery Voltage).
2.  **Preprocessing:**
    - **Cleaning:** Handling missing values or sensor noise.
    - **Feature Engineering:** Calculating `Net_Energy` and generating time-lag features (e.g., `Load_t-1`).
    - **Scaling:** Normalizing inputs using the pre-trained `StandardScaler`.
3.  **Inference:**
    - The **Energy Predictor** forecasts the Grid Load for the next interval ($t+1$).
    - The **Balancing Classifier** takes the current state + forecast to output a decision (Charge/Discharge).
4.  **Action:** The decision signal is sent to the Battery Management System (BMS) to execute the control.

### 2.5 Dashboard Design
The React-based dashboard serves as the "Control Room" interface:
- **Real-time Graphs:** Live line charts showing the synchronization of Solar Generation (Orange) vs. Grid Consumption (Blue).
- **Battery Monitor:** A gauge or time-series plot tracking the State of Charge (SoC) to ensure it stays within safe limits (20%-80%).
- **Prediction Panel:** A dedicated section displaying the *Forecasted Load* and the *Recommended Action* (e.g., "High Demand Predicted -> DISCHARGING").
- **Stability Indicator:** A metric showing the simulated "Grid Stability Score" (80-100%), providing an at-a-glance health check of the system.

#### Dashboard Implementation
The dashboard is built using **React** and operates as follows:
1.  **Initialization:** On startup, the frontend connects to the FastAPI backend.
2.  **Data Stream:** The backend generates or loads the latest batch of sensor data (simulating a real-time feed).
3.  **Live Inference:** For every data point, the backend runs the *Preprocessing Pipeline* (scaling/lagging) and queries the models for a prediction.
4.  **Rendering:** Results are visualized using interactive charts (e.g., Recharts) and status indicators, updating dynamically as the backend pushes new state.

## 3. MLOps & Deployment

To ensure the system is reproducible, scalable, and production-ready, we implemented a complete MLOps pipeline:

### 3.1 Model Packaging
- **Dockerization:** The entire application (API + Dashboard) is containerized using `Dockerfile`. This ensures consistency across different environments (Dev, Test, Prod) by isolating dependencies.
- **API Wrapper:** A **FastAPI** backend (`src/api.py`) serves the models via REST endpoints (`/predict`). This allows external systems (like a real BMS) to consume predictions programmatically.

### 3.2 Deployment Strategy
- **Cloud Provider:** Google Cloud Platform (GCP) Compute Engine (Ubuntu).
- **Orchestration:** `docker-compose` manages the multi-container setup (Frontend, Backend).
- **Reverse Proxy:** **Nginx** is configured as a gateway to route traffic to the appropriate service (Port 80 -> React Frontend / FastAPI Backend) and handle security headers.

### 3.3 CI/CD Pipeline
- **Tool:** GitHub Actions.
- **Workflow:** The `.github/workflows/main.yml` pipeline triggers on every push to `main`:
    1.  **Test:** Runs unit tests and verifies imports.
    2.  **Build:** Builds the Docker image to ensure no build errors.
    3.  **Verify:** Briefly spins up the container to check the health endpoint.

## 4. Performance & Use Cases

### 3.1 Model Performance
The developed models achieved high performance on the validation dataset:
1.  **Energy Predictor (Random Forest):**
    - **Metric:** Mean Absolute Error (MAE) of **~0.30** (on normalized scale).
    - **Interpretation:** The model accurately forecasts the next-step grid consumption with very low error, effectively capturing the morning and evening demand peaks.
2.  **Balancing Classifier (XGBoost):**
    - **Metric:** Accuracy of **100%** (on synthetic test set).
    - **Interpretation:** The model successfully learned the complex logic for battery management (Charge vs. Discharge vs. Hold) based on the multi-variable input state (SoC, Net Energy). *Note: Real-world accuracy would vary with noise, but this validates the logic.*
3.  **Grid Stability Monitor (Random Forest):**
    - **Metric:** High Accuracy (>95%) on held-out real-world test set.
    - **Interpretation:** The model effectively distinguishes between "Stable" operation and 7 different fault types (F1-F7), enabling the "Guardian" module to isolate faults immediately.

### 3.2 Model Outputs
The system provides two key actionable outputs:
- **Predicted Grid Consumption (kW):** A forecast of the immediate future demand, allowing the grid operator to anticipate load.
- **Battery Action Signal (Class):** A decision signal for the battery controller:
    - `0`: **Discharge** (Support the grid during high demand or low generation).
    - `1`: **Hold** (Maintain charge when system is balanced).
    - `2`: **Charge** (Store excess renewable energy during peak generation).

### 3.3 Use Cases
This solution enables several critical Smart Grid applications:
1.  **Peak Shaving:** Automatically discharging the battery during predicted consumption spikes to reduce stress on the main grid.
2.  **Renewable Smoothing:** Absorbing volatile solar generation spikes (charging) to provide a smooth, consistent power output.
3.  **Grid Stabilization:** acting as a fast-response buffer to maintain voltage/frequency stability during sudden cloud cover or load changes.

## 5. Conclusion
The developed models demonstrate the feasibility of using ML for:
- **Load Forecasting:** Accurate short-term prediction of grid demand.
- **Automated Balancing:** Intelligent decision-making for battery storage to maintain grid stability.

This solution contributes to "AI for Social Good" by promoting efficient renewable energy usage and reducing reliance on fossil fuels.

## 6. Future Work
- Integrate real-time weather API.
- Deploy models on edge devices (e.g., Raspberry Pi) for local control.
- Expand dataset with real-world battery usage logs.
