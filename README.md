# Renewable Energy GCPBBB Optimization System

This project implements a Machine Learning solution for optimizing renewable energy integration in a Grid-Connected Photo Sensor Based Battery Balancing (GCPBBB) system. It uses synthetic data (supplemented by real-world electrical characteristics) to train models for energy consumption forecasting and battery balancing.

## 📂 Project Structure

*   **`src/`**: Core machine learning logic (data loading, preprocessing, model training).
*   **`backend/`**: FastAPI application serving the ML models and simulation logic.
*   **`frontend/`**: React-based web dashboard for real-time visualization.
*   **`notebooks/`**: Jupyter notebooks for experimentation.
*   **`models/`**: Saved trained models (`.pkl` files).
*   **`output/`**: Generated EDA plots and reports.
*   **`dataset/`**: Directory for the Mendeley GPVS-Faults dataset (and generated synthetic data).


## 🚀 Setup Instructions

### 1. Prerequisites
*   Python 3.11 or higher
*   Git

### 2. Clone the Repository
```bash
git clone https://github.com/rahulwork252/SOLARPRED_PROJ_ML.git
cd SOLARPRED_PROJ_ML
```

### 3. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Running the Project

### 1. Exploratory Data Analysis (EDA)
Generate synthetic data (calibrated with real-world Mendeley data) and view visualizations (saved to `output/`):
```bash
python src/eda.py
```

### 2. Train the Models
Train the **Energy Predictor** (Random Forest) and **Balancing Classifier** (XGBoost). This will save the models to the `models/` directory.
```bash
python src/train_model.py
```

### 3. Launch the Application
This project now uses a modern web stack (React + FastAPI). The easiest way to run it is via Docker (see below).

To run manually without Docker:
**1. Start Backend:**
```bash
uvicorn backend.main:app --reload --port 8000
```
**2. Start Frontend:**
```bash
cd frontend
npm install
npm run dev
```
Then visit `http://localhost:5173`.


### 4. Running with Docker 🐳
For a consistent environment, you can run the entire stack (Frontend + Backend) using Docker.
See the **[Docker Setup & Usage Guide](DOCKER_README.md)** for detailed instructions.


## 📊 Dashboard Features
*   **Real-Time Simulation:** Controls to Play/Pause simulation and adjust time dilation speed.
*   **Fault Injection:** Interactively inject electrical faults (F1-F7) into specific PV strings to test system stability.
*   **Scenario Manager:** Simulate different weather conditions (Sunny, Cloudy, Stormy) to validate energy balancing logic.
*   **Live Telemetry:** Visualization of Solar Generation, Grid Consumption, and Battery SoC in real-time.
*   **AI Decision Explainer:** Transparently displays *why* the system chose to Charge, Discharge, or Hold.


## 📝 Technical Report
For a detailed explanation of the methodology, model performance, and system architecture, please refer to the [Technical Report](technical_report.md).
