import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict

def load_mendeley_data(data_dir: str) -> pd.DataFrame:
    """
    Load all CSV files from the Mendeley dataset and create a labeled dataset.
    """
    all_data = []
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(data_dir, file)
            try:
                df = pd.read_csv(file_path)
                
                # Extract labels from filename (e.g., F0L.csv -> fault_type='F0', mode='L')
                filename_parts = file.replace('.csv', '')
                if len(filename_parts) >= 3:
                    fault_type = f'F{filename_parts[1]}'  # F0, F1, ..., F7
                    mode = filename_parts[2]  # L or M
                    
                    df['fault_type'] = fault_type
                    df['operating_mode'] = mode
                    df['source_file'] = file
                    all_data.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")

    if not all_data:
        return pd.DataFrame()

    combined_data = pd.concat(all_data, ignore_index=True)
    # Sort by time if 'Time' column exists
    if 'Time' in combined_data.columns:
        combined_data = combined_data.sort_values('Time').reset_index(drop=True)
        
    return combined_data

def generate_synthetic_data(n_samples: int, start_date: str = '2023-01-01') -> pd.DataFrame:
    """
    Generate synthetic data for Weather, Grid Consumption, and Battery SoC.
    """
    dates = pd.date_range(start=start_date, periods=n_samples, freq='min') # Minute-level data
    
    # Synthetic Solar Irradiance (approximate diurnal cycle)
    # Peak at noon, zero at night
    hour_of_day = dates.hour + dates.minute / 60.0
    irradiance = np.maximum(0, 1000 * np.sin(np.pi * (hour_of_day - 6) / 12)) 
    # Add some noise/clouds
    noise = np.random.normal(0, 50, n_samples)
    irradiance = np.maximum(0, irradiance + noise)
    
    # Synthetic Temperature (correlated with irradiance but lagged)
    temperature = 20 + 10 * np.sin(np.pi * (hour_of_day - 9) / 12) + np.random.normal(0, 2, n_samples)
    
    # Synthetic Grid Consumption (peaks in morning and evening)
    consumption = 500 + 300 * np.sin(np.pi * (hour_of_day - 7) / 12)**2 + \
                  400 * np.sin(np.pi * (hour_of_day - 19) / 12)**2 + \
                  np.random.normal(0, 50, n_samples)
    consumption = np.maximum(200, consumption)

    # Synthetic Battery SoC (starts at 50%, changes based on generation vs consumption)
    # This is a simplified simulation
    soc = np.zeros(n_samples)
    soc[0] = 50.0
    battery_capacity = 10000 # Wh
    
    for i in range(1, n_samples):
        # Generation (proportional to irradiance) - Consumption
        net_energy = (irradiance[i] * 0.2) - (consumption[i] * 0.1) # Scaling factors
        delta_soc = net_energy / battery_capacity * 100
        new_soc = soc[i-1] + delta_soc
        soc[i] = np.clip(new_soc, 0, 100)

    data = pd.DataFrame({
        'Timestamp': dates,
        'Irradiance': irradiance,
        'Temperature': temperature,
        'Grid_Consumption': consumption,
        'Battery_SoC': soc
    })
    
    return data
