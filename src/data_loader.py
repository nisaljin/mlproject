import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict

def load_mendeley_data(data_dir: str) -> pd.DataFrame:
    """
    Load all CSV files from the Mendeley dataset and create a labeled dataset.
    Returns the combined DataFrame.
    """
    all_data = []
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory not found: {data_dir}")
        return pd.DataFrame()

    print(f"Loading Mendeley data from {data_dir}...")
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(data_dir, file)
            try:
                # Read only necessary columns to save memory if needed, but dataset is small enough
                df = pd.read_csv(file_path)
                
                # Extract labels from filename (e.g., F0L.csv -> fault_type='F0', mode='L')
                filename_parts = file.replace('.csv', '')
                if len(filename_parts) >= 3:
                    fault_type = f'F{filename_parts[1]}'  # F0, F1, ..., F7
                    mode = filename_parts[2]  # L or M
                    
                    df['fault_type'] = fault_type
                    df['operating_mode'] = mode
                    df['source_file'] = file
                    
                    # Calculate Power if columns exist
                    if 'Vpv' in df.columns and 'Ipv' in df.columns:
                        df['Power'] = df['Vpv'] * df['Ipv']
                    
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

def load_stability_data(data_dir: str, window_size: int = 1000) -> pd.DataFrame:
    """
    Load all CSV files and extract features for Grid Stability Monitoring.
    Segments the high-frequency data into windows to create multiple samples per file.
    """
    samples = []
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory not found: {data_dir}")
        return pd.DataFrame()

    print(f"Loading Stability Data from {data_dir}...")
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(data_dir, file)
            try:
                df = pd.read_csv(file_path)
                
                # Extract labels
                filename_parts = file.replace('.csv', '')
                if len(filename_parts) >= 3:
                    fault_type = f'F{filename_parts[1]}'  # F0, F1, ..., F7
                    mode = filename_parts[2]  # L or M
                    
                    # Segment Data
                    num_windows = len(df) // window_size
                    cols_to_use = ['Vpv', 'Ipv', 'Vdc', 'ia', 'ib', 'ic', 'va', 'vb', 'vc']
                    
                    for i in range(num_windows):
                        start_idx = i * window_size
                        end_idx = start_idx + window_size
                        window = df.iloc[start_idx:end_idx]
                        
                        features = {
                            'Fault_Type': fault_type,
                            'Mode': mode,
                            'Filename': file
                        }
                        
                        for col in cols_to_use:
                            if col in window.columns:
                                features[f'{col}_mean'] = window[col].mean()
                                features[f'{col}_std'] = window[col].std()
                                features[f'{col}_max'] = window[col].max()
                                features[f'{col}_min'] = window[col].min()
                        
                        samples.append(features)
                        
            except Exception as e:
                print(f"Error processing {file}: {e}")

    return pd.DataFrame(samples)

def get_real_world_stats(data_dir: str) -> Dict:
    """
    Load Mendeley data and extract key statistics to calibrate the simulation.
    """
    df = load_mendeley_data(data_dir)
    stats = {}
    
    if not df.empty and 'Power' in df.columns:
        # Use the 95th percentile as "Peak Power" to avoid outliers
        stats['peak_power'] = df['Power'].quantile(0.95)
        stats['mean_voltage'] = df['Vpv'].mean()
        stats['mean_current'] = df['Ipv'].mean()
        print(f"Derived Real-World Stats -- Peak Power: {stats['peak_power']:.2f}W, Mean V: {stats['mean_voltage']:.2f}V")
    else:
        # Fallback defaults if data missing
        print("Warning: Could not derive stats from Mendeley data. Using defaults.")
        stats['peak_power'] = 200.0 # Default 200W
        
    return stats

def generate_synthetic_data(n_samples: int, start_date: str = '2023-01-01', real_stats: Dict = None) -> pd.DataFrame:
    """
    Generate synthetic data for Weather, Grid Consumption, and Battery SoC.
    If real_stats is provided, use it to calibrate the generation.
    """
    dates = pd.date_range(start=start_date, periods=n_samples, freq='min') # Minute-level data
    
    # Determine Peak Power
    peak_power = 200.0 # Default
    if real_stats and 'peak_power' in real_stats:
        peak_power = real_stats['peak_power']
    
    # Synthetic Solar Irradiance (approximate diurnal cycle)
    # Peak at noon, zero at night
    hour_of_day = dates.hour + dates.minute / 60.0
    # Normalize sine wave 0-1
    solar_profile = np.maximum(0, np.sin(np.pi * (hour_of_day - 6) / 12)) 
    
    # Irradiance (W/m^2) - purely for feature set
    irradiance = solar_profile * 1000 
    # Add some noise/clouds
    noise = np.random.normal(0, 50, n_samples)
    irradiance = np.maximum(0, irradiance + noise)
    
    # Generated Power (W) - Calibrated by Real Data
    # Assuming Peak Irradiance (1000) produces Peak Power (from real data)
    generated_power = (irradiance / 1000.0) * peak_power
    
    # Synthetic Temperature (correlated with irradiance but lagged)
    temperature = 20 + 10 * np.sin(np.pi * (hour_of_day - 9) / 12) + np.random.normal(0, 2, n_samples)
    
    # Synthetic Grid Consumption (peaks in morning and evening)
    consumption = 500 + 300 * np.sin(np.pi * (hour_of_day - 7) / 12)**2 + \
                  400 * np.sin(np.pi * (hour_of_day - 19) / 12)**2 + \
                  np.random.normal(0, 50, n_samples)
    consumption = np.maximum(200, consumption)

    # Synthetic Battery SoC (starts at 50%, changes based on generation vs consumption)
    soc = np.zeros(n_samples)
    soc[0] = 50.0
    battery_capacity = 10000 # Wh
    
    for i in range(1, n_samples):
        # Net Energy = Generated Power - Consumption
        # Note: Consumption is usually much higher than a single panel's output
        # So we might need to assume an array of panels. 
        # Let's assume we have 10 panels if peak_power is small (~150W)
        
        n_panels = 1
        if peak_power < 500:
             n_panels = 5 # Scale up to make it realistic for a home
             
        total_generation = generated_power[i] * n_panels
        
        net_energy = total_generation - (consumption[i] * 0.1) # Scaling consumption down or generation up
        
        delta_soc = net_energy / battery_capacity * 100
        new_soc = soc[i-1] + delta_soc
        soc[i] = np.clip(new_soc, 0, 100)

    data = pd.DataFrame({
        'Timestamp': dates,
        'Irradiance': irradiance,
        'Temperature': temperature,
        'Grid_Consumption': consumption,
        'Battery_SoC': soc,
        'Generated_Power': generated_power # Add this for reference
    })
    
    return data
