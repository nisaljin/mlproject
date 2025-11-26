import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from data_loader import generate_synthetic_data

def perform_eda():
    print("Performing EDA...")
    
    # Generate synthetic data
    n_samples = 1440 * 7 # 1 week of minute data
    data = generate_synthetic_data(n_samples)
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # 1. Time Series Plot
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(data['Timestamp'], data['Irradiance'], label='Irradiance (W/m^2)', color='orange')
    plt.title('Solar Irradiance')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(data['Timestamp'], data['Grid_Consumption'], label='Grid Consumption (W)', color='blue')
    plt.title('Grid Consumption')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(data['Timestamp'], data['Battery_SoC'], label='Battery SoC (%)', color='green')
    plt.title('Battery State of Charge')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('output/eda_timeseries.png')
    print("Saved output/eda_timeseries.png")
    
    # 2. Correlation Matrix
    plt.figure(figsize=(8, 6))
    corr = data.drop(columns=['Timestamp']).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('output/eda_correlation.png')
    print("Saved output/eda_correlation.png")
    
    # 3. Distribution Plots
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.histplot(data['Irradiance'], kde=True, color='orange')
    plt.title('Irradiance Distribution')
    
    plt.subplot(1, 3, 2)
    sns.histplot(data['Grid_Consumption'], kde=True, color='blue')
    plt.title('Consumption Distribution')
    
    plt.subplot(1, 3, 3)
    sns.histplot(data['Battery_SoC'], kde=True, color='green')
    plt.title('SoC Distribution')
    
    plt.tight_layout()
    plt.savefig('output/eda_distributions.png')
    print("Saved output/eda_distributions.png")
    
    print("EDA Complete.")

if __name__ == "__main__":
    perform_eda()
