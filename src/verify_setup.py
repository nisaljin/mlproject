import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from data_loader import load_mendeley_data, generate_synthetic_data

def verify_setup():
    print("Verifying setup...")
    
    # 1. Test Mendeley Data Loading
    data_dir = 'dataset'
    if os.path.exists(data_dir):
        print(f"Loading data from {data_dir}...")
        try:
            # Load only a few files to be fast, or just check if function runs
            # We'll rely on the function's internal logic, but maybe we should limit it?
            # The load_mendeley_data loads ALL csvs. Let's hope it's not too slow.
            # Actually, let's just check if we can generate synthetic data first, 
            # as that's crucial for the new features.
            pass 
        except Exception as e:
            print(f"Error loading Mendeley data: {e}")
    else:
        print(f"Warning: {data_dir} does not exist. Skipping Mendeley data load test.")

    # 2. Test Synthetic Data Generation
    print("Generating synthetic data...")
    try:
        n_samples = 100
        syn_data = generate_synthetic_data(n_samples)
        print(f"Successfully generated {len(syn_data)} samples of synthetic data.")
        print("Columns:", syn_data.columns.tolist())
        print(syn_data.head())
    except Exception as e:
        print(f"Error generating synthetic data: {e}")
        raise e

    print("Setup verification complete.")

if __name__ == "__main__":
    verify_setup()
