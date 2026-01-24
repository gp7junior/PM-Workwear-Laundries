import pandas as pd
import numpy as np
import os
import json

# --- CONFIGURATION ---
INPUT_PATH = 'data/features/final_features.csv'
OUTPUT_DIR = 'data/modelling/'

def split_and_balance():
    # 0. Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    print(f"Loading feature data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 2. STRICT TIME SORTING
    # This is non-negotiable for time-series splitting
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    # 3. Define Split Point (80% of the timeline)
    # We find the timestamp at the 80th percentile
    split_index = int(len(df) * 0.80)
    split_date = df.iloc[split_index]['timestamp']
    
    print(f"Splitting data at 80% cutoff: {split_date}")
    
    # 4. Perform the Split
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    
    # 5. Calculate Class Weights (Crucial for Imbalance)
    # We only calculate this on the TRAINING set to avoid leakage
    num_neg = (train_df['is_failing_next_7_days'] == 0).sum()
    num_pos = (train_df['is_failing_next_7_days'] == 1).sum()
    
    scale_pos_weight = num_neg / num_pos
    
    print("\n--- Split Statistics ---")
    print(f"Train Set: {len(train_df):,} rows (Ends: {train_df['timestamp'].max()})")
    print(f"Test Set:  {len(test_df):,} rows (Starts: {test_df['timestamp'].min()})")
    print("-" * 30)
    print(f"Training Class Imbalance:")
    print(f"   Healthy (0): {num_neg:,}")
    print(f"   Failures (1): {num_pos:,}")
    print(f"   Calculated 'scale_pos_weight': {scale_pos_weight:.2f}")
    print("(Use this value for XGBoost parameter 'scale_pos_weight')")
    
    # 6. Save Data
    # We drop timestamps from the exported files if we want, 
    # but usually we keep them for error analysis later.
    train_path = os.path.join(OUTPUT_DIR, 'train.csv')
    test_path = os.path.join(OUTPUT_DIR, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # 7. Save Metadata (Weights) for the Model Training script
    metadata = {
        'scale_pos_weight': float(scale_pos_weight),
        'split_date': str(split_date),
        'train_rows': len(train_df),
        'test_rows': len(test_df)
    }
    
    with open(os.path.join(OUTPUT_DIR, 'train_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nFiles saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    split_and_balance()