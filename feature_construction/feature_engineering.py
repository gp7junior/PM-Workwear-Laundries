import pandas as pd
import numpy as np
import os
import argparse
from rich.console import Console
from rich.table import Table

# --- CONFIGURATION ---
OUTPUT_DIR = 'data/features/'

# We remove the hardcoded LABELED_DATA_PATH constant
MAINTENANCE_LOG_PATH = 'data/cleaned/maintenance_log_full.csv'
METADATA_PATH = 'data/cleaned/machine_metadata_full.csv'

def engineer_features(window_hours=168, input_path='data/labelled/telemetry_labeled.csv'):
    """
    Generates features based on a specific rolling window size (in hours).
    Reads from a specific labeled input file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Datasets
    print(f"Loading labeled data from: {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"ERROR: Could not find {input_path}. Did you run the labeling step with the correct --days?")
        return None, None

    maint = pd.read_csv(MAINTENANCE_LOG_PATH)
    meta = pd.read_csv(METADATA_PATH)
    
    # Ensure types
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    maint['maintenance_timestamp'] = pd.to_datetime(maint['maintenance_timestamp'])
    df['machine_id'] = df['machine_id'].astype(str)
    maint['machine_id'] = maint['machine_id'].astype(str)
    meta['machine_id'] = meta['machine_id'].astype(str)
    
    # Sort for rolling windows
    df = df.sort_values(['machine_id', 'timestamp'])

    # --- PART 1: ROLLING WINDOW STATISTICS ---
    window_str = f'{window_hours}h'
    suffix = f'_{window_hours}h' 

    print(f"Calculating rolling statistics for window: {window_str}...")

    # Group by machine
    # 1. Vibration Features
    df[f'vibration_rolling_mean{suffix}'] = df.groupby('machine_id').rolling(window_str, on='timestamp')['vibration_hz'].mean().reset_index(drop=True)
    df[f'vibration_rolling_max{suffix}'] = df.groupby('machine_id').rolling(window_str, on='timestamp')['vibration_hz'].max().reset_index(drop=True)
    df[f'vibration_rolling_std{suffix}'] = df.groupby('machine_id').rolling(window_str, on='timestamp')['vibration_hz'].std().reset_index(drop=True)

    # 2. Temperature Stability
    df[f'temp_rolling_std{suffix}'] = df.groupby('machine_id').rolling(window_str, on='timestamp')['temperature_c'].std().reset_index(drop=True)
    
    # 3. Pressure Mean
    df[f'pressure_rolling_mean{suffix}'] = df.groupby('machine_id').rolling(window_str, on='timestamp')['pressure_bar'].mean().reset_index(drop=True)
    
    # 4. Power Strain
    df[f'power_rolling_max{suffix}'] = df.groupby('machine_id').rolling(window_str, on='timestamp')['power_kw'].max().reset_index(drop=True)

    # --- PART 2: CATEGORICAL & TIME-BASED FEATURES ---
    print("Calculating categorical and time-based features...")
    
    # Re-sort globally
    df = df.sort_values(by=['timestamp', 'machine_id']).reset_index(drop=True)
    maint = maint.sort_values(by=['maintenance_timestamp', 'machine_id']).reset_index(drop=True)

    # 5. Time Since Last Maintenance
    df = pd.merge_asof(
        df, 
        maint[['machine_id', 'maintenance_timestamp']], 
        left_on='timestamp', 
        right_on='maintenance_timestamp', 
        by='machine_id', 
        direction='backward'
    )
    
    df['hours_since_maintenance'] = (df['timestamp'] - df['maintenance_timestamp']).dt.total_seconds() / 3600
    df['hours_since_maintenance'] = df['hours_since_maintenance'].fillna(10000) 

    # 6. Machine Age & Location
    print("Joining metadata...")
    df = df.merge(meta[['machine_id', 'age_months', 'location']], on='machine_id', how='left')
    
    df['machine_age_group'] = pd.cut(df['age_months'], 
                                    bins=[0, 24, 60, 500], 
                                    labels=['new', 'mid_age', 'legacy'])

    # 7. One-Hot Encoding
    df = pd.get_dummies(df, columns=['location', 'machine_age_group'], prefix=['loc', 'age'])

    # --- CLEANUP ---
    df = df.dropna().drop(columns=['maintenance_timestamp'])
    df = df.dropna().drop(columns=['failure_timestamp'])
    
    return df, suffix

def print_feature_summary(df):
    console = Console()
    table = Table(title="[bold blue]Feature Engineering Output[/bold blue]", show_header=True, header_style="bold magenta")
    
    table.add_column("Index", style="dim", width=6)
    table.add_column("Feature Name", style="cyan")
    table.add_column("Dtype", style="green")

    for i, col in enumerate(df.columns):
        table.add_row(str(i), col, str(df[col].dtype))

    console.print(table)
    console.print(f"[bold green]Total Features:[/bold green] {len(df.columns)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate features with variable rolling window sizes.")
    parser.add_argument('--window', type=int, default=168, help='Rolling window size in hours (default: 168). Use 48 for 2 days.')
    args = parser.parse_args()

    # --- INTELLIGENT PATH SELECTION ---
    # Automatically decide which labeled file to load based on the window arg
    if args.window == 48:
        # If using 48h features, look for the 2-day target file
        input_file = 'data/labelled/telemetry_labeled_2d.csv'
        base_name = "final_features_48h"
    else:
        # Default to standard 7-day
        input_file = 'data/labelled/telemetry_labeled.csv'
        base_name = "final_features"

    feature_df, file_suffix = engineer_features(window_hours=args.window, input_path=input_file)
    
    if feature_df is not None:
        # Cleanup types
        feature_df['timestamp'] = pd.to_datetime(feature_df['timestamp'])
        
        # Save
        output_path_parquet = os.path.join(OUTPUT_DIR, f'{base_name}.parquet')
        output_path_csv = os.path.join(OUTPUT_DIR, f'{base_name}.csv')
        
        feature_df.to_parquet(output_path_parquet, index=False)
        feature_df.to_csv(output_path_csv, index=False)

        print(f"Feature engineering complete. Shape: {feature_df.shape}")
        print(f"Saved to: {output_path_csv}")
        print_feature_summary(feature_df)