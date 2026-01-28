import pandas as pd
import numpy as np
import os
from rich.console import Console
from rich.table import Table

# --- CONFIGURATION ---
LABELED_DATA_PATH = 'data/labelled/telemetry_labeled.csv'
MAINTENANCE_LOG_PATH = 'data/cleaned/maintenance_log_full.csv'
METADATA_PATH = 'data/cleaned/machine_metadata_full.csv'
OUTPUT_DIR = 'data/features/'

def engineer_features():
    # 0. Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Datasets
    print("Loading datasets...")
    df = pd.read_csv(LABELED_DATA_PATH)
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

    # --- PART 1: ROLLING WINDOW STATISTICS (7-DAY) ---
    print("Calculating rolling statistics...")
    # 168 hours / 0.25 (15 min intervals) = 672 rows per window
    window_size = 672 
    
    # Group by machine to ensure windows don't cross between different machines
    grouper = df.groupby('machine_id')
    
    # 1 & 2. Vibration Features
    df['vibration_rolling_mean_7d'] = grouper['vibration_hz'].transform(lambda x: x.rolling(window_size).mean())
    df['vibration_rolling_max_7d'] = grouper['vibration_hz'].transform(lambda x: x.rolling(window_size).max())
    
    # 3. Temperature Stability
    df['temp_rolling_std_7d'] = grouper['temperature_c'].transform(lambda x: x.rolling(window_size).std())
    
    # 4. Pressure Mean
    df['pressure_rolling_mean_7d'] = grouper['pressure_bar'].transform(lambda x: x.rolling(window_size).mean())
    
    # 5. Power Strain
    df['power_rolling_max_7d'] = grouper['power_kw'].transform(lambda x: x.rolling(window_size).max())

    # --- PART 2: CATEGORICAL & TIME-BASED FEATURES ---
    print("Calculating categorical and time-based features...")
    
    # CRITICAL: Re-sort globally before merge_asof to prevent "keys must be sorted" error
    df = df.sort_values(by=['timestamp', 'machine_id']).reset_index(drop=True)
    maint = maint.sort_values(by=['maintenance_timestamp', 'machine_id']).reset_index(drop=True)

    # 6. Time Since Last Maintenance
    # We look 'backward' to find the most recent maintenance event
    df = pd.merge_asof(
        df, 
        maint[['machine_id', 'maintenance_timestamp']], 
        left_on='timestamp', 
        right_on='maintenance_timestamp', 
        by='machine_id', 
        direction='backward'
    )
    
    df['hours_since_maintenance'] = (df['timestamp'] - df['maintenance_timestamp']).dt.total_seconds() / 3600
    # Fill missing maintenance with a high value (e.g. 10,000 hours) indicating "long time ago"
    df['hours_since_maintenance'] = df['hours_since_maintenance'].fillna(10000) 

    # 7. Machine Age & Location (Metadata Join)
    print("Joining metadata...")
    df = df.merge(meta[['machine_id', 'age_months', 'location']], on='machine_id', how='left')
    
    # Binning Age
    df['machine_age_group'] = pd.cut(df['age_months'], 
                                    bins=[0, 24, 60, 500], 
                                    labels=['new', 'mid_age', 'legacy'])

    # 8. One-Hot Encoding
    # sparse=False ensures compatibility with standard Pandas/Scikit-Learn workflows
    df = pd.get_dummies(df, columns=['location', 'machine_age_group'], prefix=['loc', 'age'])

    # --- CLEANUP ---
    # Drop rows with NaNs (the first 7 days of history per machine will have NaN rolling stats)
    # Drop helper timestamp columns to avoid leakage
    df = df.dropna().drop(columns=['maintenance_timestamp'])
    
    return df


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

# Call it like this:

if __name__ == "__main__":
    feature_df = engineer_features()
    
    # --- FINAL CLEANUP FOR FEATURE STORE ---
    # Ensure timestamp is actually a datetime object
    feature_df['timestamp'] = pd.to_datetime(feature_df['timestamp'])
    
    # Save as Parquet for the Feature Store (Offline Path)
    output_path_parquet = os.path.join(OUTPUT_DIR, 'final_features.parquet')
    feature_df.to_parquet(output_path_parquet, index=False)
    
    # Keep the CSV for your own manual checks if you like
    output_path_csv = os.path.join(OUTPUT_DIR, 'final_features.csv')
    feature_df.to_csv(output_path_csv, index=False)

    print(f"Feature engineering complete. Shape: {feature_df.shape}")
    print(f"Saved to: {output_path_parquet} and {output_path_csv}")
    print_feature_summary(feature_df)