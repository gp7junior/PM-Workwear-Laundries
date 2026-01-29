import os
import argparse
import pandas as pd
import numpy as np
from datetime import timedelta
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Initialize Rich Console
console = Console()

# --- CONFIGURATION ---
CLEANED_TELEMETRY_PATH = 'data/cleaned/telemetry_data_full.csv'
FAILURE_LOG_PATH = 'data/cleaned/failure_log_full.csv'
OUTPUT_DIR = 'data/labelled/'

def load_and_domain_clean(telemetry_path):
    """Step 1: Load pre-cleaned data and apply domain constraints."""
    with console.status(f"[bold green]Loading telemetry from {telemetry_path}...", spinner="dots"):
        df = pd.read_csv(telemetry_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Clip Temperature: 0°C to 100°C
    df['temperature_c'] = df['temperature_c'].clip(lower=0, upper=100)
    
    # Clip Vibration: Magnitudes cannot be negative
    if 'vibration_hz' in df.columns:
         df['vibration_hz'] = df['vibration_hz'].clip(lower=0)
    
    return df

def label_data(telemetry_df, failure_path, lookahead_days=7):
    """
    Step 2: Create the binary target variable using a variable look-ahead window.
    """
    console.log(f"[blue]Loading failure logs. Target Window: [bold]{lookahead_days} Days[/bold]...[/blue]")
    failures = pd.read_csv(failure_path)
    
    # Standardize types
    telemetry_df['machine_id'] = telemetry_df['machine_id'].astype(str)
    failures['machine_id'] = failures['machine_id'].astype(str)
    telemetry_df['timestamp'] = pd.to_datetime(telemetry_df['timestamp'], errors='coerce')
    failures['failure_timestamp'] = pd.to_datetime(failures['failure_timestamp'], errors='coerce')
    
    telemetry_df = telemetry_df.dropna(subset=['timestamp'])
    failures = failures.dropna(subset=['failure_timestamp'])
    
    # Global Sort
    telemetry_df = telemetry_df.sort_values(by=['timestamp', 'machine_id']).reset_index(drop=True)
    failures = failures.sort_values(by=['failure_timestamp', 'machine_id']).reset_index(drop=True)
    telemetry_df = telemetry_df.drop_duplicates(subset=['timestamp', 'machine_id'])

    # Merge Asof (Forward look to find next failure)
    with console.status("[bold magenta]Mapping failures to timeline...", spinner="bouncingBar"):
        labeled = pd.merge_asof(
            telemetry_df,
            failures[['machine_id', 'failure_timestamp', 'failure_type']],
            left_on='timestamp',
            right_on='failure_timestamp',
            by='machine_id',
            direction='forward', 
            tolerance=pd.Timedelta(days=30) 
        )

    # --- DYNAMIC TARGET CALCULATION ---
    # Convert days to hours
    lookahead_hours = lookahead_days * 24
    
    # Calculate time to failure
    labeled['time_to_failure_hrs'] = (labeled['failure_timestamp'] - labeled['timestamp']).dt.total_seconds() / 3600
    
    # Create the dynamic column name, e.g., 'is_failing_next_2_days'
    target_col = f'is_failing_next_{lookahead_days}_days'
    
    labeled[target_col] = np.where(
        (labeled['time_to_failure_hrs'] > 0) & (labeled['time_to_failure_hrs'] <= lookahead_hours), 
        1, 0
    )
    
    # Rename specifically for the rest of the pipeline to understand it? 
    # Actually, better to keep it specific so we know what we predicted.
    # But for compatibility with your 'train_model.py', you might want to alias it.
    # Let's keep the specific name to be safe and rigorous.
    
    return labeled, target_col

def report_imbalance(df, target_col):
    total = len(df)
    positives = df[target_col].sum()
    negatives = total - positives
    
    stats_table = Table(show_header=False, box=None)
    stats_table.add_row("Total Rows", f"[bold]{total:,}[/bold]")
    stats_table.add_row("Healthy (0)", f"[green]{negatives:,}[/green]", f"[dim]{negatives/total*100:.2f}%[/dim]")
    stats_table.add_row("Failing (1)", f"[bold red]{positives:,}[/bold red]", f"[dim]{positives/total*100:.2f}%[/dim]")

    console.print("\n")
    console.print(Panel(
        stats_table,
        title=f"[bold blue]Target Distribution ({target_col})[/bold blue]",
        border_style="bright_blue",
        expand=False
    ))

if __name__ == "__main__":
    # Add CLI Argument Parsing
    parser = argparse.ArgumentParser(description="Label telemetry data with variable lookahead windows.")
    parser.add_argument('--days', type=int, default=7, help='Number of days to look ahead for failure (default: 7). Use 2 for 48h strategy.')
    args = parser.parse_args()

    console.print(Panel.fit("[bold white]Labeling Pipeline[/bold white]", border_style="green"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load
    telemetry = load_and_domain_clean(CLEANED_TELEMETRY_PATH)
    
    # Label with dynamic window
    labeled_data, target_col_name = label_data(telemetry, FAILURE_LOG_PATH, lookahead_days=args.days)
    
    # Report
    report_imbalance(labeled_data, target_col_name)
    
    # Save with specific name so we don't overwrite the 7-day version
    if args.days == 7:
        filename = 'telemetry_labeled.csv' # Keep original name for compatibility
    else:
        filename = f'telemetry_labeled_{args.days}d.csv'

    labeled_path = os.path.join(OUTPUT_DIR, filename)
    labeled_data.to_csv(labeled_path, index=False)
    
    console.print(f"\n[bold green]✔[/bold green] Pipeline complete. Saved to: [underline]{labeled_path}[/underline]")
    console.print(f"[dim]Target Column Created: {target_col_name}[/dim]\n")