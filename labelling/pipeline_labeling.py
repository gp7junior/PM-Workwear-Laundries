import os
import pandas as pd
import numpy as np
from datetime import timedelta
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.status import Status
from rich.logging import RichHandler

# Initialize Rich Console
console = Console()

# --- CONFIGURATION ---
CLEANED_TELEMETRY_PATH = 'data/cleaned/telemetry_data_full.csv'
FAILURE_LOG_PATH = 'data/cleaned/failure_log_full.csv'
OUTPUT_DIR = 'data/labelled/'

def load_and_domain_clean(telemetry_path):
    """
    Step 1: Load pre-cleaned data and apply domain-specific physics rules.
    """
    with console.status(f"[bold green]Loading telemetry from {telemetry_path}...", spinner="dots"):
        df = pd.read_csv(telemetry_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    console.log("[blue]Applying domain-specific constraints...[/blue]")
    # Clip Temperature: 0°C to 100°C (liquid water range for washing)
    df['temperature_c'] = df['temperature_c'].clip(lower=0, upper=100)
    
    # Clip Vibration: Magnitudes cannot be negative
    if 'vibration_hz' in df.columns:
         df['vibration_hz'] = df['vibration_hz'].clip(lower=0)
    
    return df

def label_data(telemetry_df, failure_path):
    """
    Step 2: Create the binary target variable using a 7-day look-ahead window.
    """
    console.log("[blue]Loading failure logs and creating labels...[/blue]")
    failures = pd.read_csv(failure_path)
    
    # 2.1. Standardize types and handle potential NaTs
    telemetry_df['machine_id'] = telemetry_df['machine_id'].astype(str)
    failures['machine_id'] = failures['machine_id'].astype(str)
    telemetry_df['timestamp'] = pd.to_datetime(telemetry_df['timestamp'], errors='coerce')
    failures['failure_timestamp'] = pd.to_datetime(failures['failure_timestamp'], errors='coerce')
    
    telemetry_df = telemetry_df.dropna(subset=['timestamp'])
    failures = failures.dropna(subset=['failure_timestamp'])
    
    # 2.2. Global Chronological Sort
    console.log("[yellow]Performing global chronological sort...[/yellow]")
    telemetry_df = telemetry_df.sort_values(by=['timestamp', 'machine_id']).reset_index(drop=True)
    failures = failures.sort_values(by=['failure_timestamp', 'machine_id']).reset_index(drop=True)
    telemetry_df = telemetry_df.drop_duplicates(subset=['timestamp', 'machine_id'])

    # 2.3. Perform the Merge Asof
    with console.status("[bold magenta]Calculating time-to-failure windows...", spinner="bouncingBar"):
        try:
            labeled = pd.merge_asof(
                telemetry_df,
                failures[['machine_id', 'failure_timestamp', 'failure_type']],
                left_on='timestamp',
                right_on='failure_timestamp',
                by='machine_id',
                direction='forward', 
                tolerance=pd.Timedelta(days=30) 
            )
        except ValueError as e:
            console.print("[bold red]Critical Error during merge_asof![/bold red]")
            is_sorted = telemetry_df.groupby('machine_id')['timestamp'].apply(lambda x: x.is_monotonic_increasing).all()
            console.print(f"Internal machine sort: [bold]{is_sorted}[/bold]")
            raise e

    # 2.4. Target Calculation (168 hours = 7 days)
    labeled['time_to_failure_hrs'] = (labeled['failure_timestamp'] - labeled['timestamp']).dt.total_seconds() / 3600
    labeled['is_failing_next_7_days'] = np.where(
        (labeled['time_to_failure_hrs'] > 0) & (labeled['time_to_failure_hrs'] <= 168), 
        1, 0
    )
    
    return labeled.drop(columns=['failure_timestamp'])

def report_imbalance(df):
    total = len(df)
    positives = df['is_failing_next_7_days'].sum()
    negatives = total - positives
    
    # Create a nice Rich table
    stats_table = Table(show_header=False, box=None)
    stats_table.add_row("Total Rows", f"[bold]{total:,}[/bold]")
    stats_table.add_row("Healthy (0)", f"[green]{negatives:,}[/green]", f"[dim]{negatives/total*100:.2f}%[/dim]")
    stats_table.add_row("Failing (1)", f"[bold red]{positives:,}[/bold red]", f"[dim]{positives/total*100:.2f}%[/dim]")

    console.print("\n")
    console.print(Panel(
        stats_table,
        title="[bold blue]Target Variable Distribution[/bold blue]",
        subtitle="7-Day Failure Window",
        border_style="bright_blue",
        expand=False
    ))

if __name__ == "__main__":
    console.print(Panel.fit("[bold white]PM-Workwear-Laundries[/bold white]\n[dim]Labeling Pipeline[/dim]", border_style="green"))
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    telemetry = load_and_domain_clean(CLEANED_TELEMETRY_PATH)
    labeled_data = label_data(telemetry, FAILURE_LOG_PATH)
    
    report_imbalance(labeled_data)
    
    labeled_path = os.path.join(OUTPUT_DIR, 'telemetry_labeled.csv')
    labeled_data.to_csv(labeled_path, index=False)
    
    console.print(f"\n[bold green]✔[/bold green] Pipeline complete. Saved to: [underline]{labeled_path}[/underline]\n")