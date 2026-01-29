import pandas as pd
import numpy as np
import os
import json
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.status import Status
from rich.columns import Columns
from rich.style import Style

# Initialize Console
console = Console()

# --- CONFIGURATION (Dynamic) ---
BASE_FEATURE_DIR = 'data/features/'
BASE_OUTPUT_DIR = 'data/modelling/'

def split_and_balance(window_hours=168):
    # 1. Determine Paths and Targets based on Strategy
    if window_hours == 48:
        input_filename = 'final_features_48h.csv'
        output_subdir = os.path.join(BASE_OUTPUT_DIR, '48h')
        target_col = 'is_failing_next_2_days'
        strategy_name = "48-Hour (High Sensitivity)"
    else:
        # Default 7-day
        input_filename = 'final_features.csv'
        output_subdir = os.path.join(BASE_OUTPUT_DIR, '7d') # Recommended to separate this too
        target_col = 'is_failing_next_7_days'
        strategy_name = "7-Day (Baseline)"

    input_path = os.path.join(BASE_FEATURE_DIR, input_filename)
    
    # 0. Setup Output
    os.makedirs(output_subdir, exist_ok=True)
    
    console.print(f"[dim]Strategy: {strategy_name} | Target: {target_col}[/dim]")
    console.print(f"[dim]Input: {input_path} -> Output: {output_subdir}[/dim]\n")

    # 2. Load Data with Spinner
    with console.status(f"[bold green]Loading feature data from {input_path}...", spinner="bouncingBall"):
        if not os.path.exists(input_path):
            console.print(f"[bold red]ERROR: Input file not found: {input_path}[/bold red]")
            return

        df = pd.read_csv(input_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 3. STRICT TIME SORTING
    console.log("[yellow]Performing strict chronological sort...[/yellow]")
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    # 4. Define Split Point (80% Cutoff)
    split_index = int(len(df) * 0.80)
    split_date = df.iloc[split_index]['timestamp']
    
    # 5. Perform the Split
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    
    # 6. Calculate Class Weights (Dynamic Target)
    try:
        num_neg = (train_df[target_col] == 0).sum()
        num_pos = (train_df[target_col] == 1).sum()
    except KeyError:
        console.print(f"[bold red]CRITICAL ERROR:[/bold red] Target column '{target_col}' not found in dataset columns: {df.columns.tolist()}")
        return

    # Avoid division by zero
    if num_pos == 0:
        scale_pos_weight = 1.0
        console.print("[bold red]WARNING: No failures found in training set! scale_pos_weight set to 1.0[/bold red]")
    else:
        scale_pos_weight = num_neg / num_pos

    # --- UI: DATA SPLIT SUMMARY ---
    split_table = Table(title=f"[bold blue]Data Split Summary ({strategy_name})[/bold blue]", border_style="blue")
    split_table.add_column("Partition", style="cyan")
    split_table.add_column("Rows", justify="right")
    split_table.add_column("Timeline Range", style="magenta")
    
    split_table.add_row(
        "Training", 
        f"{len(train_df):,}", 
        f"{train_df['timestamp'].min().date()} to {train_df['timestamp'].max().date()}"
    )
    split_table.add_row(
        "Testing", 
        f"{len(test_df):,}", 
        f"{test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()}"
    )
    
    console.print("\n")
    console.print(split_table)

    # --- UI: IMBALANCE & WEIGHTS ---
    imbalance_content = (
        f"[green]Healthy (0):[/green] {num_neg:,}\n"
        f"[red]Failures (1):[/red] {num_pos:,}\n\n"
        f"[bold gold1]Recommended 'scale_pos_weight': {scale_pos_weight:.2f}[/bold gold1]\n"
        f"[dim](Target: {target_col})[/dim]"
    )
    
    console.print(Panel(
        imbalance_content, 
        title="[bold red]Class Imbalance Strategy[/bold red]", 
        border_style="red",
        expand=False
    ))

    # 7. Save Data
    with console.status("[bold cyan]Exporting CSVs and Metadata...", spinner="dots"):
        train_path = os.path.join(output_subdir, 'train.csv')
        test_path = os.path.join(output_subdir, 'test.csv')
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        metadata = {
            'scale_pos_weight': float(scale_pos_weight),
            'split_date': str(split_date),
            'train_rows': len(train_df),
            'test_rows': len(test_df),
            'target_col': target_col, # Save target name so future scripts know what to predict
            'strategy': strategy_name
        }
        with open(os.path.join(output_subdir, 'train_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        
    console.print(f"\n[bold green]✔[/bold green] Success! Artifacts saved to: [underline]{output_subdir}[/underline]\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data for training/testing.")
    parser.add_argument('--window', type=int, default=168, help='Window strategy: 168 (7-day) or 48 (2-day).')
    args = parser.parse_args()

    # Header
    console.print(Panel.fit(
        " [bold white]Modeling Preparation Pipeline[/bold white] ", 
        style="on blue", 
        subtitle=f"Time-Series Split: {args.window}h Window"
    ))
    
    split_and_balance(window_hours=args.window)