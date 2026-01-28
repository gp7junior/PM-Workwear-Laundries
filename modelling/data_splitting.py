import pandas as pd
import numpy as np
import os
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.status import Status
from rich.columns import Columns
from rich.style import Style

# Initialize Console
console = Console()

# --- CONFIGURATION ---
INPUT_PATH = 'data/features/final_features.csv'
OUTPUT_DIR = 'data/modelling/'

def split_and_balance():
    # 0. Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data with Spinner
    with console.status(f"[bold green]Loading feature data from {INPUT_PATH}...", spinner="bouncingBall"):
        df = pd.read_csv(INPUT_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 2. STRICT TIME SORTING
    console.log("[yellow]Performing strict chronological sort...[/yellow]")
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    # 3. Define Split Point
    split_index = int(len(df) * 0.80)
    split_date = df.iloc[split_index]['timestamp']
    
    # 4. Perform the Split
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    
    # 5. Calculate Class Weights
    num_neg = (train_df['is_failing_next_7_days'] == 0).sum()
    num_pos = (train_df['is_failing_next_7_days'] == 1).sum()
    scale_pos_weight = num_neg / num_pos

    # --- UI: DATA SPLIT SUMMARY ---
    split_table = Table(title="[bold blue]Data Split Summary[/bold blue]", border_style="blue")
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
        f"[dim](Balancing ratio to be used in XGBoost parameters)[/dim]"
    )
    
    console.print(Panel(
        imbalance_content, 
        title="[bold red]Class Imbalance Strategy[/bold red]", 
        border_style="red",
        expand=False
    ))

    # 6. Save Data
    with console.status("[bold cyan]Exporting CSVs and Metadata...", spinner="dots"):
        train_path = os.path.join(OUTPUT_DIR, 'train.csv')
        test_path = os.path.join(OUTPUT_DIR, 'test.csv')
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        metadata = {
            'scale_pos_weight': float(scale_pos_weight),
            'split_date': str(split_date),
            'train_rows': len(train_df),
            'test_rows': len(test_df)
        }
        with open(os.path.join(OUTPUT_DIR, 'train_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        
    console.print(f"\n[bold green]✔[/bold green] Success! Artifacts saved to: [underline]{OUTPUT_DIR}[/underline]\n")

if __name__ == "__main__":
    # Header
    console.print(Panel.fit(
        " [bold white]Modeling Preparation Pipeline[/bold white] ", 
        style="on blue", 
        subtitle="Time-Series Split & Imbalance Handling"
    ))
    split_and_balance()