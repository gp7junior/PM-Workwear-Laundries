import pandas as pd
import numpy as np
import optuna
import json
import os
import argparse
import xgboost as xgb
from sklearn.metrics import recall_score, precision_score, roc_auc_score

# Rich UI Imports
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.status import Status

console = Console()

# --- CONFIGURATION (Dynamic Defaults) ---
BASE_MODEL_DIR = 'data/modelling/'
BASE_RESULT_DIR = 'data/results/'

def load_data(window_hours):
    # 1. Determine Paths
    if window_hours == 48:
        subdir = '48h'
    else:
        subdir = '7d'
        if not os.path.exists(os.path.join(BASE_MODEL_DIR, '7d')):
            subdir = '' 
        else:
            subdir = '7d'

    data_dir = os.path.join(BASE_MODEL_DIR, subdir)
    # Return the result dir too so we know where to save later
    result_dir = os.path.join(BASE_RESULT_DIR, subdir)
    
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    meta_path = os.path.join(data_dir, 'train_metadata.json')

    with console.status(f"[bold green]Loading data from {subdir}...", spinner="dots"):
        if not os.path.exists(train_path):
            console.print(f"[bold red]ERROR: {train_path} not found.[/bold red]")
            return None

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        base_weight = meta['scale_pos_weight']
        target_col = meta.get('target_col', 'is_failing_next_7_days')

        # DYNAMIC DROP: Use the target from metadata
        mandatory_drop = ['machine_id', 'timestamp', 'time_to_failure_hrs', target_col]
        string_cols = train_df.select_dtypes(include=['object', 'string']).columns.tolist()
        final_drop = list(set(mandatory_drop + string_cols))
        
        existing_drop = [c for c in final_drop if c in train_df.columns]
        
        X = train_df.drop(columns=existing_drop)
        y = train_df[target_col]
        
        X_test_final = test_df.drop(columns=existing_drop)
        y_test_final = test_df[target_col]
        
    console.log(f"[bold green]✔[/bold green] Data ready. Target: [magenta]{target_col}[/magenta]")
    return X, y, X_test_final, y_test_final, base_weight, result_dir

def objective(trial, X_train, y_train, X_val, y_val, base_weight):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', base_weight * 0.5, base_weight * 3.0),
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    # Optimization Metric: Recall (Task 6 Requirement)
    return recall_score(y_val, y_pred)

def run_optimization(window_hours=168):
    strategy_name = "48h High Sensitivity" if window_hours == 48 else "7d Baseline"
    console.print(Panel.fit(f" [bold white]XGBoost Tuning: {strategy_name}[/bold white] ", style="on purple"))
    
    data_tuple = load_data(window_hours)
    if not data_tuple: return
    
    X_full, y_full, X_test, y_test, base_weight, result_dir = data_tuple
    
    # Time-Series Split for Tuning (First 80% of Train for Fit, Last 20% of Train for Validate)
    split_idx = int(len(X_full) * 0.8)
    X_train_tune, y_train_tune = X_full.iloc[:split_idx], y_full.iloc[:split_idx]
    X_val_tune, y_val_tune = X_full.iloc[split_idx:], y_full.iloc[split_idx:]
    
    console.log(f"Tuning Split: [cyan]{len(X_train_tune)}[/cyan] train / [cyan]{len(X_val_tune)}[/cyan] validation")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    
    func = lambda trial: objective(trial, X_train_tune, y_train_tune, X_val_tune, y_val_tune, base_weight)

    # 50 Trials is enough for a demo
    with console.status("[bold magenta]Running 50 Trials...", spinner="bouncingBall"):
        study.optimize(func, n_trials=50)

    # --- UI: BEST PARAMETERS ---
    best_table = Table(title="Top Parameters Found", box=None)
    best_table.add_column("Parameter", style="magenta")
    best_table.add_column("Value", style="bold yellow")
    
    for k, v in study.best_params.items():
        best_table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))

    console.print("\n")
    console.print(Panel(best_table, title="[bold green]Optimization Results[/bold green]", border_style="green", expand=False))
    console.print(f"🏆 [bold white]Best Validation Recall:[/bold white] [bold cyan]{study.best_value:.4f}[/bold cyan]\n")

    # Retrain Final Model
    with console.status("[bold blue]Retraining Final Champion Model...", spinner="runner"):
        best_params = study.best_params
        best_params.update({'random_state': 42, 'n_jobs': -1})
        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(X_full, y_full)
        
        y_pred_test = final_model.predict(X_test)
        y_prob_test = final_model.predict_proba(X_test)[:, 1]

    # Final Evaluation
    res_table = Table(title="Final Performance on Unseen Test Data", border_style="cyan")
    res_table.add_column("Metric", style="bold")
    res_table.add_column("Score", justify="right")
    
    res_table.add_row("Recall", f"{recall_score(y_test, y_pred_test):.4f}")
    res_table.add_row("Precision", f"{precision_score(y_test, y_pred_test):.4f}")
    res_table.add_row("ROC AUC", f"[bold cyan]{roc_auc_score(y_test, y_prob_test):.4f}[/bold cyan]")
    
    console.print(res_table)

    # Save Results
    os.makedirs(result_dir, exist_ok=True)
    save_path = os.path.join(result_dir, 'best_params.json')
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    console.print(f"\n[bold green]✔[/bold green] Best parameters saved to [underline]{save_path}[/underline]\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune XGBoost Hyperparameters")
    parser.add_argument('--window', type=int, default=168, help='Window strategy: 168 (7-day) or 48 (2-day).')
    args = parser.parse_args()
    
    run_optimization(window_hours=args.window)