import pandas as pd
import numpy as np
import optuna
import json
import os
import xgboost as xgb
from sklearn.metrics import recall_score, precision_score, roc_auc_score

# Rich UI Imports
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.status import Status
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# --- CONFIGURATION ---
TRAIN_PATH = 'data/modelling/train.csv'
TEST_PATH = 'data/modelling/test.csv'
META_PATH = 'data/modelling/train_metadata.json'
OUTPUT_DIR = 'data/results/'

def load_data():
    with console.status("[bold green]Loading and sanitizing data...", spinner="dots"):
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
        
        with open(META_PATH, 'r') as f:
            meta = json.load(f)
        base_weight = meta['scale_pos_weight']

        drop_cols = ['machine_id', 'timestamp', 'is_failing_next_7_days', 'time_to_failure_hrs']
        string_cols = train_df.select_dtypes(include=['object', 'string']).columns.tolist()
        final_drop = list(set(drop_cols + string_cols))
        
        X = train_df.drop(columns=[c for c in final_drop if c in train_df.columns])
        y = train_df['is_failing_next_7_days']
        
        X_test_final = test_df.drop(columns=[c for c in final_drop if c in test_df.columns])
        y_test_final = test_df['is_failing_next_7_days']
        
    console.log(f"[bold green]✔[/bold green] Data ready. Features: [bold cyan]{len(X.columns)}[/bold cyan]")
    return X, y, X_test_final, y_test_final, base_weight

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
        'verbosity': 0 # Keep XGBoost quiet during tuning
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return recall_score(y_val, y_pred)

def run_optimization():
    console.print(Panel.fit(" [bold white]XGBoost Hyperparameter Tuning[/bold white] ", style="on purple"))
    
    X_full, y_full, X_test, y_test, base_weight = load_data()
    
    split_idx = int(len(X_full) * 0.8)
    X_train_tune, y_train_tune = X_full.iloc[:split_idx], y_full.iloc[:split_idx]
    X_val_tune, y_val_tune = X_full.iloc[split_idx:], y_full.iloc[split_idx:]
    
    console.log(f"Time-Series Split: [cyan]{len(X_train_tune)}[/cyan] train, [cyan]{len(X_val_tune)}[/cyan] validation")

    # 3. Optimization with Live Logging
    optuna.logging.set_verbosity(optuna.logging.WARNING) # Disable default Optuna spam
    study = optuna.create_study(direction='maximize')
    
    func = lambda trial: objective(trial, X_train_tune, y_train_tune, X_val_tune, y_val_tune, base_weight)

    with console.status("[bold magenta]Running 50 Trials of Optimization...", spinner="bouncingBall"):
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

    # 4. Retrain Final
    with console.status("[bold blue]Retraining Final Champion Model...", spinner="runner"):
        best_params = study.best_params
        best_params.update({'random_state': 42, 'n_jobs': -1})
        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(X_full, y_full)
        
        y_pred_test = final_model.predict(X_test)
        y_prob_test = final_model.predict_proba(X_test)[:, 1]

    # --- UI: FINAL EVALUATION ---
    res_table = Table(title="Final Performance on Unseen Test Data", border_style="cyan")
    res_table.add_column("Metric", style="bold")
    res_table.add_column("Score", justify="right")
    
    res_table.add_row("Recall (Failure Catch Rate)", f"{recall_score(y_test, y_pred_test):.4f}")
    res_table.add_row("Precision", f"{precision_score(y_test, y_pred_test):.4f}")
    res_table.add_row("ROC AUC", f"[bold cyan]{roc_auc_score(y_test, y_prob_test):.4f}[/bold cyan]")
    
    console.print(res_table)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=4)
    console.print(f"\n[bold green]✔[/bold green] Best parameters saved to [underline]{OUTPUT_DIR}best_params.json[/underline]\n")

if __name__ == "__main__":
    run_optimization()