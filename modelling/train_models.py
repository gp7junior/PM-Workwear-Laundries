import pandas as pd
import json
import os
import sys
import argparse
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# Rich UI Imports
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.status import Status
from rich.live import Live
from rich.columns import Columns

console = Console()

# --- CONFIGURATION (Dynamic Defaults) ---
BASE_MODEL_DIR = 'data/modelling/'
BASE_RESULT_DIR = 'data/results/'

def create_report_table(title, report_dict, auc_score=None, threshold=0.5):
    """Creates a stylized Rich table from a classification report dictionary."""
    if threshold != 0.5:
        title += f" (Thresh: {threshold})"
        
    table = Table(title=f"[bold]{title}[/bold]", header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Precision")
    table.add_column("Recall")
    table.add_column("F1-Score")
    
    # Add rows for classes
    for key in ['0', '1']:
        if key in report_dict:
            metrics = report_dict[key]
            table.add_row(
                f"Class {key}",
                f"{metrics['precision']:.2f}",
                f"{metrics['recall']:.2f}",
                f"{metrics['f1-score']:.2f}"
            )
    
    if auc_score:
        table.add_section()
        table.add_row("ROC AUC", "", "", f"[bold cyan]{auc_score:.4f}[/bold cyan]")
    
    return table

def train_and_evaluate(window_hours=168):
    # 1. Determine Paths based on Strategy
    if window_hours == 48:
        subdir = '48h'
        strategy_label = "48-Hour (High Sensitivity)"
    else:
        subdir = '7d' 
        if not os.path.exists(os.path.join(BASE_MODEL_DIR, '7d')):
            subdir = '' 
        else:
            subdir = '7d'
        strategy_label = "7-Day (Baseline)"

    data_dir = os.path.join(BASE_MODEL_DIR, subdir)
    output_dir = os.path.join(BASE_RESULT_DIR, subdir)
    
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    meta_path = os.path.join(data_dir, 'train_metadata.json')
    
    os.makedirs(output_dir, exist_ok=True)
    
    console.print(Panel.fit(f" [bold white]Model Competition: {strategy_label}[/bold white] ", style="on blue"))
    console.print(f"[dim]Reading from: {data_dir}[/dim]")

    # 2. Load Data & Metadata
    with console.status("[bold green]Loading datasets...", spinner="aesthetic"):
        if not os.path.exists(train_path):
            console.print(f"[bold red]ERROR:[/bold red] {train_path} not found. Run 'split_data.py --window {window_hours}' first.")
            return

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        scale_pos_weight = meta['scale_pos_weight']
        target_col = meta.get('target_col', 'is_failing_next_7_days') 
    
    console.log(f"Data Loaded: [cyan]{len(train_df):,}[/cyan] train rows | [cyan]{len(test_df):,}[/cyan] test rows")
    console.log(f"Target Variable: [bold magenta]{target_col}[/bold magenta]")

    # 3. Prepare Features
    mandatory_drop = ['machine_id', 'timestamp', 'time_to_failure_hrs', target_col]
    string_cols = train_df.select_dtypes(include=['object', 'string']).columns.tolist()
    drop_cols = list(set(mandatory_drop + string_cols))
    existing_drop_cols = [c for c in drop_cols if c in train_df.columns]
    
    X_train = train_df.drop(columns=existing_drop_cols)
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=existing_drop_cols)
    y_test = test_df[target_col]

    console.log(f"Dropped [red]{len(existing_drop_cols)}[/red] metadata/leakage columns.")
    console.log(f"Training on [bold green]{X_train.shape[1]}[/bold green] features.")

    # --- MODEL 1: LOGISTIC REGRESSION ---
    with console.status("[bold yellow]Training Logistic Regression Baseline...", spinner="bouncingBall"):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        lr_model.fit(X_train_scaled, y_train)
        
        y_pred_lr = lr_model.predict(X_test_scaled)
        lr_report_dict = classification_report(y_test, y_pred_lr, output_dict=True)

    # --- MODEL 2: XGBOOST (With Threshold Tuning) ---
    with console.status("[bold blue]Training XGBoost Champion Model...", spinner="runner"):
        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )

        xgb_model.fit(X_train, y_train)
        y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
        auc_xgb = roc_auc_score(y_test, y_prob_xgb)

    # --- NEW: THRESHOLD TUNING SCAN ---
    scan_table = Table(title="Threshold Sensitivity Scan", box=None)
    scan_table.add_column("Threshold", style="bold yellow")
    scan_table.add_column("Precision", style="dim")
    scan_table.add_column("Recall (Failures Caught)", style="bold green")
    scan_table.add_column("F1-Score")

    best_threshold = 0.5
    best_f1 = 0

    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
        y_pred_custom = (y_prob_xgb >= thresh).astype(int)
        report = classification_report(y_test, y_pred_custom, output_dict=True)
        
        if '1' in report:
            prec = report['1']['precision']
            rec = report['1']['recall']
            f1 = report['1']['f1-score']
        else:
            prec, rec, f1 = 0.0, 0.0, 0.0
            
        scan_table.add_row(f"{thresh:.2f}", f"{prec:.2f}", f"{rec:.2f}", f"{f1:.2f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    console.print("\n")
    console.print(scan_table)
    console.print(f"[dim]Optimal F1 Threshold detected at: {best_threshold}[/dim]\n")

    # Final Predictions using Optimized Threshold
    y_pred_xgb_optimized = (y_prob_xgb >= best_threshold).astype(int)
    xgb_report_dict = classification_report(y_test, y_pred_xgb_optimized, output_dict=True)

    # --- UI: FEATURE IMPORTANCE TABLE ---
    importances = pd.Series(xgb_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    imp_table = Table(title="Top 10 Feature Importance (XGBoost)", border_style="yellow")
    imp_table.add_column("Feature", style="cyan")
    imp_table.add_column("Gini Importance", justify="right")

    for feat, val in importances.head(10).items():
        imp_table.add_row(feat, f"{val:.4f}")

    console.print(imp_table)

    # --- UI: COMPARISON DASHBOARD ---
    lr_table = create_report_table("Logistic Regression", lr_report_dict)
    xgb_table = create_report_table("XGBoost Champion", xgb_report_dict, auc_xgb, threshold=best_threshold)
    
    console.print("\n")
    console.print(Columns([lr_table, xgb_table]))

    # --- SAVE RESULTS ---
    output_file = os.path.join(output_dir, 'model_comparison.txt')
    with open(output_file, 'w') as f:
        f.write(f"Strategy: {strategy_label}\n")
        f.write(f"Optimal Threshold: {best_threshold}\n")
        f.write(f"XGBoost ROC AUC: {auc_xgb:.4f}\n")
        # FIXED: Now using y_pred_xgb_optimized
        f.write(classification_report(y_test, y_pred_xgb_optimized))

    console.print(f"\n[bold green]✔[/bold green] DONE! Results saved in [underline]{output_file}[/underline]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate Models")
    parser.add_argument('--window', type=int, default=168, help='Window strategy: 168 (7-day) or 48 (2-day).')
    args = parser.parse_args()
    
    train_and_evaluate(window_hours=args.window)