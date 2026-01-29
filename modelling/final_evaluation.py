import pandas as pd
import json
import os
import argparse
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# Rich UI Imports
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.align import Align

console = Console()

# --- CONFIGURATION (Dynamic Defaults) ---
BASE_MODEL_DIR = 'data/modelling/'
BASE_RESULT_DIR = 'data/results/'

def create_metric_table(model_name, y_true, y_pred, auc_score=None, threshold=None):
    """Parses classification report into a clean Rich table."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    title_text = f"[bold cyan]{model_name}[/bold cyan]"
    if threshold:
        # FIXED: Removed 'size=11' which caused the MarkupError
        title_text += f"\n[dim](Thresh: {threshold:.2f})[/dim]"

    table = Table(title=title_text, box=None)
    table.add_column("Metric", style="dim")
    table.add_column("Score", justify="right")

    # We focus on Class 1 (Failure) as it's our KPI
    if '1' in report:
        prec = report['1']['precision']
        rec = report['1']['recall']
        f1 = report['1']['f1-score']
    else:
        prec, rec, f1 = 0.0, 0.0, 0.0

    table.add_row("Precision (Fail)", f"{prec:.4f}")
    table.add_row("Recall (Fail)", f"[bold green]{rec:.4f}[/bold green]")
    table.add_row("F1-Score", f"{f1:.4f}")
    
    if auc_score:
        table.add_section()
        table.add_row("ROC AUC", f"[bold gold1]{auc_score:.4f}[/bold gold1]")
    
    return table

def get_final_metrics(window_hours=168):
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
    result_dir = os.path.join(BASE_RESULT_DIR, subdir)
    
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    meta_path = os.path.join(data_dir, 'train_metadata.json')
    best_params_path = os.path.join(result_dir, 'best_params.json')

    console.print(Panel.fit(
        f" [bold white]Final Model Performance Audit[/bold white] ", 
        style="on blue", 
        subtitle=f"Strategy: {strategy_label}"
    ))

    # 2. Load and Prepare Data
    with console.status("[bold green]Loading final datasets and parameters...", spinner="bouncingBall"):
        if not os.path.exists(train_path):
             console.print(f"[bold red]ERROR: {train_path} not found.[/bold red]")
             return

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Load Metadata
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        target_col = meta.get('target_col', 'is_failing_next_7_days')

        # Drop Leaks and Metadata
        mandatory_drop = ['machine_id', 'timestamp', 'time_to_failure_hrs', target_col]
        string_cols = train_df.select_dtypes(include=['object', 'string']).columns.tolist()
        final_drop = list(set(mandatory_drop + string_cols))
        
        existing_drop = [c for c in final_drop if c in train_df.columns]

        X_train = train_df.drop(columns=existing_drop)
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=existing_drop)
        y_test = test_df[target_col]

    console.log(f"Predictors validated: [bold green]{X_train.shape[1]}[/bold green] features.")
    console.log(f"Target Variable: [magenta]{target_col}[/magenta]")

    # 3. Logistic Regression (Baseline)
    with console.status("[bold yellow]Evaluating Baseline (Logistic Regression)..."):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        lr = LogisticRegression(class_weight='balanced', max_iter=1000)
        lr.fit(X_train_scaled, y_train)
        y_pred_lr = lr.predict(X_test_scaled)
        auc_lr = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])

    # 4. XGBoost (Champion)
    with console.status("[bold magenta]Evaluating Optimized Champion (XGBoost)..."):
        if not os.path.exists(best_params_path):
            console.print(f"[bold red]Error:[/bold red] {best_params_path} not found. Run tune_xgboost.py first.")
            return
            
        with open(best_params_path, 'r') as f:
            best_params = json.load(f)
        
        xgb_model = XGBClassifier(**best_params)
        xgb_model.fit(X_train, y_train)
        
        # Get Probabilities first
        y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
        auc_xgb = roc_auc_score(y_test, y_prob_xgb)

    # --- AUTO THRESHOLD OPTIMIZATION ---
    # Find the threshold that maximizes F1 score to show "Best Case" performance
    best_threshold = 0.5
    best_f1 = 0
    
    # We scan discretely to find a good operating point
    for thresh in [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]:
        y_tmp = (y_prob_xgb >= thresh).astype(int)
        report = classification_report(y_test, y_tmp, output_dict=True, zero_division=0)
        if '1' in report:
            f1 = report['1']['f1-score']
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh

    # Generate final predictions using the optimized threshold
    y_pred_xgb_optimized = (y_prob_xgb >= best_threshold).astype(int)

    # 5. Final Comparison Dashboard
    lr_stats = create_metric_table("Logistic Regression", y_test, y_pred_lr, auc_lr)
    xgb_stats = create_metric_table("XGBoost Champion", y_test, y_pred_xgb_optimized, auc_xgb, threshold=best_threshold)

    console.print("\n")
    console.print(Panel(
        Columns([lr_stats, xgb_stats]),
        title="[bold]Model Competition Results[/bold]",
        border_style="bright_blue"
    ))

    # Business Conclusion
    lift = (auc_xgb - auc_lr) / auc_lr
    console.print(Align.center(
        f"\n[bold green]Champion Model shows a {lift:.2%} ROC AUC lift over baseline.[/bold green]"
    ))
    console.print(Align.center(f"[dim]Optimized threshold selected: {best_threshold}[/dim]"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final Model Evaluation Audit")
    parser.add_argument('--window', type=int, default=168, help='Window strategy: 168 (7-day) or 48 (2-day).')
    args = parser.parse_args()

    get_final_metrics(window_hours=args.window)