import pandas as pd
import json
import os
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# Rich UI Imports
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.status import Status
from rich.columns import Columns
from rich.align import Align

console = Console()

# --- CONFIGURATION ---
TRAIN_PATH = 'data/modelling/train.csv'
TEST_PATH = 'data/modelling/test.csv'
BEST_PARAMS_PATH = 'data/results/best_params.json'

def create_metric_table(model_name, y_true, y_pred, auc_score=None):
    """Parses classification report into a clean Rich table."""
    report = classification_report(y_true, y_pred, output_dict=True)
    
    table = Table(title=f"[bold cyan]{model_name}[/bold cyan]", box=None)
    table.add_column("Metric", style="dim")
    table.add_column("Score", justify="right")

    # We focus on Class 1 (Failure) as it's our KPI
    table.add_row("Precision (Fail)", f"{report['1']['precision']:.4f}")
    table.add_row("Recall (Fail)", f"[bold green]{report['1']['recall']:.4f}[/bold green]")
    table.add_row("F1-Score", f"{report['1']['f1-score']:.4f}")
    
    if auc_score:
        table.add_section()
        table.add_row("ROC AUC", f"[bold gold1]{auc_score:.4f}[/bold gold1]")
    
    return table

def get_final_metrics():
    console.print(Panel.fit(
        " [bold white]Final Model Performance Audit[/bold white] ", 
        style="on blue", 
        subtitle="Comparing Baseline vs. Optimized Champion"
    ))

    # 1. Load and Prepare Data
    with console.status("[bold green]Loading final datasets and parameters...", spinner="bouncingBall"):
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
        
        # Drop Leaks and Metadata
        drop_cols = ['machine_id', 'timestamp', 'is_failing_next_7_days', 'time_to_failure_hrs']
        string_cols = train_df.select_dtypes(include=['object', 'string']).columns.tolist()
        final_drop = list(set(drop_cols + string_cols))
        
        X_train = train_df.drop(columns=[c for c in final_drop if c in train_df.columns])
        y_train = train_df['is_failing_next_7_days']
        X_test = test_df.drop(columns=[c for c in final_drop if c in test_df.columns])
        y_test = test_df['is_failing_next_7_days']

    console.log(f"Feature Audit: [bold green]{X_train.shape[1]}[/bold green] predictors validated.")

    # 2. Logistic Regression (Baseline)
    with console.status("[bold yellow]Evaluating Baseline (Logistic Regression)..."):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        lr = LogisticRegression(class_weight='balanced', max_iter=1000)
        lr.fit(X_train_scaled, y_train)
        y_pred_lr = lr.predict(X_test_scaled)
        auc_lr = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])

    # 3. XGBoost (Champion)
    with console.status("[bold magenta]Evaluating Optimized Champion (XGBoost)..."):
        if not os.path.exists(BEST_PARAMS_PATH):
            console.print(f"[bold red]Error:[/bold red] {BEST_PARAMS_PATH} not found. Run optimization script first.")
            return
            
        with open(BEST_PARAMS_PATH, 'r') as f:
            best_params = json.load(f)
        
        xgb_model = XGBClassifier(**best_params)
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
        auc_xgb = roc_auc_score(y_test, y_prob_xgb)

    # 4. Final Comparison Dashboard
    lr_stats = create_metric_table("Logistic Regression", y_test, y_pred_lr, auc_lr)
    xgb_stats = create_metric_table("XGBoost Champion", y_test, y_pred_xgb, auc_xgb)

    console.print("\n")
    console.print(Panel(
        Columns([lr_stats, xgb_stats]),
        title="[bold]Model Competition Results[/bold]",
        border_style="bright_blue"
    ))

    # Business Conclusion
    lift = (auc_xgb - auc_lr) / auc_lr
    console.print(Align.center(
        f"\n[bold green]Champion Model shows a {lift:.2%} performance lift over baseline.[/bold green]"
    ))

if __name__ == "__main__":
    get_final_metrics()