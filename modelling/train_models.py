import pandas as pd
import json
import os
import sys
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

# --- CONFIGURATION ---
TRAIN_PATH = 'data/modelling/train.csv'
TEST_PATH = 'data/modelling/test.csv'
META_PATH = 'data/modelling/train_metadata.json'
OUTPUT_DIR = 'data/results/'

console = Console()

def create_report_table(title, report_dict, auc_score=None):
    """Creates a stylized Rich table from a classification report dictionary."""
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

def train_and_evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    console.print(Panel.fit(" [bold white]Model Competition[/bold white] ", style="on blue"))

    # 1. Load Data
    with console.status("[bold green]Loading datasets...", spinner="aesthetic"):
        if not os.path.exists(TRAIN_PATH):
            console.print(f"[bold red]ERROR:[/bold red] {TRAIN_PATH} not found.")
            return

        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)

        with open(META_PATH, 'r') as f:
            meta = json.load(f)
        scale_pos_weight = meta['scale_pos_weight']
    
    console.log(f"Data Loaded: [cyan]{len(train_df):,}[/cyan] train rows | [cyan]{len(test_df):,}[/cyan] test rows")

    # 2. Prepare Features
    mandatory_drop = ['machine_id', 'timestamp', 'is_failing_next_7_days', 'time_to_failure_hrs']
    string_cols = train_df.select_dtypes(include=['object', 'string']).columns.tolist()
    drop_cols = list(set(mandatory_drop + string_cols))
    
    existing_drop_cols = [c for c in drop_cols if c in train_df.columns]
    X_train = train_df.drop(columns=existing_drop_cols)
    y_train = train_df['is_failing_next_7_days']
    X_test = test_df.drop(columns=existing_drop_cols)
    y_test = test_df['is_failing_next_7_days']

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

    # --- MODEL 2: XGBOOST ---
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
        
        y_pred_xgb = xgb_model.predict(X_test)
        y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
        xgb_report_dict = classification_report(y_test, y_pred_xgb, output_dict=True)
        auc_xgb = roc_auc_score(y_test, y_prob_xgb)

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
    xgb_table = create_report_table("XGBoost Champion", xgb_report_dict, auc_xgb)
    
    console.print("\n")
    console.print(Columns([lr_table, xgb_table]))

    # --- SAVE RESULTS ---
    output_file = os.path.join(OUTPUT_DIR, 'model_comparison.txt')
    with open(output_file, 'w') as f:
        f.write(f"XGBoost ROC AUC: {auc_xgb:.4f}\n")
        f.write(classification_report(y_test, y_pred_xgb))

    console.print(f"\n[bold green]✔[/bold green] DONE! Results saved in [underline]{output_file}[/underline]")

if __name__ == "__main__":
    train_and_evaluate()