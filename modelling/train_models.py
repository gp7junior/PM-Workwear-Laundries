import pandas as pd
import json
import os
import sys
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# --- CONFIGURATION ---
TRAIN_PATH = 'data/modelling/train.csv'
TEST_PATH = 'data/modelling/test.csv'
META_PATH = 'data/modelling/train_metadata.json'
OUTPUT_DIR = 'data/results/'

def train_and_evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    print(">>> [1/5] Loading datasets...", flush=True)
    if not os.path.exists(TRAIN_PATH):
        print(f"ERROR: {TRAIN_PATH} not found. Run splitting script first.", flush=True)
        return

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    with open(META_PATH, 'r') as f:
        meta = json.load(f)
    scale_pos_weight = meta['scale_pos_weight']
    
    print(f"    Train size: {len(train_df):,} rows")
    print(f"    Test size:  {len(test_df):,} rows")
    print(f"    Class Weight: {scale_pos_weight:.2f}")

    # 2. Prepare Features
    # Define mandatory drop columns (IDs + Target)
    mandatory_drop = [
        'machine_id', 
        'timestamp', 
        'is_failing_next_7_days', 
        'time_to_failure_hrs'
    ]

    # AUTO-DETECTION: Find any other string/object columns that shouldn't be there
    # (e.g., 'failure_type', 'maintenance_type' from previous merges)
    string_cols = train_df.select_dtypes(include=['object', 'string']).columns.tolist()
    
    # Combine lists (set avoids duplicates)
    drop_cols = list(set(mandatory_drop + string_cols))
    
    print(f"    Correctly dropping leakage/metadata columns: {drop_cols}", flush=True)

    # Verify columns exist before dropping to avoid errors
    existing_drop_cols = [c for c in drop_cols if c in train_df.columns]

    X_train = train_df.drop(columns=existing_drop_cols)
    y_train = train_df['is_failing_next_7_days']
    
    X_test = test_df.drop(columns=existing_drop_cols)
    y_test = test_df['is_failing_next_7_days']

    print(f"    Features used: {X_train.shape[1]} columns", flush=True)

    # --- MODEL 1: LOGISTIC REGRESSION ---
    print("\n>>> [2/5] Scaling data for Logistic Regression...", flush=True)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(">>> [3/5] Training Logistic Regression (Baseline)...", flush=True)
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    
    # Evaluate LR
    y_pred_lr = lr_model.predict(X_test_scaled)
    lr_report = classification_report(y_test, y_pred_lr)
    print("\n--- Logistic Regression Baseline Results ---", flush=True)
    print(lr_report, flush=True)

    # --- MODEL 2: XGBOOST ---
    print("\n>>> [4/5] Training XGBoost (Champion)... This may take a minute...", flush=True)
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1  # Use all cores
    )
    xgb_model.fit(X_train, y_train)

    import matplotlib.pyplot as plt

    # This is important to find exactly if any column is "cheating"
    # I previously found "time_to_failure_hrs " so I included on the drop cols list    
    print("\n>>> Feature Importance Check:")
    # Get feature importance
    importances = pd.Series(xgb_model.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)

    print(importances.head(10))
    
    # Evaluate XGB
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    xgb_report = classification_report(y_test, y_pred_xgb)
    
    print("\n--- XGBoost Champion Results ---", flush=True)
    print(xgb_report, flush=True)
    print(f"    ROC AUC Score: {roc_auc_score(y_test, y_prob_xgb):.4f}", flush=True)

    # --- SAVE RESULTS ---
    print("\n>>> [5/5] Saving results to file...", flush=True)
    output_file = os.path.join(OUTPUT_DIR, 'model_comparison.txt')
    with open(output_file, 'w') as f:
        f.write("LOGISTIC REGRESSION:\n" + lr_report + "\n")
        f.write("XGBOOST:\n" + xgb_report + f"\nROC AUC: {roc_auc_score(y_test, y_prob_xgb):.4f}")

    print(f"DONE! Results saved in {output_file}", flush=True)

if __name__ == "__main__":
    train_and_evaluate()