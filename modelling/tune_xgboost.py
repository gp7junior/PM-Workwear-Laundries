import pandas as pd
import numpy as np
import optuna
import json
import os
import xgboost as xgb
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
TRAIN_PATH = 'data/modelling/train.csv'
TEST_PATH = 'data/modelling/test.csv'
META_PATH = 'data/modelling/train_metadata.json'
OUTPUT_DIR = 'data/results/'

def load_data():
    """Loads and prepares data, removing the known leak columns."""
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    # LOAD METADATA (For the base scale_pos_weight)
    with open(META_PATH, 'r') as f:
        meta = json.load(f)
    base_weight = meta['scale_pos_weight']

    # DROP COLUMNS (Including the Leak we found!)
    # We re-define the drop list here to be safe
    drop_cols = ['machine_id', 'timestamp', 'is_failing_next_7_days', 'time_to_failure_hrs']
    
    # Auto-detect strings
    string_cols = train_df.select_dtypes(include=['object', 'string']).columns.tolist()
    final_drop = list(set(drop_cols + string_cols))
    
    # Prepare X and y
    X = train_df.drop(columns=[c for c in final_drop if c in train_df.columns])
    y = train_df['is_failing_next_7_days']
    
    X_test_final = test_df.drop(columns=[c for c in final_drop if c in test_df.columns])
    y_test_final = test_df['is_failing_next_7_days']
    
    return X, y, X_test_final, y_test_final, base_weight

def objective(trial, X_train, y_train, X_val, y_val, base_weight):
    """
    The Optimization Function: Optuna tries to maximize the return value (Recall).
    """
    # 1. Suggest Hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        # We allow Optuna to adjust the class weight around our calculated base
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', base_weight * 0.5, base_weight * 3.0),
        'random_state': 42,
        'n_jobs': -1
    }
    
    # 2. Train Model
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # 3. Predict on Validation Set
    y_pred = model.predict(X_val)
    
    # 4. Calculate Metric (Recall)
    recall = recall_score(y_val, y_pred)
    
    return recall

def run_optimization():
    # 1. Load Data
    X_full, y_full, X_test, y_test, base_weight = load_data()
    
    # 2. Internal Time-Split for Tuning
    # We use the first 80% of the TRAINING set for fitting, last 20% for tuning validation
    split_idx = int(len(X_full) * 0.8)
    X_train_tune = X_full.iloc[:split_idx]
    y_train_tune = y_full.iloc[:split_idx]
    X_val_tune = X_full.iloc[split_idx:]
    y_val_tune = y_full.iloc[split_idx:]
    
    print(f"Tuning on {len(X_train_tune)} rows, Validating on {len(X_val_tune)} rows.")
    
    # 3. Create Study
    print("Starting Optuna Optimization (maximize Recall)...")
    study = optuna.create_study(direction='maximize', study_name='xgboost_tuning_predictive_maintenance')
    
    # We use a lambda to pass our specific data arguments to the objective
    func = lambda trial: objective(trial, X_train_tune, y_train_tune, X_val_tune, y_val_tune, base_weight)
    
    # Run for 20 trials (keep it small for speed)
    study.optimize(func, n_trials=200)
    
    print("\n" + "="*40)
    print("BEST PARAMETERS FOUND")
    print("="*40)
    print(study.best_params)
    print(f"Best Validation Recall: {study.best_value:.4f}")
    
    # 4. Train Final Champion on FULL Train Set
    print("\nRetraining Final Champion Model on full training data...")
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_full, y_full)
    
    # 5. Final Evaluation on TEST Set
    y_pred_test = final_model.predict(X_test)
    y_prob_test = final_model.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*40)
    print("FINAL TEST SET RESULTS")
    print("="*40)
    print(f"Recall:    {recall_score(y_test, y_pred_test):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_test):.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_test, y_prob_test):.4f}")
    
    # Save parameters for MLOps
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"Best parameters saved to {OUTPUT_DIR}best_params.json")

if __name__ == "__main__":
    run_optimization()