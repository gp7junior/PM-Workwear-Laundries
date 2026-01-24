import pandas as pd
import json
import os
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# --- CONFIGURATION ---
TRAIN_PATH = 'data/modelling/train.csv'
TEST_PATH = 'data/modelling/test.csv'
BEST_PARAMS_PATH = 'data/results/best_params.json'

def get_final_metrics():
    # 1. Load and Prepare Data
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

    # 2. Logistic Regression (Baseline)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)

    # 3. XGBoost (Champion)
    with open(BEST_PARAMS_PATH, 'r') as f:
        best_params = json.load(f)
    
    xgb_model = XGBClassifier(**best_params)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

    # 4. Final Terminal Output
    print("\n" + "="*30)
    print("LOGISTIC REGRESSION RESULTS")
    print("="*30)
    print(classification_report(y_test, y_pred_lr))

    print("\n" + "="*30)
    print("XGBOOST CHAMPION RESULTS")
    print("="*30)
    print(classification_report(y_test, y_pred_xgb))
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob_xgb):.4f}")

if __name__ == "__main__":
    get_final_metrics()