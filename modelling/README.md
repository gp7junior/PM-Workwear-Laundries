# Data Splitting & Imbalance Strategy

This module prepares the engineered feature set for machine learning by performing a rigorous **Time-Based Split** and calculating **Class Weights**.

## 1. Time-Series Splitting Strategy
Unlike standard classification problems, predictive maintenance requires strict temporal separation between training and testing data to prevent **Data Leakage**.

### The "No-Look-Ahead" Rule
We strictly enforce that the model is trained only on *past* data and evaluated on *future* data.
* **Method:** We sort the entire dataset chronologically.
* **Split Point:** The first **80%** of the timeline is designated for **Training**. The remaining **20%** is held out for **Testing**.
* **Why:** Random shuffling (e.g., `train_test_split(shuffle=True)`) would mix future observations into the training set, allowing the model to "cheat" by memorizing future failure patterns, leading to unrealistically high performance that fails in production.



## 2. Handling Class Imbalance
Industrial failure data is inherently imbalanced (e.g., 98% healthy vs. 2% failure). To address this without generating synthetic data (which can introduce noise in time-series), we use **Cost-Sensitive Learning**.

### The `scale_pos_weight`
We calculate the ratio of Negative samples (Healthy) to Positive samples (Failures) in the Training set:

$$\text{scale\_pos\_weight} = \frac{\text{Count(Healthy)}}{\text{Count(Failures)}}$$

This value is saved to `train_metadata.json` and passed to the XGBoost/LightGBM model. It effectively penalizes the model heavily for missing a failure (False Negative), forcing it to prioritize the minority class.

## Output Files
* **`train.csv`**: Historical data (first 80%) used for model fitting.
* **`test.csv`**: Future data (last 20%) used for final evaluation metrics (Recall/Precision).
* **`train_metadata.json`**: Contains the calculated class weight and split statistics for reproducible training.

-----------

### First run:

```
bash
LOGISTIC REGRESSION:
              precision    recall  f1-score   support

           0       0.68      0.59      0.63     19923
           1       0.29      0.37      0.33      8940

    accuracy                           0.52     28863
   macro avg       0.48      0.48      0.48     28863
weighted avg       0.56      0.52      0.54     28863

XGBOOST:
              precision    recall  f1-score   support

           0       0.67      0.73      0.70     19923
           1       0.26      0.21      0.24      8940

    accuracy                           0.57     28863
   macro avg       0.47      0.47      0.47     28863
weighted avg       0.55      0.57      0.56     28863

ROC AUC: 0.4504
```

