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

# Predictive Maintenance: Final Model Report

## 1. Metric Performance Summary
The following table compares our linear baseline against the optimized Gradient Boosting model on the held-out test set (future data).

| Metric              | Logistic Regression (Baseline) | XGBoost (Champion) |
|---------------------|--------------------------------|--------------------|
| Recall (Class 1)    | 37%                            | 99%                |
| Precision (Class 1) | 29%                            | 31%                |
| F1-Score (Class 1)  | 0.33                           | 0.47               |
| Overall Accuracy    | 52%                            | 31%                |
| ROC AUC             | 0.48                           | 0.48               |

## 2. Business Context & Justification
In the workwear laundry industry, the cost of machine failure is asymmetrical.

- Cost of a False Negative (Missed Failure): Catastrophic. Results in unplanned downtime, broken SLAs, emergency repair costs, and potential workwear backlog.

- Cost of a False Positive (False Alarm): Low. Results in a technician performing a 15-minute inspection on a healthy machine.

**Why XGBoost is Selected for Production?**

Despite the lower overall accuracy, the Tuned XGBoost model is the superior choice for production deployment for the following reasons:

**Maximum Risk Mitigation:** A 99% Recall ensures that nearly every failing component is flagged before a breakdown occurs. The Baseline model misses 63% of failures, which is unacceptable for a critical operation.

**Superior Identification of Complex Patterns:** The XGBoost model successfully utilized rolling features (vibration and power strain) to identify failures that a linear model (Logistic Regression) could not separate.

**Operational Strategy**: While the model is "aggressive" (high False Positive rate), it acts as a highly effective safety net. The business can now move from Reactive Maintenance to a Predictive Inspection workflow.

## 3. Final Recommendation
Deploy the Tuned XGBoost Model. To optimize the operational load on technicians, we recommend a phased rollout where the probability threshold is monitored. However, given the primary directive to avoid downtime, the current configuration provides the highest level of protection for the facility's uptime.

# A Second Study: The 48h Window Analysis

| Metric    | 7-Day Strategy (Baseline)  | 48-Hour Strategy (New)     | The MLE Analysis                                                                                                                                |
|-----------|----------------------------|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| **ROC AUC**   | 0.52(Random Guess)         | 0.60 - 0.69(Signal Found!) | The Big Win. Improved the "Discriminative Power" by ~15-30%. The 7-day model was guessing; the 48-hour model is actually learning patterns. |
| **Recall**    | 99%(Artificial)            | 67%(Realistic)             | The 7-day model achieved 99% by flagging everything as a failure (high false alarms). The 48-hour model misses some but is far more discerning. |
| **Precision** | 30%                        | 10%                        | The Cost. Because the 48h target is so rare (harder to hit), precision dropped. This is the trade-off for a sharper signal.                     |
| **Winner**    | Logistic Regression (0.54) | Logistic Regression (0.69) | Surprise Insight: Simple linear models work best on this short-term signal. XGBoost might be overfitting.                                       |


1. When moving from 7 days to 48 hours, the ROC AUC jumped from 0.52 to 0.69 (on the baseline). This confirms the hypothesis: the failure signals in this equipment are short-term. The 7-day window was drowning the signal in noise."

2. Seing a Recall drop from 99% to 67% might lead to the thinking that the model got worse. It didn't. The 7-day model was like a fire alarm that rings 24/7—it catches every fire, but is useless because nobody trusts it. The 48-hour model captures 2 out of 3 failures with a significantly higher ability to rank risk (AUC), making it a much better foundation for a real alert system."

3. Interestingly, the simpler Logistic Regression (0.69 AUC) beat the complex XGBoost (0.59 AUC) in the 48-hour test. This suggests that the indicators of immediate failure (like temperature spikes) are likely linear and strong. If I were deploying this today, I would actually choose the Logistic Regression model for its interpretability and superior performance on this specific window.