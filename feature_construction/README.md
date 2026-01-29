# Feature Engineering Pipeline

This module transforms the raw, labeled telemetry data into a rich feature set suitable for predictive modeling.

## 1. Rolling Window Statistics
We calculate statistics over a **7-Day Sliding Window** (approx. 672 observations of 15-minute intervals). This captures the *trend* rather than just the instantaneous state of the machine.

| Feature | Type | Reasoning |
| :--- | :--- | :--- |
| `vibration_rolling_mean_7d` | Mean | Sustained high vibration indicates bearing wear. |
| `vibration_rolling_max_7d` | Max | Captures extreme shocks that might not affect the mean. |
| `temp_rolling_std_7d` | Std Dev | High variance suggests thermostat instability or cooling failure. |
| `pressure_rolling_mean_7d` | Mean | Drops in average pressure indicate leaks. |
| `power_rolling_max_7d` | Max | Spikes in power suggest the motor is straining against resistance. |

## 2. Maintenance History
We calculate **`hours_since_maintenance`** using a backward-looking `merge_asof`.
* **Logic:** At any given timestamp $t$, we find the most recent maintenance timestamp $t_{maint}$ such that $t_{maint} < t$.
* **Leakage Prevention:** We strictly look *backward*. Future maintenance events are not visible to the model.

## 3. Static Metadata
We integrate machine-specific characteristics:
* **Age Group:** Binning machines into `New` (<2 years), `Mid-Age` (2-5 years), and `Legacy` (>5 years).
* **Location:** One-Hot Encoded to capture site-specific environmental factors (humidity, water quality).

## Data Integrity Note
* **Sorting:** The pipeline strictly enforces Global Chronological Sorting (`timestamp` then `machine_id`) to ensure `merge_asof` functions correctly.
* **Warm-up Period:** The first 7 days of data for every machine are dropped, as they lack sufficient history to calculate the rolling window statistics.

## How to Run This

For your Standard 7-Day Baseline (Default):

```bash
python feature_engineering.py
# Output: data/features/final_features.csv
```

## For 48-Hour Strategy:

```bash
python feature_engineering.py --window 48
# Output: data/features/final_features_48h.csv
```