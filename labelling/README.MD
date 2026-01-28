# Data Labelling Pipeline

This directory contains the logic for transforming raw time-series telemetry into a labeled dataset suitable for Supervised Machine Learning.

## Overview
The primary goal of this script is to create the **Target Variable**: `is_failing_next_7_days`. This is a binary flag (0 or 1) that indicates whether a specific machine will experience a catastrophic failure within the next 168 hours.

## Key Features

### 1. Domain-Specific "Physics" Cleanup
While general cleaning handles statistical outliers, this script enforces physical constraints relevant to industrial washing equipment:
* **Temperature:** Clipped to $[0, 100]$°C.
* **Vibration:** Clipped to a minimum of $0$, as negative magnitudes are physically impossible sensor readings.

### 2. The `merge_asof` Strategy
Because telemetry (recorded every 15 mins) and failures (recorded at the exact moment of breakdown) do not share the same timestamps, we use a **"Forward Merge"**.
* The script looks ahead from every sensor reading to find the *nearest future failure* for that specific machine ID.
* A `tolerance` of 30 days is applied to optimize processing and ensure we aren't matching failures from the distant future.

### 3. Leakage Prevention
To ensure the ML model remains valid for real-world production:
* All helper columns used for calculation (like the exact `failure_timestamp`) are dropped before the final file is saved.
* The model only "knows" what happened up to the current timestamp.

## Sorting Requirements
**Important:** `merge_asof` is highly sensitive to data order. This script implements a **Global Chronological Sort**. Both datasets are sorted by `timestamp` first and `machine_id` second. This ensures the search engine moves forward monotonically across the entire 3.5M row dataset.

## Output
The final labeled dataset is saved to `data/labelled/telemetry_labeled.csv`. This file includes all original sensor features plus the `is_failing_next_7_days` column, ready for Feature Engineering.

```bash
╭─ Target Variable Distribution ─╮
│  Total Rows   8,784            │
│  Healthy (0)  8,010  91.19%    │
│  Failing (1)  774    8.81%     │
╰───── 7-Day Failure Window ─────╯
```

