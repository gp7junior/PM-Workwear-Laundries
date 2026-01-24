# PM-Workwear-Laundries

## The Objective

The core of this solution is a Predictive Maintenance (PdM) engine. By leveraging sensor data from a large fleet of advanced washing machines and drying equipment, we have built a machine learning pipeline designed to:

- Forecast Potential Failures: Detect subtle anomalies in machine telemetry that precede a breakdown.

- Optimize Scheduling: Empower service teams to perform maintenance during off-peak hours, ensuring zero impact on daily laundry throughput.

- Reduce Operational Costs: Lower the high costs associated with emergency repairs, technician overtime, and equipment idling.

## Data Description (Hypothetical) 

Use a scalable language (Python/PySpark/Scala) and is provided with the full dataset.

| Dataset                      | Data Fields                                                                                                 | Description    
|------------------------------|-------------------------------------------------------------------------------------------------------------|---------------------------------------------------|
| Telemetry Data (Time-Series) | machine_id, timestamp, temperature_c, vibration_hz, pressure_bar, power_kw, runtime_hours, error_code_count | Sensor readings recorded every 15 minutes. (Approx. 3.5 Million Rows) | 
| Failure Log                  | machine_id, failure_timestamp, failure_type                                                                 | Historical records of catastrophic failures.                          | 
| Maintenance Log              | machine_id, maintenance_timestamp, maintenance_type, parts_replaced                                         | Records of scheduled and unscheduled maintenance.                     |   
| Machine Metadata             | machine_id, model_year, location, age_months, capacity_kg                                                   | Static machine characteristics.       

# Solution Guide

## Task 1: Data Loading and Initial Cleanup

Implement the necessary code (Python/Pandas/Spark) to load all four datasets. Implement basic error handling for outliers (e.g., temperatures outside 0°C to 100°C) by setting them to the median, and ensure all timestamp columns are correctly formatted. Output: Cleaned DataFrames and schema documentation. 

### Solution

The main steps regarding the Data Loading and Cleaning are inside the <<data_cleaning>> folder. 

how to run

```bash
PM-Workwear-Laundries % python3 data_cleaning/clean_data.py \
  --input data/raw/telemetry_data_full.csv \ 
  --output data/cleaned/telemetry_data_full.csv \              
  --timestamp-col timestamp \
  --freq 1min \
  --outlier-method iqr \
  --outlier-action mark
```

to run in a folder:

```bash
python data_cleaning/clean_data.py --input-dir data/raw --output-dir data/cleaned --timestamp-col timestamp --freq 1min --outlier-action median
```

## Task 2: Defining the Target Variable (Labeling) 

Implement the logic to create the binary target variable (is_failing_next_7_days). This is a critical step: 
- for every row in the Telemetry data, check if that machine fails within the next 7 days (168 hours). 
- Crucially, ensure no look-ahead leakage. 
- Output: The labeling function/code and a descriptive statistic of the resulting target variable imbalance (e.g., percentage of positive labels). 

### Solution

to run 
```bash
python3 labelling/pipeline_labeling.py
```

## Task 3: Feature Construction 

Implement code to calculate at least eight (8) high-quality features. These must combine data from Telemetry and Metadata: 
 
1. Five (5) rolling window statistics (e.g., mean, max, standard deviation) over a 7-day window for continuous sensor variables (Vibration, Temperature, Pressure). 
 
2. Three (3) binary, categorical, or time-based features (e.g., Time since last major maintenance, Machine age bin, One-Hot Encoding of location). 

### Solution

```
bash
python3 feature_construction/feature_engineering.py
```

