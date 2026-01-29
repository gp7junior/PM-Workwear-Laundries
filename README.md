# PM-Workwear-Laundries

**Table of Contents:**
- [PM-Workwear-Laundries](#pm-workwear-laundries)
  - [The Objective](#the-objective)
  - [Data Description (Hypothetical)](#data-description-hypothetical)
- [Solution Guide](#solution-guide)
  - [Running the Code](#running-the-code)
  - [Task 1: Data Loading and Initial Cleanup](#task-1-data-loading-and-initial-cleanup)
  - [Task 2: Defining the Target Variable (Labeling)](#task-2-defining-the-target-variable-labeling)
  - [Task 3: Feature Construction](#task-3-feature-construction)
  - [Task 4: Data Imbalance and Splitting](#task-4-data-imbalance-and-splitting)
  - [Task 5: Model Implementation \& Training](#task-5-model-implementation--training)
  - [Task 6: Hyperparameter Tuning](#task-6-hyperparameter-tuning)
  - [Task 7: Evaluation Metric \& Reporting](#task-7-evaluation-metric--reporting)
- [3. Solution Architecture \& MLOps](#3-solution-architecture--mlops)
  - [Task 8: Feature Store Design (Detailed)](#task-8-feature-store-design-detailed)
  - [Task 9: Production MLOps Workflow Document](#task-9-production-mlops-workflow-document)

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

## Running the Code

To run the predictive maintenance engine, follow these steps:

1. **Set Up the Environment**: Ensure you have Python 3.14 or higher installed. You can create a virtual environment using the following command:
   
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**: Navigate to the project directory and install the required packages listed in `requirements.txt`:
   
   ```bash
   pip install -r requirements.txt
   ```

## Task 1: Data Loading and Initial Cleanup

Implement the necessary code (Python/Pandas/Spark) to load all four datasets. Implement basic error handling for outliers (e.g., temperatures outside 0°C to 100°C) by setting them to the median, and ensure all timestamp columns are correctly formatted. Output: Cleaned DataFrames and schema documentation. 

**Solution**

The steps regarding the Data Loading and Cleaning are inside the `data_cleaning` folder. 

how to run

```bash
python3 data_cleaning/clean_data.py \
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

**Solution**

Please find the code for this solution inside the `labelling` folder.
```bash
python3 labelling/pipeline_labeling.py
```

## Task 3: Feature Construction 

Implement code to calculate at least eight (8) high-quality features. These must combine data from Telemetry and Metadata: 
 
1. **Five (5) rolling window statistics** (e.g., mean, max, standard deviation) over a 7-day window for continuous sensor variables (Vibration, Temperature, Pressure). 
 
2. **Three (3) binary, categorical**, or **time-based** features (e.g., Time since last major maintenance, Machine age bin, One-Hot Encoding of location). 

**Solution**:
Folder `feature_construction`
```bash
python3 feature_construction/feature_engineering.py
```

## Task 4: Data Imbalance and Splitting 
Implement a strategy to handle the imbalanced nature of the labeled dataset (e.g., using class weights in the model, or implementing a mild over/undersampling technique). Clearly define and implement a time-based train/test split (e.g., using the first 80% of time for training and the last 20% for testing) to rigorously prevent time-series leakage. 
2. Model Development, Optimization, & Evaluation  

**Solution**:
Files `labelling/pipeline_labelling.py` and `modelling/data_splitting.py`
```bash
python3 labelling/pipeline_labelling.py
python3 modelling/data_splitting.py
```
Output is saved on `data/labelled` and `data/modelling`

## Task 5: Model Implementation & Training 
Implement and train two contrasting machine learning models: a Gradient Boosting Classifier (XGBoost/LightGBM) and a simple Logistic Regression. Use the processed data from Section 1. 

**Solution**
Files `modelling/train_models.py`
```bash
python3 modelling/train_models.py`
```
Output is saved on `data/results/model_comparison.txt`

## Task 6: Hyperparameter Tuning 
Perform basic hyperparameter tuning (e.g., a small Randomized Search or Bayesian Optimization) on the Gradient Boosting model to optimize performance specifically for the Recall metric.

**Solution**
Files `modelling/tune_xgboost.py`
```bash
python3 modelling/tune_xgboost.py`
```
Output is saved on `data/results/best_params.json`


## Task 7: Evaluation Metric & Reporting 
Prioritize the Recall metric due to the high cost of False Negatives. 
Output: Provide the final classification report (including Precision, Recall, F1-Score, and AUC) for both models on the held-out test set. 
Justify which model is best suited for production deployment based on the achieved metrics and the business context (high False Negative cost). 

**Solution**
Files `modelling/final_evaluation.py`
```bash
python3 modelling/final_evaluation.py`
```
Output is showed on terminal, best paramenters are saved on `data/results/best_params.json`

# 3. Solution Architecture & MLOps  

## Task 8: Feature Store Design (Detailed)
Design the full conceptual workflow for the Feature Store (e.g., using Feast or a cloud-native store like SageMaker/Databricks). Detail how features are computed for offline training (historical data) and how they are retrieved for online inference (real-time prediction) using distinct data paths. 

## Task 9: Production MLOps Workflow Document 
Create a detailed written plan (document format) outlining the end-to-end MLOps workflow for continuous deployment and retraining. Include: 
 
1. Data Pipeline: Continuous ingestion (e.g., Kafka/Event Hub) into the Bronze layer. 
2. Feature Pipeline: Scheduled feature calculation and population of the Feature Store. 
3. Training Pipeline: Triggering model retraining based on data drift or performance drop. 
4. Deployment: How the champion model is containerized and exposed as a low-latency API endpoint.

**Solution Tasks 8 & 9**

For more details on the architecture and data flow, refer to the architecture documentation in the `architecture/` folder.



