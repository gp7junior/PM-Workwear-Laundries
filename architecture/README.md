# Solution Architecture: PM Workwear Laundries

This document outlines the end-to-end MLOps architecture for predicting laundry machine failures. The solution is designed to maximize Recall (99%), ensuring that high-cost machine downtime is minimized through a "Dual-Loop" predictive system.

```mermaid
---
config:
      theme: redux
---
flowchart TD
    classDef process fill:#e1bee7,stroke:#4a148c,stroke-width:2px,color:#000;
    classDef storage fill:#dbf0f9,stroke:#01579b,stroke-width:2px,color:#000;
    subgraph MainMedallion [Data Sources - Medallion]
        style MainMedallion fill: #419ccd,stroke:#827717 
        direction TB
        IoT["Laundry Machines IoT Agg."]
        MTL["Maintenance <br> Logs"]
        IoT -- Streaming Telemetry --> Kafka{{Kafka <br> Event Hubs}}:::process
        Kafka -- Raw Ingestion --> Bronze[(Bronze Layer <br> Data Lake)]:::storage
        Bronze -- Clean & Validate --> Silver[(Silver Layer <br> Data Lake)]:::storage
        MTL -- Batch ETL --> Silver:::storage
        Silver -- Join & Aggregate --> Gold[(Gold Layer <br> Aggregated)]:::storage
        end
    
    %% --- Feature Engineering & Store ---
    subgraph FeatureStore [Feature Store eg. Feast]
        style FeatureStore fill: #3AB681,stroke:#827717
        %% FRegistry{{Feature Registry <br> Definitions}}
        
        %% Offline Path
        Gold -- Scheduled Batch Job --> BatchProc[Spark Batch <br> Processor]:::process
        BatchProc -- Write Historical Features --> OfflineStore[(Offline Store <br> BigQuery/S3)]:::storage
        
        %% Online Path
        Kafka -- Real time Event --> StreamProc[Flink Stream Processor]:::process
        StreamProc -- Update Rolling Windows --> OnlineStore[(Online Store <br> Redis/DynamoDB)]:::storage
        OfflineStore -- Periodic Sync --> OnlineStore
    end

    %% --- Training Pipeline ---
    subgraph Training [ML Training Pipeline]
        style Training fill: #d3d3d3,stroke:#827717
        Orchestrator[Airflow Orchestrator]:::process --> TriggerTrain((Trigger))
        TriggerTrain -- 1. Request Data --> OfflineStore
        OfflineStore -- 2. Point in Time Join --> TrainModel[Model Training <br> XGBoost]:::process
        TrainModel -- 3. Register Champion --> ModelReg{{MLflow Model <br> Registry}}
    end
```

## 1. Data Foundations & The Medallion Flow
To maintain high data quality for our XGBoost model, we utilize a Medallion Architecture. This ensures that raw sensor "noise" is filtered before reaching the Feature Store.

```mermaid
graph LR
    %% Class Definitions
    classDef ingestion fill:#f9f,stroke:#333,stroke-width:2px;
    classDef storage fill:#dbf0f9,stroke:#01579b,stroke-width:2px;
    classDef compute fill:#fff3e0,stroke:#ff6f00,stroke-width:2px;

    %% Ingestion Source
    Source1[IoT Sensors]:::ingestion
    Source2[Maintenance Logs]:::ingestion

    %% Bronze Layer
    subgraph DataLake [Data Lake]
        direction LR
        Bronze[(Bronze Layer: <br/> Raw Files)]:::storage
        Silver[(Silver Layer: <br/> Validated Data)]:::storage
        Gold[(Gold Layer: <br/> Feature Ready)]:::storage
    end

    %% Process Flow
    Source1 -->|Raw Stream| Bronze
    Source2 -->|Batch ETL| Bronze

    Bronze -->|Task 1: Cleaning & <br/> Physics Clipping| Silver
    Silver -->|Task 2 & 3: Joins & <br/> Rolling Aggregates| Gold

    %% Output to Offline Store
    Gold -->|Managed by Feast| OfflineStore[Offline Feature Store]:::compute

    style DataLake fill:#f4f4f4,stroke:#333,stroke-dasharray: 5 5
```
**Bronze (Ingestion):** Captures raw, immutable telemetry from Kafka.

**Silver (Validation):** Applies physics-based constraints (e.g., temperature clipping) and deduplication. Maintenance logs are integrated here to provide historical context.

**Gold (Aggregated):** Features are finalized for ML consumption, including pre-calculated health indicators and cumulative runtime stats.

## 2. The Feature Store (Feast Implementation)

The Feature Store acts as the "Single Source of Truth," eliminating the risk of Training-Serving Skew by using identical logic for both historical and live data.

| Feature        | Offline Path (Training)          | Online Path (Inference)         |
|----------------|----------------------------------|---------------------------------|
| Logic Source   | Gold Layer (S3/Parquet)          | Kafka / Streaming Telemetry     |
| Processing     | Spark Batch / Pandas             | Flink or Spark Streaming        |
| Feast Role     | Point-in-Time Joins (Historical) | Low-Latency Serving (Real-time) |
| Storage Medium | Offline Store (Data Lake)        | Online Store                    |
| Technology	 | S3 / Parquet / BigQuery	        | Redis / SQLite / DynamoDB       |

# 3. Training & Model Registry
**Orchestration:** Managed by Apache Airflow, which schedules retraining based on time intervals or "Drift" triggers.

**Registry:** The MLflow Model Registry manages versioning. Every model is tagged as either a *"Challenger"* (under evaluation) or *"Champion"* (production-active).

# 4. Deployment & Real-Time Inference
This phase describes how the live production environment interacts with the Feature Store to generate actionable alerts.

```mermaid

---
config:
      theme: redux
---
flowchart
    classDef process fill:#e1bee7,stroke:#4a148c,stroke-width:2px,color:#000;
    classDef storage fill:#dbf0f9,stroke:#01579b,stroke-width:2px,color:#000;
    classDef serving fill:#c8e6c9,stroke:#1b5e20,stroke-width:2px,color:#000;

    %% --- Data Sources ---
        IoT["Laundry Machines IoT Agg."]

    %% --- Training Pipeline ---
    subgraph Training [ML Training Pipeline]
        style Training fill: #d3d3d3,stroke:#827717
        MLF(Retrain the Model):::process
        ORQ(Airflow Orchestrator):::process
    end

    %% --- Inference & Deployment ---   
        API[Model Serving API K8s/Docker]:::serving
        Dashboard[Alerting Dashboard]:::source
        PrF{Prediction <br> > Threshold}:::process
        DMn{Performance Drop <br> or Data Drift <br>Detected ?}:::process
        MLA(ML Engine <br> XGBoost):::serving
        ONS[(Online Store)]:::storage

    %% --- Production    Loop ---
    IoT  -- 0. Current Sensor Reading           --> API
    API <-- 1. Low Latency Lookup               --> ONS
    API  -- 2.1 Current & <br> Historical data  --> MLA
    MLA  -- 2.2 Generates <br> Prediction       --> API
    API  -- 3. Predict Failure Prob.            --> PrF
    PrF  -- 3. Alerts                           --> Dashboard
    API  -- 4. Inference Data                   --> DMn
    %% --- Monitoring Loop ---
    DMn  -- Triggers                            --> ORQ
    ORQ                                         --> MLF
    MLF  -- 6. Register New Model               --> MLA
    
    linkStyle 1 stroke-width:2px,fill:none,stroke:blue;
    linkStyle 2 stroke-width:2px,fill:none,stroke:blue;
    linkStyle 3 stroke-width:2px,fill:none,stroke:blue;
    linkStyle 4 stroke-width:2px,fill:none,stroke:blue;
    linkStyle 5 stroke-width:2px,fill:none,stroke:blue;
    linkStyle 6 stroke-width:2px,fill:none,stroke:blue;
    linkStyle 7 stroke-width:2px,fill:none,stroke:red;
    linkStyle 8 stroke-width:2px,fill:none,stroke:red;
    linkStyle 9 stroke-width:2px,fill:none,stroke:red;

```

## 4.1 Prediction Workflow (Blue Path)

**Request:** Machine sends a payload (e.g., Temperature, Current Vibration).

**Enrichment:** API queries the Online Store (Redis/SQLite) for the 7-day historical features (e.g., *vibration_rolling_max_7d*).

**Inference:** Data is passed to the XGBoost Engine.

```
IF Prediction Probability > 0.45 THEN 
    Dispatch Maintenance Alert to Dashboard.
```

## 4.2 Monitoring & Retraining (Red Path)

**Drift Detection:** Input features are monitored. If the average vibration of the fleet shifts >2σ, the system flags Data Drift.

```
IF Data Drift detected OR Recall < 85% THEN 
    Signal Airflow to launch a new Training Job.
```

**Registry Sync:** The new model is registered and promoted to the ML Engine.

# 5. Operational Maintenance
## 5.1 CI/CD & Shadow Deployment

New models are deployed in Shadow Mode first. They process live data without sending alerts to technicians. We only promote a *"Challenger"* to *"Champion"* once it proves it maintains 99% Recall on real-world distributions.

## 5.2 The Feedback Loop

Technician findings (e.g., "Actual Part Failure" vs "False Alarm") are logged in the maintenance app. This data is piped back into the Gold Layer, creating a "Ground Truth" label for the next month's training cycle, ensuring the model evolves with the hardware.

## Tech Stack Summary

| Component      | Tool / Technology    | Role in Your Project                          |
|----------------|----------------------|-----------------------------------------------|
| Ingestion      | Kafka / Event Hubs   | Sensor data streaming.          |
| Processing     | Spark / Pandas       | **Task 1** & **3** cleaning and rolling features.     |
| Data Lake      | Delta Lake / Parquet | Medallion storage (Bronze/Silver/Gold).       |
| Feature Store  | Feast                | The registry and manager for feature logic.   |
| Offline Store  | S3 / Parquet         | Historical data for Point-in-Time joins.      |
| Online Store   | Redis / SQLite       | Low-latency storage for the Inference API.    |
| Model Registry | MLflow               | Versioning the XGBoost "Champion" models.    |
| Orchestration  | Apache Airflow       | Triggering **Task 2** labels and **Task 6** training. |
| Deployment     | FastAPI / Docker     | Serving the model as a microservice.          |