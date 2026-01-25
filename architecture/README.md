# Solution Architecture for PM Workwear Laundries

The following sections describes a concept to the proposed ML Solution to predict when a laundry machine is most likely to break, providing an on time alert indicating the necessity of maintenance and avoiding the high costs of a machine change. 

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
        IoT["Lundry Machines IoT Agg."]
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

<!-- ![Arch1](architecture_1.svg "Architecture Part 1") -->

## Diagram Explanation

### 1.1 Data Sources & Ingestion

- Laundry Machines (IoT) Agg.: The source of truth. Streams raw sensor data (temperature, vibration) constantly.

- Kafka Event Hub: The central nervous system buffering high-velocity streaming data.

### 1.2 The Data Lake "Medallion" Flow (Blue Subgraph)

- Bronze: Raw, immutable data dumped straight from Kafka.

- Silver: Cleaned data. Physics constraints applied (e.g., temp clipped at 100°C). Maintenance logs are joined here.

- Gold: Highly aggregated data ready for machine learning (e.g., pre-calculated daily summaries).

### 2. The Feature Store (Green Subgraph)

This is the critical bridge that solves the "training-serving skew" problem.

**The Offline Path:**

- Uses Batch Processors (Spark) to calculate heavy historical features from the Gold Layer. Stores them in the Offline Store (S3/BigQuery).

- Crucial Function: Performs Point-in-Time Joins for training, ensuring the model only sees data that existed before a failure occurred (preventing leakage).

** The Online Path:**

- Uses Stream Processors (Flink) to calculate rolling windows (e.g., "Max Vibration Last 7 Days") in near real-time straight from Kafka.

- Updates the Online Store (Redis) immediately. This store is optimized for sub-millisecond retrieval, not massive storage.

### 3. ML Training Pipeline (Gray Subgraph)

- Orchestrator (Airflow): manages the schedule.
It pulls historical data from the Offline Store to train the XGBoost model.

- The final "Champion" model is saved to the Model Registry (MLflow), versioned and ready for deployment.

### 5. Deployment & Inference

This image describes the live production architecture and how it triggers the creation of new models.

<!-- ![Arch1](architecture_2.svg "Architecture Part 2") -->
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
        IoT["Lundry Machines IoT Agg."]

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

**Prediction Workflow:** (blue arrows)

0. A machine sends its **current sensor reading** to the API. 
1. The API instantly **queries** the Online Store (Redis) for the machine's 7-day historical context.
2. Combining **current** and **historical** data, passes it to the loaded XGBoost model, and generates a **prediction**. 
3.
    ``` 
    If prediction > the threshold:
        send an alert to the Dashboard.
    ```

**The Monitoring Loop** (red arrows)

4. Drift Detection: The API sends **input data** and **predictions** to a monitoring system. 
5. If the incoming data starts looking different from the training data (Data Drift), or if Recall drops (Concept Drift), it signals the Orchestrator to trigger a re-training cycle.
    ```
    If data_drift or recall drops:
       trigger a re-training cycle
    ``` 
6. The new Model is then registered on the Model Register (ML Enginee)