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

