from feast import Entity, FeatureView, Field, FileSource, PushSource
from feast.types import Float32, Int64
from datetime import timedelta

# 1. Define the Entity (The "Who")
machine = Entity(name="machine_id", join_keys=["machine_id"])

# 2. Define the Sources (The "Where")
# This points to the GOLD layer for training
historical_source = FileSource(
    path="data/modelling/gold_features.parquet",
    event_timestamp_column="timestamp"
)

# This is the "Speed Lane" for real-time Kafka data
online_push_source = PushSource(
    name="telemetry_push_source",
    batch_source=historical_source
)

# 3. Define the Feature View (The "What")
# This is where your Task 3 logic is officially registered
machine_stats_view = FeatureView(
    name="machine_health_features",
    entities=[machine],
    ttl=timedelta(days=7), # Only keep the last 7 days online
    schema=[
        Field(name="vibration_rolling_max_7d", dtype=Float32),
        Field(name="temp_rolling_std_7d", dtype=Float32),
        Field(name="power_rolling_mean_7d", dtype=Float32),
        Field(name="hours_since_maintenance", dtype=Float32),
    ],
    online=True,
    source=online_push_source,
)