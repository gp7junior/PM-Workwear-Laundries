from feast import FeatureStore
import pandas as pd
from datetime import datetime

# 1. Connect to your local store
store = fs = FeatureStore(repo_path=".")

# 2. Define who we want to look up and at what time
# Note: Use a machine_id and timestamp that actually exists in your final_features.parquet

query_time = datetime.now().replace(tzinfo=None)

entity_df = pd.DataFrame.from_dict({
    "machine_id": ["MCH_044"], 
    "timestamp": [pd.to_datetime("2024-02-07 06:56:00")]
})

# 3. Pull features from the Offline Store
print("Pulling historical features...")
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "machine_health_features:vibration_rolling_max_7d",
        "machine_health_features:hours_since_maintenance",
        "machine_health_features:loc_Warehouse A",
        "machine_health_features:loc_Warehouse B",
        "machine_health_features:loc_Warehouse C",
        "machine_health_features:loc_Warehouse D",
        "machine_health_features:error_code_count"
    ],
    full_feature_names=True
).to_df()

if training_df.empty:
    print("\n❌ STILL EMPTY: No data found for this ID/Timestamp combination.")
    print("Check if machine_id exists and if timestamp is AFTER the data starts.")
else:
    print("\n✅ SUCCESS: Features Found!")
    print(training_df.transpose()) # Transpose makes it easier to read one row