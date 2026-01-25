from feast import FeatureStore
import pprint

# Online Path testing script

# 1. Connect to the store
store = FeatureStore(repo_path=".")

# 2. Define the machine we want to inspect
# This represents a request coming from a sensor in the factory
entity_rows = [
    {"machine_id": "MCH_044"} # Use a machine_id you know exists
]

print(f"--- Fetching Real-Time Features for {entity_rows[0]['machine_id']} ---")

# 3. Pull from the ONLINE Store (SQLite)
try:
    online_features = store.get_online_features(
        features=[
            "machine_health_features:vibration_rolling_max_7d",
            "machine_health_features:hours_since_maintenance",
            "machine_health_features:loc_Warehouse A",
            "machine_health_features:error_code_count",
            "machine_health_features:temperature_c"
        ],
        entity_rows=entity_rows
    ).to_dict()

    # 4. Show the result
    pprint.pprint(online_features)
    
except Exception as e:
    print(f"\n❌ Error: {e}")