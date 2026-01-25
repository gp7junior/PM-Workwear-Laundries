import pandas as pd
df = pd.read_parquet("../data/features/final_features.parquet")
print(df[['machine_id', 'timestamp']].head(1))