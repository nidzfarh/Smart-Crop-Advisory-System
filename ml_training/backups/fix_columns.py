import pandas as pd

csv_path = 'ml_training/datasets/crop_recommendation.csv'
df = pd.read_csv(csv_path)

if 'crop' in df.columns:
    df['crop'] = df['crop'].fillna(df['label'])
else:
    df['crop'] = df['label']

# Enforce column order to match the user's exact specification
cols = ['district', 'crop', 'soil_type', 'N', 'P', 'K', 'ph', 'temperature', 'humidity', 'rainfall', 'label']

# Keep only those columns and in that exact order
df = df[cols]

# Overwrite dataset
df.to_csv(csv_path, index=False)
print("Dataset columns aligned and correctly ordered!")
