import pandas as pd
import random

# Load the existing pure dataset
csv_path = 'ml_training/datasets/crop_recommendation.csv'
df = pd.read_csv(csv_path)

# Synthetic Tea Data bounds
tea_bounds = {
    "N": (80, 110),
    "P": (25, 35),
    "K": (50, 70),
    "pH": (4.5, 5.5),
    "temperature": (20, 28),
    "humidity": (75, 90),
    "rainfall": (2000, 3000)
}

tea_targets = [
    ("Idukki", "forest"),
    ("Wayanad", "forest")
]

new_rows = []
# heavily weight tea in these districts to ensure it triggers (generate 500 rows each)
for dist, soil in tea_targets:
    for _ in range(500):
        new_rows.append({
            'N': random.randint(*tea_bounds['N']),
            'P': random.randint(*tea_bounds['P']),
            'K': random.randint(*tea_bounds['K']),
            'temperature': round(random.uniform(*tea_bounds['temperature']), 2),
            'humidity': round(random.uniform(*tea_bounds['humidity']), 2),
            'ph': round(random.uniform(*tea_bounds['pH']), 2),
            'rainfall': round(random.uniform(*tea_bounds['rainfall']), 2),
            'label': 'tea',
            'district': dist,
            'soil_type': soil
        })

tea_df = pd.DataFrame(new_rows)
# Merge and overwrite the dataset
updated_df = pd.concat([df, tea_df], ignore_index=True)
updated_df.to_csv(csv_path, index=False)

print(f"Injected {len(new_rows)} highly weighed rows of Tea for Idukki and Wayanad.")
