import pandas as pd
import random

csv_path = 'ml_training/datasets/crop_recommendation.csv'
df = pd.read_csv(csv_path)

# Remove any existing tea rows just to be absolutely clean
df = df[df['label'] != 'tea']

tea_rows = []

# Distribute weightings: Idukki (major), Wayanad (minor)
# Soil: Forest (primary), Loamy (secondary)
distributions = [
    ('Idukki', 'forest', 400),
    ('Idukki', 'loamy', 150),
    ('Wayanad', 'forest', 150),
    ('Wayanad', 'loamy', 50)
]

for district, soil_type, count in distributions:
    for _ in range(count):
        tea_rows.append({
            'district': district,
            'crop': 'tea',
            'soil_type': soil_type,
            'N': random.randint(60, 120),
            'P': random.randint(20, 40),
            'K': random.randint(40, 80),
            'ph': round(random.uniform(4.5, 5.5), 2),
            'temperature': round(random.uniform(18.0, 30.0), 2),
            'humidity': round(random.uniform(70.0, 90.0), 2),
            'rainfall': round(random.uniform(1500.0, 3000.0), 2),
            'label': 'tea'
        })

tea_df = pd.DataFrame(tea_rows)

# Enforce exact header alignment
cols = ['district', 'crop', 'soil_type', 'N', 'P', 'K', 'ph', 'temperature', 'humidity', 'rainfall', 'label']

# Combine
updated_df = pd.concat([df, tea_df], ignore_index=True)

# Just in case there are NaN errors from previous steps, drop them on the critical columns
updated_df = updated_df.dropna(subset=['N', 'P', 'K', 'ph', 'temperature', 'humidity', 'rainfall', 'label', 'district', 'soil_type'])

# Ensure order
updated_df = updated_df[cols]

updated_df.to_csv(csv_path, index=False)
print(f"Successfully generated {sum([c[2] for c in distributions])} pristine Tea samples mapped to Idukki and Wayanad!")
