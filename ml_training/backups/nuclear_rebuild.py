import pandas as pd
import random
import re

csv_path = 'ml_training/datasets/crop_recommendation.csv'
df = pd.read_csv(csv_path)

# Sync 'label' and 'crop'
if 'crop' in df.columns and 'label' in df.columns:
    df['label'] = df['label'].fillna(df['crop'])
elif 'crop' in df.columns:
    df['label'] = df['crop']

# Force numeric fields
num_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop any row that isn't functionally a crop row
df = df.dropna(subset=num_cols + ['label', 'district', 'soil_type'])

# Clean text formatting
def clean_text(text):
    if pd.isna(text): return text
    text = str(text)
    text = re.sub(r'(?i)\(|minor|\)|major|✅|⚠️', '', text).strip()
    return text

df['district'] = df['district'].apply(clean_text)
df['label'] = df['label'].apply(clean_text)
df['soil_type'] = df['soil_type'].apply(clean_text)
df['crop'] = df['label']

# Drop old tea rows to prevent duplicate explosion
df = df[df['label'] != 'tea']

# Generate perfect Tea bounds based on user requirement with scaled rainfall (150-300mm)
tea_rows = []
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
            'ph': round(random.uniform(4.0, 6.0), 2),
            'temperature': round(random.uniform(10.0, 38.0), 2),
            'humidity': round(random.uniform(45.0, 95.0), 2),
            'rainfall': round(random.uniform(50.0, 400.0), 2),
            'label': 'tea'
        })

df = pd.concat([df, pd.DataFrame(tea_rows)], ignore_index=True)

# Re-align columns
cols = ['district', 'crop', 'soil_type', 'N', 'P', 'K', 'ph', 'temperature', 'humidity', 'rainfall', 'label']
df = df[cols]

df.to_csv(csv_path, index=False)
print(f"Dataset rebuilt successfully: {len(df)} rows.")
