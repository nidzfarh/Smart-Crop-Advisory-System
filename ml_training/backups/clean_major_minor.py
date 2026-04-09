import pandas as pd
import re

csv_path = 'ml_training/datasets/crop_recommendation.csv'
df = pd.read_csv(csv_path)

# Scrub the exact string artifacts User explicitly reported
def clean_text(text):
    if pd.isna(text): return text
    text = str(text)
    # Remove (major), (minor), and emojis
    text = re.sub(r'(?i)\(major\)', '', text)
    text = re.sub(r'(?i)\(minor\)', '', text)
    text = re.sub(r'major', '', text, flags=re.IGNORECASE)
    text = re.sub(r'minor', '', text, flags=re.IGNORECASE)
    text = text.replace('✅', '').replace('⚠️', '')
    return text.strip()

# Clean visual artifacts from all categorical columns
df['district'] = df['district'].apply(clean_text)
df['crop'] = df['crop'].apply(clean_text)
df['soil_type'] = df['soil_type'].apply(clean_text)
df['label'] = df['label'].apply(clean_text)

df.to_csv(csv_path, index=False)
print("Sanitized 'major' & 'minor' text formatting from the dataset natively.")
