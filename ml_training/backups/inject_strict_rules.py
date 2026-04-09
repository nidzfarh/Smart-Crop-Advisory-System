import pandas as pd
import random
import os

df = pd.read_csv("ml_training/datasets/crop_recommendation.csv")

# Ensure columns align perfectly
df.columns = df.columns.str.strip()
if 'pH' in df.columns:
    df.rename(columns={'pH': 'ph'}, inplace=True)

# 1. Purge data that violates the strict pairs!
# (We want the model to learn ONLY valid pairs)

rules = [
    # Top Crops (Massive weight: 300 rows)
    ('Thiruvananthapuram', 'laterite', 'coconut', 300),
    ('Thiruvananthapuram', 'sandy', 'coconut', 300),
    ('Kollam', 'laterite', 'cashew', 300),
    ('Pathanamthitta', 'laterite', 'rubber', 300),
    ('Alappuzha', 'alluvial', 'rice', 300),
    ('Alappuzha', 'clayey', 'rice', 300),
    ('Kottayam', 'laterite', 'rubber', 300),
    ('Idukki', 'forest', 'cardamom', 300),
    ('Ernakulam', 'laterite', 'coconut', 300),
    ('Ernakulam', 'sandy', 'coconut', 300),
    ('Thrissur', 'alluvial', 'rice', 300),
    ('Thrissur', 'clayey', 'rice', 300),
    ('Palakkad', 'alluvial', 'rice', 300),
    ('Palakkad', 'clayey', 'rice', 300),
    ('Malappuram', 'laterite', 'coconut', 300),
    ('Malappuram', 'sandy', 'coconut', 300),
    ('Kozhikode', 'laterite', 'coconut', 300),
    ('Kozhikode', 'sandy', 'coconut', 300),
    ('Wayanad', 'forest', 'coffee', 300),
    ('Kannur', 'laterite', 'coconut', 300),
    ('Kannur', 'sandy', 'coconut', 300),
    ('Kasaragod', 'laterite', 'coconut', 300),
    ('Kasaragod', 'sandy', 'coconut', 300),

    # Secondary Crops (High weight: 150 rows)
    ('Thiruvananthapuram', 'laterite', 'rubber', 150),
    ('Thiruvananthapuram', 'alluvial', 'banana', 150),
    ('Thiruvananthapuram', 'loamy', 'banana', 150),
    ('Kollam', 'sandy', 'coconut', 150),
    ('Kollam', 'laterite', 'rubber', 150),
    ('Pathanamthitta', 'forest', 'pepper', 150),
    ('Pathanamthitta', 'laterite', 'pepper', 150),
    ('Pathanamthitta', 'alluvial', 'banana', 150),
    ('Alappuzha', 'sandy', 'coconut', 150),
    ('Kottayam', 'sandy', 'coconut', 150),
    ('Kottayam', 'forest', 'pepper', 150),
    ('Kottayam', 'forest', 'cardamom', 150),
    ('Idukki', 'forest', 'pepper', 150),
    ('Idukki', 'forest', 'coffee', 150),
    ('Ernakulam', 'alluvial', 'banana', 150),
    ('Ernakulam', 'loamy', 'tomato', 150), # Veg
    ('Ernakulam', 'loamy', 'bitter gourd', 150), # Veg
    ('Thrissur', 'laterite', 'coconut', 150),
    ('Thrissur', 'alluvial', 'banana', 150),
    ('Palakkad', 'alluvial', 'sugarcane', 150),
    ('Palakkad', 'loamy', 'sugarcane', 150),
    ('Palakkad', 'loamy', 'tomato', 150), # Veg
    ('Palakkad', 'alluvial', 'tomato', 150), # Veg
    ('Malappuram', 'laterite', 'arecanut', 150),
    ('Malappuram', 'alluvial', 'banana', 150),
    ('Kozhikode', 'alluvial', 'banana', 150),
    ('Kozhikode', 'forest', 'pepper', 150),
    ('Kozhikode', 'laterite', 'pepper', 150),
    ('Wayanad', 'forest', 'pepper', 150),
    ('Wayanad', 'alluvial', 'rice', 150),
    ('Wayanad', 'clayey', 'rice', 150),
    ('Kannur', 'laterite', 'cashew', 150),
    ('Kannur', 'forest', 'pepper', 150),
    ('Kannur', 'laterite', 'pepper', 150),
    ('Kasaragod', 'laterite', 'arecanut', 150),
    ('Kasaragod', 'forest', 'pepper', 150),
    ('Kasaragod', 'laterite', 'pepper', 150)
]

crop_profiles = {
    'coconut': (40, 20, 40, 27, 80, 200, 6.0),
    'cashew': (30, 15, 25, 28, 70, 180, 5.5),
    'rubber': (50, 20, 30, 26, 85, 250, 5.0),
    'rice': (80, 40, 40, 25, 80, 220, 6.0),
    'cardamom': (40, 30, 40, 22, 85, 300, 5.5),
    'coffee': (60, 30, 40, 24, 80, 200, 5.5),
    'pepper': (50, 40, 40, 25, 80, 250, 5.5),
    'banana': (80, 40, 80, 28, 75, 180, 6.5),
    'tomato': (60, 40, 40, 25, 70, 150, 6.5),
    'bitter gourd': (50, 40, 40, 26, 75, 160, 6.0),
    'sugarcane': (100, 50, 60, 30, 75, 250, 6.5),
    'arecanut': (50, 20, 40, 27, 85, 280, 5.5)
}

syn_rows = []
for d, s, c, count in rules:
    base = crop_profiles[c]
    for _ in range(count):
        syn_rows.append({
            'district': d,
            'soil_type': s,
            'crop': c,
            'N': max(10, base[0] + random.randint(-15, 15)),
            'P': max(10, base[1] + random.randint(-10, 10)),
            'K': max(10, base[2] + random.randint(-15, 15)),
            'temperature': base[3] + random.uniform(-3, 3),
            'humidity': base[4] + random.uniform(-10, 10),
            'rainfall': base[5] + random.uniform(-50, 50),
            'ph': base[6] + random.uniform(-0.5, 0.5)
        })

syn_df = pd.DataFrame(syn_rows)

# Merge datasets, but filter out bad geographical mappings from the original data
valid_pairs = set([(r[0].lower(), r[1].lower(), r[2].lower()) for r in rules])

def is_valid_original(row):
    d = str(row['district']).lower().strip()
    s = str(row['soil_type']).lower().strip()
    c = str(row['crop']).lower().strip()
    # If the district is one of our strict targets, but the crop contradicts our top maps, reject it to let the synthetic data own it completely.
    if d in [r[0].lower() for r in rules]:
        # Loosely accept it if it's one of the user's allowed crops
        allowed_crops = set([r[2] for r in rules if r[0].lower() == d])
        if c not in allowed_crops:
            return False # Reject bad data like 'Rice' in Idukki entirely
    return True

df_clean = df[df.apply(is_valid_original, axis=1)]

final = pd.concat([df_clean, syn_df], ignore_index=True)
final.to_csv("ml_training/datasets/crop_recommendation.csv", index=False)
print(f"Purged invalid rows. Injected {len(syn_df)} heavily prioritized facts.")
