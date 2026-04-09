import pandas as pd
import random
import os

# Ensure we are in the right directory or use absolute paths?
# Using relative to CWD which should be the project root.

# DATA CONFIG (sync with generate_data.py)
KERALA_DISTRICTS = [
    'Thiruvananthapuram', 'Kollam', 'Pathanamthitta', 'Alappuzha', 'Kottayam', 
    'Idukki', 'Ernakulam', 'Thrissur', 'Palakkad', 'Malappuram', 
    'Kozhikode', 'Wayanad', 'Kannur', 'Kasaragod'
]

KERALA_SOIL_TYPES = ['Laterite', 'Sandy', 'Clayey', 'Loamy', 'Alluvial', 'Red']

CROP_SOIL_MAP = {
    'Black Pepper': ['Laterite', 'Red', 'Loamy'],
    'Cocoa': ['Laterite', 'Loamy'],
    'Mango': ['Laterite', 'Alluvial', 'Red'],
    'Papaya': ['Loamy', 'Alluvial'],
    'Pineapple': ['Laterite', 'Sandy', 'Loamy'],
    'Tapioca': ['Laterite', 'Sandy', 'Red'],
    'Ginger': ['Loamy', 'Laterite'],
    'Turmeric': ['Loamy', 'Laterite', 'Alluvial'],
    'Clove': ['Laterite', 'Loamy'],
    'Nutmeg': ['Laterite', 'Loamy'],
    'Cinnamon': ['Laterite', 'Sandy'],
    'Vanilla': ['Laterite', 'Loamy']
}

NEW_CROPS = list(CROP_SOIL_MAP.keys())

# Ideal values for generation
CROP_IDEALS = {
    'Black Pepper': {'N': 60, 'P': 40, 'K': 100, 'temp': 28, 'humidity': 80, 'ph': 5.5, 'rainfall': 250},
    'Cocoa': {'N': 40, 'P': 40, 'K': 60, 'temp': 27, 'humidity': 85, 'ph': 6.0, 'rainfall': 200},
    'Mango': {'N': 80, 'P': 40, 'K': 80, 'temp': 32, 'humidity': 60, 'ph': 6.5, 'rainfall': 100},
    'Papaya': {'N': 100, 'P': 80, 'K': 120, 'temp': 30, 'humidity': 70, 'ph': 6.5, 'rainfall': 150},
    'Pineapple': {'N': 50, 'P': 40, 'K': 100, 'temp': 25, 'humidity': 80, 'ph': 5.0, 'rainfall': 150},
    'Tapioca': {'N': 40, 'P': 30, 'K': 100, 'temp': 30, 'humidity': 70, 'ph': 5.5, 'rainfall': 150},
    'Ginger': {'N': 80, 'P': 60, 'K': 100, 'temp': 28, 'humidity': 80, 'ph': 6.0, 'rainfall': 200},
    'Turmeric': {'N': 80, 'P': 60, 'K': 100, 'temp': 28, 'humidity': 80, 'ph': 6.0, 'rainfall': 200},
    'Clove': {'N': 40, 'P': 40, 'K': 40, 'temp': 26, 'humidity': 85, 'ph': 5.5, 'rainfall': 250},
    'Nutmeg': {'N': 40, 'P': 40, 'K': 40, 'temp': 26, 'humidity': 85, 'ph': 5.5, 'rainfall': 250},
    'Cinnamon': {'N': 40, 'P': 40, 'K': 40, 'temp': 26, 'humidity': 85, 'ph': 5.5, 'rainfall': 250},
    'Vanilla': {'N': 30, 'P': 20, 'K': 30, 'temp': 28, 'humidity': 85, 'ph': 6.0, 'rainfall': 250}
}

def add_recommendation_data(n_per_crop=250):
    csv_path = 'ml_training/datasets/crop_recommendation.csv'
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    new_data = []
    
    # Columns expected: district,crop,soil_type,N,P,K,ph,temperature,humidity,rainfall,label
    for crop in NEW_CROPS:
        ideal = CROP_IDEALS[crop]
        for _ in range(n_per_crop):
            district = random.choice(KERALA_DISTRICTS)
            soil_type = random.choice(CROP_SOIL_MAP[crop])
            
            n = ideal['N'] + random.uniform(-10, 10)
            p = ideal['P'] + random.uniform(-10, 10)
            k = ideal['K'] + random.uniform(-10, 10)
            ph = ideal['ph'] + random.uniform(-0.5, 0.5)
            temp = ideal['temp'] + random.uniform(-3, 3)
            hum = ideal['humidity'] + random.uniform(-10, 10)
            rain = ideal['rainfall'] + random.uniform(-30, 30)
            
            # Match existing column order: district,crop,soil_type,N,P,K,ph,temperature,humidity,rainfall,label
            new_data.append([district, crop, soil_type, n, p, k, ph, temp, hum, rain, crop])
            
    new_df = pd.DataFrame(new_data, columns=df.columns)
    final_df = pd.concat([df, new_df], ignore_index=True)
    final_df.to_csv(csv_path, index=False)
    print(f"Added {len(new_data)} rows to crop_recommendation.csv")

def add_yield_data(n_per_crop=200):
    csv_path = 'ml_training/datasets/yield_data.csv'
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    new_data = []
    
    # Columns: Crop,N,P,K,Temperature,Rainfall,Yield_Tons_Acre
    for crop in NEW_CROPS:
        ideal = CROP_IDEALS[crop]
        
        base_yield = 1.0
        if crop == 'Tapioca': base_yield = 12.0
        elif crop == 'Papaya': base_yield = 15.0
        elif crop == 'Mango': base_yield = 5.0
        elif crop == 'Pineapple': base_yield = 20.0
        elif crop == 'Black Pepper': base_yield = 0.5
        elif crop in ['Vanilla', 'Clove', 'Nutmeg', 'Cinnamon']: base_yield = 0.3
        
        for _ in range(n_per_crop):
            n = ideal['N'] + random.uniform(-10, 10)
            p = ideal['P'] + random.uniform(-10, 10)
            k = ideal['K'] + random.uniform(-10, 10)
            temp = ideal['temp'] + random.uniform(-3, 3)
            rain = ideal['rainfall'] + random.uniform(-30, 30)
            
            yield_val = base_yield * random.uniform(0.8, 1.2) + (n+p+k)/2000
            new_data.append([crop, n, p, k, temp, rain, max(0.1, yield_val)])
            
    new_df = pd.DataFrame(new_data, columns=df.columns)
    final_df = pd.concat([df, new_df], ignore_index=True)
    final_df.to_csv(csv_path, index=False)
    print(f"Added {len(new_data)} rows to yield_data.csv")

if __name__ == "__main__":
    add_recommendation_data()
    add_yield_data()
