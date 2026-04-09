import pandas as pd
import numpy as np
import random
import os
from datetime import datetime

# Ensure datasets directory exists
os.makedirs('ml_training/datasets', exist_ok=True)

# FULL KERALA CROP LIST
KERALA_CROPS = [
    'Coconut', 'Rubber', 'Black Pepper', 'Cardamom', 'Coffee', 'Tea', 'Cocoa', 'Arecanut',
    'Banana', 'Mango', 'Papaya', 'Pineapple', 'Jackfruit', 'Guava', 'Sapota', 'Custard Apple',
    'Rice', 'Tapioca', 'Maize', 'Cowpea', 'Green gram', 'Black gram', 'Ginger', 'Turmeric',
    'Clove', 'Nutmeg', 'Cinnamon', 'Vanilla', 'Bitter gourd', 'Snake gourd', 'Ash gourd',
    'Brinjal', 'Okra', 'Tomato', 'Chilli', 'Cucumber'
]
KERALA_SOIL_TYPES = ['Laterite', 'Sandy', 'Clayey', 'Loamy', 'Alluvial', 'Red']
KERALA_DISTRICTS = [
    'Thiruvananthapuram', 'Kollam', 'Pathanamthitta', 'Alappuzha', 'Kottayam', 
    'Idukki', 'Ernakulam', 'Thrissur', 'Palakkad', 'Malappuram', 
    'Kozhikode', 'Wayanad', 'Kannur', 'Kasaragod'
]
KERALA_CROPS = [
    'Coconut', 'Rubber', 'Black Pepper', 'Cardamom', 'Coffee', 'Tea', 'Cocoa', 'Arecanut',
    'Banana', 'Mango', 'Papaya', 'Pineapple', 'Jackfruit', 'Guava', 'Sapota', 'Custard Apple',
    'Rice', 'Tapioca', 'Maize', 'Cowpea', 'Green gram', 'Black gram', 'Ginger', 'Turmeric',
    'Clove', 'Nutmeg', 'Cinnamon', 'Vanilla', 'Bitter gourd', 'Snake gourd', 'Ash gourd',
    'Brinjal', 'Okra', 'Tomato', 'Chilli', 'Cucumber'
]

# 1. Crop Recommendation Dataset
def generate_crop_recommendation_data(n_samples=5000):
    import os
    import sys
    sys.path.append(os.path.abspath('.'))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "crop_project.settings")
    import django
    from django.conf import settings
    if not settings.configured:
        django.setup()
        
    data = []
    
    CROP_SOIL_MAP = {
        'Rice': ['Clayey', 'Alluvial'],
        'Coconut': ['Sandy', 'Laterite', 'Alluvial'],
        'Rubber': ['Laterite', 'Red'],
        'Black Pepper': ['Laterite', 'Red', 'Loamy'],
        'Cardamom': ['Laterite', 'Loamy'],
        'Coffee': ['Laterite', 'Red', 'Loamy'],
        'Tea': ['Laterite', 'Loamy'],
        'Cocoa': ['Laterite', 'Loamy'],
        'Arecanut': ['Laterite', 'Alluvial', 'Loamy'],
        'Banana': ['Alluvial', 'Loamy'],
        'Mango': ['Laterite', 'Alluvial', 'Red'],
        'Papaya': ['Loamy', 'Alluvial'],
        'Pineapple': ['Laterite', 'Sandy', 'Loamy'],
        'Jackfruit': ['Laterite', 'Red', 'Loamy'],
        'Guava': ['Laterite', 'Alluvial'],
        'Sapota': ['Laterite', 'Alluvial', 'Loamy'],
        'Custard Apple': ['Laterite', 'Red', 'Sandy'],
        'Tapioca': ['Laterite', 'Sandy', 'Red'],
        'Maize': ['Loamy', 'Alluvial', 'Red'],
        'Cowpea': ['Loamy', 'Sandy', 'Alluvial'],
        'Green gram': ['Loamy', 'Alluvial'],
        'Black gram': ['Loamy', 'Alluvial', 'Clayey'],
        'Ginger': ['Loamy', 'Laterite'],
        'Turmeric': ['Loamy', 'Laterite', 'Alluvial'],
        'Clove': ['Laterite', 'Loamy'],
        'Nutmeg': ['Laterite', 'Loamy'],
        'Cinnamon': ['Laterite', 'Sandy'],
        'Vanilla': ['Laterite', 'Loamy'],
        'Bitter gourd': ['Loamy', 'Sandy', 'Alluvial'],
        'Snake gourd': ['Loamy', 'Sandy', 'Alluvial'],
        'Ash gourd': ['Loamy', 'Sandy', 'Alluvial'],
        'Brinjal': ['Loamy', 'Alluvial', 'Clayey'],
        'Okra': ['Loamy', 'Alluvial'],
        'Tomato': ['Loamy', 'Alluvial', 'Red'],
        'Chilli': ['Loamy', 'Alluvial', 'Red'],
        'Cucumber': ['Loamy', 'Sandy', 'Alluvial']
    }

    # Deterministic generation of "ideal" clusters so they are unique across the 36 crops
    random.seed(42)
    crop_ideals = {}
    for crop in KERALA_CROPS:
        crop_ideals[crop] = {
            'N': random.randint(20, 120),
            'P': random.randint(20, 100),
            'K': random.randint(20, 120),
            'temp': random.uniform(20.0, 35.0),
            'humidity': random.uniform(50.0, 95.0),
            'ph': random.uniform(5.0, 7.5),
            'rainfall': random.uniform(50.0, 300.0)
        }
    
    # Hardcode a few known ones for realism
    crop_ideals['Rice'] = {'N': 90, 'P': 50, 'K': 40, 'temp': 26, 'humidity': 80, 'ph': 6.0, 'rainfall': 200}
    crop_ideals['Rubber'] = {'N': 60, 'P': 50, 'K': 50, 'temp': 28, 'humidity': 85, 'ph': 5.5, 'rainfall': 250}
    crop_ideals['Coconut'] = {'N': 50, 'P': 50, 'K': 100, 'temp': 30, 'humidity': 80, 'ph': 6.5, 'rainfall': 150}
    
    for _ in range(n_samples):
        crop = random.choice(KERALA_CROPS)
        ideal = crop_ideals[crop]
        
        # Select realistic soil type for crop, introducing a 10% chance of random noise 
        if random.random() < 0.9:
            soil_type = random.choice(CROP_SOIL_MAP.get(crop, KERALA_SOIL_TYPES))
        else:
            soil_type = random.choice(KERALA_SOIL_TYPES)

        N = ideal['N'] + random.uniform(-15, 15)
        P = ideal['P'] + random.uniform(-10, 10)
        K = ideal['K'] + random.uniform(-15, 15)
        temp = ideal['temp'] + random.uniform(-3, 3)
        humidity = ideal['humidity'] + random.uniform(-10, 10)
        ph = ideal['ph'] + random.uniform(-0.5, 0.5)
        rainfall = ideal['rainfall'] + random.uniform(-30, 30)
        district = random.choice(KERALA_DISTRICTS)
        
        data.append([N, P, K, temp, humidity, ph, rainfall, soil_type, district, crop])
        
    df = pd.DataFrame(data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'soil_type', 'district', 'label'])
    df.to_csv('ml_training/datasets/crop_recommendation.csv', index=False)
    print("Generated crop_recommendation.csv")

# 2. Yield Prediction Dataset
def generate_yield_data(n_samples=3000):
    data = []
    for _ in range(n_samples):
        crop = random.choice(KERALA_CROPS)
        N, P, K = random.randint(40, 120), random.randint(20, 100), random.randint(20, 100)
        temp, rain = random.uniform(20, 35), random.uniform(50, 300)
        
        base_yield = 1.5
        if crop == 'Rice': base_yield = 4.0
        elif crop == 'Rubber': base_yield = 0.8 # Tons per acre dry rubber
        elif crop == 'Banana': base_yield = 15.0 # Tons per acre
        elif crop == 'Tapioca': base_yield = 12.0
        elif crop == 'Black Pepper': base_yield = 0.5
        elif crop == 'Pineapple': base_yield = 20.0
        elif crop in ['Tomato', 'Cucumber']: base_yield = 10.0
        
        yield_val = base_yield * (random.uniform(0.8, 1.2)) + (N+P+K)/1000
        data.append([crop, N, P, K, temp, rain, max(0.1, yield_val)])
        
    df = pd.DataFrame(data, columns=['Crop', 'N', 'P', 'K', 'Temperature', 'Rainfall', 'Yield_Tons_Acre'])
    df.to_csv('ml_training/datasets/yield_data.csv', index=False)
    print("Generated yield_data.csv")

# 3. Market Analysis Data
def generate_market_history_data(years=5):
    data = []
    current_year = datetime.now().year
    
    # Prices in ₹ per Ton
    prices = {
        'Rice': 30000, 'Rubber': 160000, 'Black Pepper': 500000, 'Cardamom': 1200000,
        'Coffee': 180000, 'Tea': 140000, 'Cocoa': 220000, 'Arecanut': 450000,
        'Banana': 25000, 'Mango': 60000, 'Papaya': 20000, 'Pineapple': 30000,
        'Jackfruit': 15000, 'Guava': 40000, 'Sapota': 35000, 'Custard Apple': 50000,
        'Tapioca': 12000, 'Maize': 20000, 'Cowpea': 45000, 'Green gram': 75000,
        'Black gram': 70000, 'Ginger': 80000, 'Turmeric': 90000, 'Clove': 800000,
        'Nutmeg': 500000, 'Cinnamon': 400000, 'Vanilla': 15000000, # Vanilla is super expensive
        'Bitter gourd': 35000, 'Snake gourd': 25000, 'Ash gourd': 20000,
        'Brinjal': 30000, 'Okra': 35000, 'Tomato': 25000, 'Chilli': 80000, 'Cucumber': 20000,
        'Coconut': 40000
    }
    
    for crop in KERALA_CROPS:
        base_price = prices.get(crop, 30000)
        for year in range(current_year - years, current_year + 1):
            for month in range(1, 13):
                price_mod, demand = 1.0, 'Medium'
                # Seasonality
                if crop == 'Banana' and month == 8: price_mod, demand = 1.4, 'High'
                if crop == 'Ginger' and month in [12, 1]: price_mod, demand = 1.2, 'High'
                if crop == 'Mango' and month in [4, 5]: price_mod, demand = 0.7, 'High'
                
                price = base_price * price_mod * random.uniform(0.9, 1.1)
                data.append([crop, month, year, round(price, 2), demand])
                
    df = pd.DataFrame(data, columns=['Crop', 'Month', 'Year', 'Price_Per_Ton', 'Demand_Level'])
    df.to_csv('ml_training/datasets/market_history.csv', index=False)
    print("Generated market_history.csv")

# 4. Seed Rate & Price Data
def generate_seed_rate_data():
    seed_info = {
        'Rice': [20, 'kg', 45], 'Rubber': [150, 'saplings', 120], 'Black Pepper': [800, 'cuttings', 25],
        'Cardamom': [400, 'suckers', 60], 'Coffee': [1000, 'saplings', 15], 'Tea': [4000, 'clones', 10],
        'Cocoa': [200, 'saplings', 40], 'Arecanut': [500, 'seedlings', 50], 'Banana': [1000, 'suckers', 20],
        'Mango': [40, 'grafts', 150], 'Papaya': [1000, 'seeds', 5], 'Pineapple': [15000, 'suckers', 8],
        'Jackfruit': [30, 'saplings', 100], 'Guava': [100, 'saplings', 80], 'Sapota': [100, 'saplings', 120],
        'Custard Apple': [100, 'saplings', 100], 'Tapioca': [4000, 'cuttings', 2], 'Maize': [8, 'kg', 130],
        'Cowpea': [10, 'kg', 100], 'Green gram': [8, 'kg', 120], 'Black gram': [10, 'kg', 110], 
        'Ginger': [600, 'kg', 60], 'Turmeric': [600, 'kg', 50], 'Clove': [100, 'saplings', 150],
        'Nutmeg': [100, 'saplings', 200], 'Cinnamon': [1000, 'seedlings', 30], 'Vanilla': [1000, 'cuttings', 100],
        'Bitter gourd': [2, 'kg', 2500], 'Snake gourd': [2, 'kg', 1800], 'Ash gourd': [2, 'kg', 1500],
        'Brinjal': [0.2, 'kg', 12000], 'Okra': [4, 'kg', 2000], 'Tomato': [0.1, 'kg', 45000],
        'Chilli': [0.5, 'kg', 8000], 'Cucumber': [1, 'kg', 5000], 'Coconut': [70, 'seedlings', 100]
    }
    
    data = [[crop] + info for crop, info in seed_info.items()]
    df = pd.DataFrame(data, columns=['Crop', 'Seed_Rate_Per_Acre', 'Seed_Unit', 'Seed_Price_Per_Unit'])
    df.to_csv('ml_training/datasets/seed_rates.csv', index=False)
    print("Generated seed_rates.csv")

if __name__ == "__main__":
    generate_crop_recommendation_data()
    generate_yield_data()
    generate_market_history_data()
    generate_seed_rate_data()
