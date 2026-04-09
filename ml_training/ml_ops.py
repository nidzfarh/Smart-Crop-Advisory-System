import pandas as pd
import numpy as np
import random
import os
import joblib
import re
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import csv

# ==========================================
# 1. CORE DATA GENERATION (from generate_data.py)
# ==========================================
def generate_base_datasets():
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
    
    # 1.1 Crop Recommendation
    def generate_rec_sub(n_samples=5000):
        CROP_SOIL_MAP = {
            'Rice': ['Clayey', 'Alluvial'], 'Coconut': ['Sandy', 'Laterite', 'Alluvial'],
            'Rubber': ['Laterite', 'Red'], 'Black Pepper': ['Laterite', 'Red', 'Loamy'],
            'Cardamom': ['Laterite', 'Loamy'], 'Coffee': ['Laterite', 'Red', 'Loamy'],
            'Tea': ['Laterite', 'Loamy'], 'Cocoa': ['Laterite', 'Loamy'],
            'Arecanut': ['Laterite', 'Alluvial', 'Loamy'], 'Banana': ['Alluvial', 'Loamy'],
            'Mango': ['Laterite', 'Alluvial', 'Red'], 'Papaya': ['Loamy', 'Alluvial'],
            'Pineapple': ['Laterite', 'Sandy', 'Loamy'], 'Jackfruit': ['Laterite', 'Red', 'Loamy'],
            'Guava': ['Laterite', 'Alluvial'], 'Sapota': ['Laterite', 'Alluvial', 'Loamy'],
            'Custard Apple': ['Laterite', 'Red', 'Sandy'], 'Tapioca': ['Laterite', 'Sandy', 'Red'],
            'Maize': ['Loamy', 'Alluvial', 'Red'], 'Cowpea': ['Loamy', 'Sandy', 'Alluvial'],
            'Green gram': ['Loamy', 'Alluvial'], 'Black gram': ['Loamy', 'Alluvial', 'Clayey'],
            'Ginger': ['Loamy', 'Laterite'], 'Turmeric': ['Loamy', 'Laterite', 'Alluvial'],
            'Clove': ['Laterite', 'Loamy'], 'Nutmeg': ['Laterite', 'Loamy'],
            'Cinnamon': ['Laterite', 'Sandy'], 'Vanilla': ['Laterite', 'Loamy'],
            'Bitter gourd': ['Loamy', 'Sandy', 'Alluvial'], 'Snake gourd': ['Loamy', 'Sandy', 'Alluvial'],
            'Ash gourd': ['Loamy', 'Sandy', 'Alluvial'], 'Brinjal': ['Loamy', 'Alluvial', 'Clayey'],
            'Okra': ['Loamy', 'Alluvial'], 'Tomato': ['Loamy', 'Alluvial', 'Red'],
            'Chilli': ['Loamy', 'Alluvial', 'Red'], 'Cucumber': ['Loamy', 'Sandy', 'Alluvial']
        }
        random.seed(42)
        crop_ideals = {crop: {'N': random.randint(20, 120), 'P': random.randint(20, 100), 'K': random.randint(20, 120), 
                             'temp': random.uniform(20.0, 35.0), 'humidity': random.uniform(50.0, 95.0), 
                             'ph': random.uniform(5.0, 7.5), 'rainfall': random.uniform(50.0, 300.0)} 
                      for crop in KERALA_CROPS}
        
        data = []
        for _ in range(n_samples):
            crop = random.choice(KERALA_CROPS)
            ideal = crop_ideals[crop]
            soil_type = random.choice(CROP_SOIL_MAP.get(crop, KERALA_SOIL_TYPES)) if random.random() < 0.9 else random.choice(KERALA_SOIL_TYPES)
            district = random.choice(KERALA_DISTRICTS)
            n, p, k = ideal['N'] + random.uniform(-15, 15), ideal['P'] + random.uniform(-10, 10), ideal['K'] + random.uniform(-15, 15)
            temp, hum, ph, rain = ideal['temp'] + random.uniform(-3, 3), ideal['humidity'] + random.uniform(-10, 10), ideal['ph'] + random.uniform(-0.5, 0.5), ideal['rainfall'] + random.uniform(-30, 30)
            data.append([district, crop, soil_type, n, p, k, ph, temp, hum, rain, crop])

        cols = ['district', 'crop', 'soil_type', 'N', 'P', 'K', 'ph', 'temperature', 'humidity', 'rainfall', 'label']
        pd.DataFrame(data, columns=cols).to_csv('ml_training/datasets/crop_recommendation.csv', index=False)
        print("Generated crop_recommendation.csv (11 columns)")

    def generate_yield_sub(n_samples=3000):
        data = []
        for _ in range(n_samples):
            crop = random.choice(KERALA_CROPS)
            N, P, K = random.randint(40, 120), random.randint(20, 100), random.randint(20, 100)
            temp, rain = random.uniform(20, 35), random.uniform(50, 300)
            base = {'Rice': 4.0, 'Rubber': 0.8, 'Banana': 15.0, 'Tapioca': 12.0, 'Black Pepper': 0.5, 'Pineapple': 20.0, 'Tomato': 10.0, 'Cucumber': 10.0}
            yield_val = base.get(crop, 1.5) * random.uniform(0.8, 1.2) + (N+P+K)/1000
            data.append([crop, N, P, K, temp, rain, max(0.1, yield_val)])
        pd.DataFrame(data, columns=['Crop', 'N', 'P', 'K', 'Temperature', 'Rainfall', 'Yield_Tons_Acre']).to_csv('ml_training/datasets/yield_data.csv', index=False)
        print("Generated yield_data.csv")

    def generate_market_sub(years=5):
        prices = {'Rice': 30000, 'Rubber': 160000, 'Black Pepper': 500000, 'Cardamom': 1200000, 'Coffee': 180000, 'Tea': 140000, 'Cocoa': 220000, 'Arecanut': 450000, 'Banana': 25000, 'Mango': 60000, 'Papaya': 20000, 'Pineapple': 30000, 'Jackfruit': 15000, 'Guava': 40000, 'Sapota': 35000, 'Custard Apple': 50000, 'Tapioca': 12000, 'Maize': 20000, 'Cowpea': 45000, 'Green gram': 75000, 'Black gram': 70000, 'Ginger': 80000, 'Turmeric': 90000, 'Clove': 800000, 'Nutmeg': 500000, 'Cinnamon': 400000, 'Vanilla': 15000000, 'Bitter gourd': 35000, 'Snake gourd': 25000, 'Ash gourd': 20000, 'Brinjal': 30000, 'Okra': 35000, 'Tomato': 25000, 'Chilli': 80000, 'Cucumber': 20000, 'Coconut': 40000}
        data = []
        cur_yr = datetime.now().year
        for crop in KERALA_CROPS:
            base = prices.get(crop, 30000)
            for year in range(cur_yr - years, cur_yr + 1):
                for month in range(1, 13):
                    mod, demand = 1.0, 'Medium'
                    if crop == 'Banana' and month == 8: mod, demand = 1.4, 'High'
                    if crop == 'Ginger' and month in [12, 1]: mod, demand = 1.2, 'High'
                    if crop == 'Mango' and month in [4, 5]: mod, demand = 0.7, 'High'
                    data.append([crop, month, year, round(base * mod * random.uniform(0.9, 1.1), 2), demand])
        pd.DataFrame(data, columns=['Crop', 'Month', 'Year', 'Price_Per_Ton', 'Demand_Level']).to_csv('ml_training/datasets/market_history.csv', index=False)
        print("Generated market_history.csv")

    def generate_seed_sub():
        seed_info = {'Rice': [20, 'kg', 45], 'Rubber': [150, 'saplings', 120], 'Black Pepper': [800, 'cuttings', 25], 'Cardamom': [400, 'suckers', 60], 'Coffee': [1000, 'saplings', 15], 'Tea': [4000, 'clones', 10], 'Cocoa': [200, 'saplings', 40], 'Arecanut': [500, 'seedlings', 50], 'Banana': [1000, 'suckers', 20], 'Mango': [40, 'grafts', 150], 'Papaya': [1000, 'seeds', 5], 'Pineapple': [15000, 'suckers', 8], 'Jackfruit': [30, 'saplings', 100], 'Guava': [100, 'saplings', 80], 'Sapota': [100, 'saplings', 120], 'Custard Apple': [100, 'saplings', 100], 'Tapioca': [4000, 'cuttings', 2], 'Maize': [8, 'kg', 130], 'Cowpea': [10, 'kg', 100], 'Green gram': [8, 'kg', 120], 'Black gram': [10, 'kg', 110], 'Ginger': [600, 'kg', 60], 'Turmeric': [600, 'kg', 50], 'Clove': [100, 'saplings', 150], 'Nutmeg': [100, 'saplings', 200], 'Cinnamon': [1000, 'seedlings', 30], 'Vanilla': [1000, 'cuttings', 100], 'Bitter gourd': [2, 'kg', 2500], 'Snake gourd': [2, 'kg', 1800], 'Ash gourd': [2, 'kg', 1500], 'Brinjal': [0.2, 'kg', 12000], 'Okra': [4, 'kg', 2000], 'Tomato': [0.1, 'kg', 45000], 'Chilli': [0.5, 'kg', 8000], 'Cucumber': [1, 'kg', 5000], 'Coconut': [70, 'seedlings', 100]}
        pd.DataFrame([[c] + i for c, i in seed_info.items()], columns=['Crop', 'Seed_Rate_Per_Acre', 'Seed_Unit', 'Seed_Price_Per_Unit']).to_csv('ml_training/datasets/seed_rates.csv', index=False)
        print("Generated seed_rates.csv")

    def generate_advice_sub():
        advice_data = [
            # Crop, Days_to_Harvest, Germination, Vegetative, Flowering, Harvesting
            ['Rice', 120, 'Keep soil moist but not flooded.', 'Maintain water level at 2-3 cm. Apply Urea if needed.', 'Critical water stage. Prevent draining.', 'Drain water 10 days before harvest.'],
            ['Coconut', 365, 'Plant in a deep pit with proper drainage.', 'Water regularly during dry spells. Apply organic manure.', 'Ensure adequate moisture to prevent button shedding.', 'Harvest mature nuts every 45 days.'],
            ['Rubber', 365, 'Plant budded stumps in well-drained pits.', 'Maintain weed-free base. Apply NPK mixed fertilizers.', 'Protect from abnormal leaf fall diseases.', 'Commence tapping when girth reaches 50cm.'],
            ['Cardamom', 365, 'Provide constant shade and moisture.', 'Mulch the plant base. Apply split doses of fertilizers.', 'Ensure bee pollination. Maintain overhead shade.', 'Harvest capsules manually at 30-day intervals.'],
            ['Coffee', 365, 'Maintain nursery shade. Water cautiously to avoid root rot.', 'Prune plants for proper framework. Apply fertilizer.', 'Blossom showers are critical. Control berry borer.', 'Hand-pick ripe berries selectively.'],
            ['Tea', 365, 'Raise cuttings in shaded nursery bags.', 'Pruning and tipping is essential for canopy formation.', 'Maintain plucking tables. Remove flowering shoots.', 'Pluck two leaves and a bud regularly.'],
            ['Cashew', 365, 'Plant grafts in pits filled with topsoil.', 'Prune low-hanging branches. Monitor for tea mosquito bug.', 'Spray against pests during flushing and flowering.', 'Collect fallen mature nuts and sun dry.'],
            ['Banana', 300, 'Ensure sucker has good root system in well-manured pit.', 'Apply Potassium-rich fertilizer. Remove weeds and desucker.', 'Propping may be required for support.', 'Harvest when fingers satisfy angularity and turn light green.'],
            ['Sugarcane', 365, 'Plant setts in furrows and maintain optimal moisture.', 'Earthing up is required to prevent lodging. High nitrogen need.', 'Withhold massive irrigation to aid sugar accumulation.', 'Harvest close to the ground when stalks turn yellowish.'],
            ['Arecanut', 365, 'Plant seedlings under shade with continuous watering.', 'Apply organic compost. Control spindle bug.', 'Protect immature nuts from fungal diseases (Koleroga).', 'Harvest ripe bunches possessing yellow-orange color.'],
            ['Black Pepper', 365, 'Tie growing vines to the standard trees. Provide shade.', 'Apply cow dung and NPK. Control quick wilt disease.', 'Ensure good drainage during monsoons.', 'Pluck spikes when one or two berries turn red.'],
            ['Cocoa', 365, 'Plant under shade trees (like Arecanut or Coconut).', 'Formative pruning is crucial for canopy structure.', 'Ensure adequate moisture. Monitor for pod rot.', 'Harvest fully mature yellow/orange pods carefully.'],
            ['Tapioca', 300, 'Plant cuttings vertically or slanted in mounds.', 'Weed regularly. Apply potassium to support tuber growth.', 'Ensure adequate drainage. Protect against mosaic virus.', 'Harvest by uprooting the plant when basal leaves yellow.'],
            ['Ginger', 240, 'Plant rhizome bits with buds facing up. Mulch heavily.', 'Keep soil consistently moist but never waterlogged.', 'Apply organic manure. Weed carefully around shoots.', 'Harvest when leaves turn yellow and start drying.'],
            ['Turmeric', 270, 'Plant healthy rhizomes in raised beds. Apply thick mulch.', 'Control shoot borer pest. Apply fertilizer split doses.', 'Keep field weed-free to avoid nutrient competition.', 'Harvest when the above-ground stem dries up completely.'],
            ['Clove', 365, 'Protect seedlings from direct sunlight.', 'Apply slow-release fertilizers. Deep watering during dry seasons.', 'Monitor buds carefully. Prevent fungal diseases on foliage.', 'Hand-pick unopened flower buds when they turn pinkish.'],
            ['Nutmeg', 365, 'Maintain strict shade for young grafts.', 'Pruning helps establish a strong framework.', 'Ensure appropriate ratio of male to female trees if seeds used.', 'Harvest fruits when they split open revealing the red mace.'],
            ['Cinnamon', 365, 'Plant seedlings in well-drained pits.', 'Coppice trees at 2 years to encourage multiple shoots.', 'Peel bark when the sap flows freely (after rains).', 'Harvest straight shoots and peel the bark for drying.'],
            ['Vanilla', 365, 'Plant cuttings and train vines on support trees.', 'Loop vines to encourage flowering. Provide 50% shade.', 'Hand-pollination is strictly required early morning.', 'Harvest beans when the tips begin to turn yellow.'],
            ['Mango', 120, 'Protect young grafts from direct sun and wind.', 'Prune center to allow light. Apply fertilizer post-harvest.', 'Withhold water to induce flowering. Protect against hoppers.', 'Pluck fruits with stalks intact when fully mature.'],
            ['Papaya', 270, 'Plant in well-drained soil on mounds.', 'High potassium requirement. Do not let water stagnate.', 'Remove virus-affected plants immediately.', 'Pluck fruits when a yellow streak appears on the skin.'],
            ['Pineapple', 450, 'Plant robust suckers. Apply basal dose of fertilizer.', 'Keep beds weed-free. Maintain soil moisture.', 'Apply growth regulators to synchronize flowering if needed.', 'Harvest when 25% of the fruit turns yellow-orange.']
        ]
        
        # Add a default fallback crop advice
        advice_data.append(['Default', 120, 'Ensure proper watering and seed placement.', 'Monitor for pests and weeds. Apply fertilizer.', 'Ensure adequate irrigation during flowering.', 'Check maturity signs before harvest.'])

        df_advice = pd.DataFrame(advice_data, columns=['Crop', 'Days_to_Harvest', 'Germination', 'Vegetative', 'Flowering', 'Harvesting'])
        df_advice.to_csv('ml_training/datasets/crop_advice.csv', index=False)
        print("Generated crop_advice.csv")

    generate_rec_sub()
    generate_yield_sub()
    generate_market_sub()
    generate_seed_sub()
    generate_advice_sub()

# ==========================================
# 2. MODEL TRAINING (from train_*.py)
# ==========================================
def train_crop_model():
    os.makedirs('ml_models', exist_ok=True)
    df = pd.read_csv('ml_training/datasets/crop_recommendation.csv')
    if 'pH' in df.columns: df.rename(columns={'pH': 'ph'}, inplace=True)
    DISTRICT_SOIL_PRIORITY = {
        'Thiruvananthapuram': {'laterite': ['coconut', 'rubber'], 'sandy': ['coconut'], 'loamy': ['coconut', 'banana'], 'alluvial': ['banana']},
        'Kollam': {'laterite': ['cashew', 'rubber'], 'sandy': ['coconut']},
        'Pathanamthitta': {'laterite': ['rubber', 'pepper', 'black pepper'], 'forest': ['pepper', 'black pepper'], 'alluvial': ['banana']},
        'Alappuzha': {'alluvial': ['rice'], 'clayey': ['rice'], 'sandy': ['coconut']},
        'Kottayam': {'laterite': ['rubber', 'pepper', 'black pepper'], 'sandy': ['coconut'], 'forest': ['pepper', 'cardamom', 'clove', 'nutmeg']},
        'Idukki': {'forest': ['tea', 'cardamom', 'pepper', 'coffee', 'black pepper']},
        'Ernakulam': {'laterite': ['coconut'], 'sandy': ['coconut'], 'alluvial': ['banana'], 'loamy': ['tomato', 'bitter gourd']},
        'Thrissur': {'alluvial': ['rice', 'banana'], 'clayey': ['rice'], 'laterite': ['coconut']},
        'Palakkad': {'alluvial': ['rice', 'sugarcane', 'tomato'], 'clayey': ['rice'], 'loamy': ['sugarcane', 'tomato', 'bitter gourd']},
        'Malappuram': {'laterite': ['coconut', 'arecanut'], 'sandy': ['coconut'], 'alluvial': ['banana']},
        'Kozhikode': {'laterite': ['coconut', 'pepper', 'black pepper'], 'sandy': ['coconut'], 'alluvial': ['banana'], 'forest': ['pepper', 'black pepper']},
        'Wayanad': {'forest': ['tea', 'coffee', 'pepper', 'black pepper'], 'alluvial': ['rice'], 'clayey': ['rice']},
        'Kannur': {'laterite': ['coconut', 'cashew', 'pepper', 'black pepper'], 'sandy': ['coconut'], 'forest': ['pepper', 'black pepper']},
        'Kasaragod': {'laterite': ['coconut', 'arecanut', 'pepper', 'black pepper'], 'sandy': ['coconut'], 'forest': ['pepper', 'black pepper']}
    }
    def get_weight(row):
        c, d, s = str(row.get('label', '')).lower().strip(), str(row.get('district', '')).strip(), str(row.get('soil_type', '')).lower().strip()
        if d in DISTRICT_SOIL_PRIORITY and s in DISTRICT_SOIL_PRIORITY[d] and c in DISTRICT_SOIL_PRIORITY[d][s]: return 50.0
        if c == 'tea' and d in ['Idukki', 'Wayanad'] and s in ['forest', 'loamy']: return 50.0
        return 1.0
    df['weight'] = df.apply(get_weight, axis=1)
    for col, name in [(LabelEncoder(), 'soil_type_encoded'), (LabelEncoder(), 'district_encoded')]:
        df[name] = col.fit_transform(df[name.replace('_encoded', '')])
        joblib.dump(col, f'ml_models/{name.replace("_encoded", "")}_encoder.pkl')
    num_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    for col in num_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
    target = 'label' if 'label' in df.columns else 'crop'
    df = df.dropna(subset=num_cols + [target, 'district', 'soil_type'])
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'soil_type_encoded', 'district_encoded']]
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, df[target], df['weight'], test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    model.fit(X_train, y_train, sample_weight=w_train)
    joblib.dump(model, 'ml_models/crop_model.pkl')
    print(f"Crop model trained data accuracy: {model.score(X_train, y_train):.4f} | test data accuracy: {model.score(X_test, y_test):.4f}")

def train_yield_model():
    os.makedirs('ml_models', exist_ok=True)
    df = pd.read_csv('ml_training/datasets/yield_data.csv')
    le = LabelEncoder()
    df['Crop'] = le.fit_transform(df['Crop'])
    joblib.dump(le, 'ml_models/yield_label_encoder.pkl')
    X = df[['Crop', 'N', 'P', 'K', 'Temperature', 'Rainfall']]
    X_train, X_test, y_train, y_test = train_test_split(X, df['Yield_Tons_Acre'], test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'ml_models/yield_model.pkl')
    print(f"Yield model trained with R2 score: {model.score(X_test, y_test):.2f}")

# ==========================================
# 3. UTILITY SCRIPTS (Refine, Inyect, Clean, etc.)
# ==========================================
def refine_and_inject():
    # Merge Refine & Inject into one flow
    df = pd.read_csv("ml_training/datasets/crop_recommendation.csv")
    df.columns = df.columns.str.strip()
    if 'pH' in df.columns: df.rename(columns={'pH': 'ph'}, inplace=True)
    
    rules = [
        # Existing rules
        ('Thiruvananthapuram', 'laterite', 'coconut', 300), ('Thiruvananthapuram', 'sandy', 'coconut', 300), 
        ('Thiruvananthapuram', 'laterite', 'rubber', 150), ('Thiruvananthapuram', 'alluvial', 'banana', 150),
        ('Kollam', 'laterite', 'cashew', 300), ('Kollam', 'laterite', 'rubber', 300),
        ('Pathanamthitta', 'laterite', 'rubber', 300), ('Pathanamthitta', 'laterite', 'Black Pepper', 300),
        ('Alappuzha', 'alluvial', 'rice', 300), ('Alappuzha', 'clayey', 'rice', 300), ('Alappuzha', 'sandy', 'coconut', 150),
        ('Kottayam', 'laterite', 'rubber', 300), ('Kottayam', 'forest', 'Black Pepper', 300),
        ('Idukki', 'forest', 'cardamom', 300), ('Idukki', 'forest', 'Black Pepper', 300), ('Idukki', 'forest', 'tea', 300),
        ('Ernakulam', 'laterite', 'coconut', 300), ('Ernakulam', 'sandy', 'coconut', 300), ('Ernakulam', 'alluvial', 'banana', 150),
        ('Thrissur', 'alluvial', 'rice', 300), ('Thrissur', 'clayey', 'rice', 300), ('Thrissur', 'laterite', 'coconut', 150),
        ('Palakkad', 'alluvial', 'rice', 300), ('Palakkad', 'clayey', 'rice', 300), ('Palakkad', 'loamy', 'sugarcane', 300),
        ('Malappuram', 'laterite', 'coconut', 300), ('Malappuram', 'sandy', 'coconut', 300), ('Malappuram', 'laterite', 'arecanut', 300),
        ('Kozhikode', 'laterite', 'coconut', 300), ('Kozhikode', 'sandy', 'coconut', 300), ('Kozhikode', 'laterite', 'Black Pepper', 300),
        ('Wayanad', 'forest', 'coffee', 300), ('Wayanad', 'forest', 'Black Pepper', 300), ('Wayanad', 'forest', 'tea', 300),
        ('Kannur', 'laterite', 'coconut', 300), ('Kannur', 'laterite', 'cashew', 300), ('Kannur', 'laterite', 'Black Pepper', 300),
        ('Kasaragod', 'laterite', 'coconut', 300), ('Kasaragod', 'laterite', 'arecanut', 300), ('Kasaragod', 'laterite', 'Black Pepper', 300),
        
        # New crops rules
        ('Idukki', 'forest', 'Cocoa', 200), ('Wayanad', 'forest', 'Ginger', 200),
        ('Palakkad', 'alluvial', 'Tapioca', 200), ('Kozhikode', 'laterite', 'Turmeric', 200),
        ('Kannur', 'laterite', 'Clove', 200), ('Wayanad', 'forest', 'Nutmeg', 200),
        ('Kottayam', 'laterite', 'Cinnamon', 200), ('Idukki', 'forest', 'Vanilla', 200),
        ('Palakkad', 'alluvial', 'Mango', 200), ('Ernakulam', 'alluvial', 'Papaya', 200),
        ('Thiruvananthapuram', 'alluvial', 'Pineapple', 200)
    ]
    
    crop_profiles = {
        'coconut': (40, 20, 40, 27, 80, 200, 6.0), 'cashew': (30, 15, 25, 28, 70, 180, 5.5), 
        'rubber': (50, 20, 30, 26, 85, 250, 5.0), 'rice': (80, 40, 40, 25, 80, 220, 6.0), 
        'cardamom': (40, 30, 40, 22, 85, 300, 5.5), 'coffee': (60, 30, 40, 24, 80, 200, 5.5), 
        'Black Pepper': (50, 40, 40, 25, 80, 250, 5.5), 'banana': (80, 40, 80, 28, 75, 180, 6.5), 
        'tomato': (60, 40, 40, 25, 70, 150, 6.5), 'bitter gourd': (50, 40, 40, 26, 75, 160, 6.0), 
        'sugarcane': (100, 50, 60, 30, 75, 250, 6.5), 'arecanut': (50, 20, 40, 27, 85, 280, 5.5),
        'tea': (80, 30, 50, 22, 80, 2500, 5.0), 'Cocoa': (40, 40, 60, 27, 85, 200, 6.0),
        'Ginger': (80, 60, 100, 28, 80, 200, 6.0), 'Tapioca': (40, 30, 100, 30, 70, 150, 5.5),
        'Turmeric': (80, 60, 100, 28, 80, 200, 6.0), 'Clove': (40, 40, 40, 26, 85, 250, 5.5),
        'Nutmeg': (40, 40, 40, 26, 85, 250, 5.5), 'Cinnamon': (40, 40, 40, 26, 85, 250, 5.5),
        'Vanilla': (30, 20, 30, 28, 85, 250, 6.0), 'Mango': (80, 40, 80, 32, 60, 100, 6.5),
        'Papaya': (100, 80, 120, 30, 70, 150, 6.5), 'Pineapple': (50, 40, 100, 25, 80, 150, 5.0)
    }
    
    syn_rows = []
    for d, s, c, count in rules:
        if c not in crop_profiles: continue # skip if profile missing
        base = crop_profiles[c]
        for _ in range(count):
            syn_rows.append({'district': d, 'soil_type': s, 'crop': c, 'N': max(10, base[0] + random.randint(-15, 15)), 'P': max(10, base[1] + random.randint(-10, 10)), 'K': max(10, base[2] + random.randint(-15, 15)), 'temperature': base[3] + random.uniform(-3, 3), 'humidity': base[4] + random.uniform(-10, 10), 'rainfall': base[5] + random.uniform(-50, 50), 'ph': base[6] + random.uniform(-0.5, 0.5), 'label': c})
    
    # Pre-calculate lowercase major crops for this district to avoid deleting valid original crops
    dist_rules_major = {}
    for d, s, c, count in rules:
        d_lower = d.lower()
        if d_lower not in dist_rules_major: dist_rules_major[d_lower] = set()
        dist_rules_major[d_lower].add(c.lower())

    def is_valid_original(row):
        d, c = str(row.get('district', '')).lower().strip(), str(row.get('crop', row.get('label', ''))).lower().strip()
        # If the district is one our target "rule districts", only allow the major crops to keep the model focused
        if d in dist_rules_major:
            if c not in dist_rules_major[d]: return False
        return True
    
    df_clean = df[df.apply(is_valid_original, axis=1)]
    pd.concat([df_clean, pd.DataFrame(syn_rows)], ignore_index=True).to_csv("ml_training/datasets/crop_recommendation.csv", index=False)
    print(f"Dataset refined with {len(syn_rows)} expert facts for major Kerala crops.")

def sanitize_and_fix():
    csv_path = 'ml_training/datasets/crop_recommendation.csv'
    df = pd.read_csv(csv_path)
    def clean_text(text):
        if pd.isna(text): return text
        text = str(text)
        text = re.sub(r'(?i)\(major\)|\(minor\)|major|minor', '', text)
        return text.replace('✅', '').replace('⚠️', '').strip()
    for col in ['district', 'soil_type']:
        if col in df.columns: df[col] = df[col].apply(clean_text)
    for col in ['crop', 'label']:
        if col in df.columns: df[col] = df[col].apply(clean_text).str.title()
        
    if 'crop' in df.columns: df['crop'] = df['crop'].fillna(df['label'])
    else: df['crop'] = df['label']
    cols = ['district', 'crop', 'soil_type', 'N', 'P', 'K', 'ph', 'temperature', 'humidity', 'rainfall', 'label']
    df[cols].to_csv(csv_path, index=False)
    
    m_df = pd.read_csv('ml_training/datasets/market_history.csv')
    if 'Price' in m_df.columns:
        m_df['Price_Per_Ton'] = m_df['Price_Per_Ton'].fillna(m_df['Price'])
        m_df['Demand_Level'] = m_df['Demand_Level'].fillna('High')
        m_df.drop('Price', axis=1, inplace=True)
    m_df.to_csv('ml_training/datasets/market_history.csv', index=False)
    print("Dataset sanitized and column headers patched.")

def rebuild_tea_precise():
    csv_path = 'ml_training/datasets/crop_recommendation.csv'
    df = pd.read_csv(csv_path)
    df = df[df['label'] != 'tea']
    distributions = [('Idukki', 'forest', 400), ('Idukki', 'loamy', 150), ('Wayanad', 'forest', 150), ('Wayanad', 'loamy', 50)]
    tea_rows = []
    for d, s, count in distributions:
        for _ in range(count):
            tea_rows.append({'district': d, 'crop': 'tea', 'soil_type': s, 'N': random.randint(60, 120), 'P': random.randint(20, 40), 'K': random.randint(40, 80), 'ph': round(random.uniform(4.0, 6.0), 2), 'temperature': round(random.uniform(10.0, 38.0), 2), 'humidity': round(random.uniform(45.0, 95.0), 2), 'rainfall': round(random.uniform(50.0, 400.0), 2), 'label': 'tea'})
    pd.concat([df, pd.DataFrame(tea_rows)], ignore_index=True).dropna(subset=['N', 'P', 'K', 'ph', 'temperature', 'humidity', 'rainfall', 'label', 'district', 'soil_type']).to_csv(csv_path, index=False)
    print("Tea data rebuilt precisely.")

def add_missing_crops():
    # Logic from add_missing_crops.py
    CROP_SOIL_MAP = {'Black Pepper': ['Laterite', 'Red', 'Loamy'], 'Cocoa': ['Laterite', 'Loamy'], 'Mango': ['Laterite', 'Alluvial', 'Red'], 'Papaya': ['Loamy', 'Alluvial'], 'Pineapple': ['Laterite', 'Sandy', 'Loamy'], 'Tapioca': ['Laterite', 'Sandy', 'Red'], 'Ginger': ['Loamy', 'Laterite'], 'Turmeric': ['Loamy', 'Laterite', 'Alluvial'], 'Clove': ['Laterite', 'Loamy'], 'Nutmeg': ['Laterite', 'Loamy'], 'Cinnamon': ['Laterite', 'Sandy'], 'Vanilla': ['Laterite', 'Loamy']}
    CROP_IDEALS = {'Black Pepper': {'N': 60, 'P': 40, 'K': 100, 'temp': 28, 'humidity': 80, 'ph': 5.5, 'rainfall': 250}, 'Cocoa': {'N': 40, 'P': 40, 'K': 60, 'temp': 27, 'humidity': 85, 'ph': 6.0, 'rainfall': 200}, 'Mango': {'N': 80, 'P': 40, 'K': 80, 'temp': 32, 'humidity': 60, 'ph': 6.5, 'rainfall': 100}, 'Papaya': {'N': 100, 'P': 80, 'K': 120, 'temp': 30, 'humidity': 70, 'ph': 6.5, 'rainfall': 150}, 'Pineapple': {'N': 50, 'P': 40, 'K': 100, 'temp': 25, 'humidity': 80, 'ph': 5.0, 'rainfall': 150}, 'Tapioca': {'N': 40, 'P': 30, 'K': 100, 'temp': 30, 'humidity': 70, 'ph': 5.5, 'rainfall': 150}, 'Ginger': {'N': 80, 'P': 60, 'K': 100, 'temp': 28, 'humidity': 80, 'ph': 6.0, 'rainfall': 200}, 'Turmeric': {'N': 80, 'P': 60, 'K': 100, 'temp': 28, 'humidity': 80, 'ph': 6.0, 'rainfall': 200}, 'Clove': {'N': 40, 'P': 40, 'K': 40, 'temp': 26, 'humidity': 85, 'ph': 5.5, 'rainfall': 250}, 'Nutmeg': {'N': 40, 'P': 40, 'K': 40, 'temp': 26, 'humidity': 85, 'ph': 5.5, 'rainfall': 250}, 'Cinnamon': {'N': 40, 'P': 40, 'K': 40, 'temp': 26, 'humidity': 85, 'ph': 5.5, 'rainfall': 250}, 'Vanilla': {'N': 30, 'P': 20, 'K': 30, 'temp': 28, 'humidity': 85, 'ph': 6.0, 'rainfall': 250}}
    districts = ['Thiruvananthapuram', 'Kollam', 'Pathanamthitta', 'Alappuzha', 'Kottayam', 'Idukki', 'Ernakulam', 'Thrissur', 'Palakkad', 'Malappuram', 'Kozhikode', 'Wayanad', 'Kannur', 'Kasaragod']
    
    # Recommendation append
    df = pd.read_csv('ml_training/datasets/crop_recommendation.csv')
    new_data = []
    for crop, ideal in CROP_IDEALS.items():
        for _ in range(250):
            new_data.append([random.choice(districts), crop, random.choice(CROP_SOIL_MAP[crop]), ideal['N'] + random.uniform(-10, 10), ideal['P'] + random.uniform(-10, 10), ideal['K'] + random.uniform(-10, 10), ideal['ph'] + random.uniform(-0.5, 0.5), ideal['temp'] + random.uniform(-3, 3), ideal['humidity'] + random.uniform(-10, 10), ideal['rainfall'] + random.uniform(-30, 30), crop])
    pd.concat([df, pd.DataFrame(new_data, columns=df.columns)], ignore_index=True).to_csv('ml_training/datasets/crop_recommendation.csv', index=False)
    
    # Yield append
    df_y = pd.read_csv('ml_training/datasets/yield_data.csv')
    base_y = {'Tapioca': 12.0, 'Papaya': 15.0, 'Mango': 5.0, 'Pineapple': 20.0, 'Black Pepper': 0.5, 'Vanilla': 0.3, 'Clove': 0.3, 'Nutmeg': 0.3, 'Cinnamon': 0.3}
    new_y = []
    for crop, ideal in CROP_IDEALS.items():
        for _ in range(200):
            n, p, k = ideal['N'] + random.uniform(-10, 10), ideal['P'] + random.uniform(-10, 10), ideal['K'] + random.uniform(-10, 10)
            val = base_y.get(crop, 1.0) * random.uniform(0.8, 1.2) + (n+p+k)/2000
            new_y.append([crop, n, p, k, ideal['temp'] + random.uniform(-3, 3), ideal['rainfall'] + random.uniform(-30, 30), max(0.1, val)])
    pd.concat([df_y, pd.DataFrame(new_y, columns=df_y.columns)], ignore_index=True).to_csv('ml_training/datasets/yield_data.csv', index=False)
    print("Missing crops data appended to recommendation and yield datasets.")

def populate_market():
    crops_data = [("Sugarcane", 3000, "High"), ("Black Pepper", 550000, "High"), ("Arecanut", 450000, "Medium"), ("Ash gourd", 12000, "Low"), ("Tomato", 18000, "Medium"), ("Cardamom", 900000, "High"), ("Cashew", 700000, "High")]
    with open(r'ml_training/datasets/market_history.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        for crop, price, demand in crops_data:
            for month in range(1, 13): writer.writerow([crop, month, 2026, price, demand])
    print(f"Market history populated for {len(crops_data)} crops.")

# ==========================================
# 4. MAIN PIPELINE EXECUTION
# ==========================================
def run_full_pipeline():
    print("--- STARTING FULL ML PIPELINE ---")
    generate_base_datasets()
    add_missing_crops()
    rebuild_tea_precise()
    refine_and_inject()
    sanitize_and_fix()
    populate_market()
    train_crop_model()
    train_yield_model()
    print("--- PIPELINE COMPLETED SUCCESSFULLY ---")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'train':
            train_crop_model()
            train_yield_model()
        elif cmd == 'generate':
            generate_base_datasets()
        elif cmd == 'full':
            run_full_pipeline()
        else:
            print("Unknown command. Use 'train', 'generate', or 'full'.")
    else:
        run_full_pipeline()
