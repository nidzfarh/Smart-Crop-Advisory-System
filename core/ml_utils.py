import joblib
import pandas as pd
# Django Hot-Reload Trigger: Forcing Cache wipe to pull newly added Sugarcane/Pepper records
import os
from django.conf import settings

# Paths to models
MODEL_DIR = os.path.join(settings.BASE_DIR, 'ml_models')
CROP_MODEL_PATH = os.path.join(MODEL_DIR, 'crop_model.pkl')
YIELD_MODEL_PATH = os.path.join(MODEL_DIR, 'yield_model.pkl')
YIELD_ENCODER_PATH = os.path.join(MODEL_DIR, 'yield_label_encoder.pkl')
YIELD_ENCODER_PATH = os.path.join(MODEL_DIR, 'yield_label_encoder.pkl')
SOIL_ENCODER_PATH = os.path.join(MODEL_DIR, 'soil_encoder.pkl')
DISTRICT_ENCODER_PATH = os.path.join(MODEL_DIR, 'district_encoder.pkl')

MARKET_HISTORY_PATH = os.path.join(settings.BASE_DIR, 'ml_training', 'datasets', 'market_history.csv')
SEED_RATES_PATH = os.path.join(settings.BASE_DIR, 'ml_training', 'datasets', 'seed_rates.csv')

# Load models and data
try:
    crop_model = joblib.load(CROP_MODEL_PATH)
    yield_model = joblib.load(YIELD_MODEL_PATH)
    yield_encoder = joblib.load(YIELD_ENCODER_PATH)
    soil_encoder = joblib.load(SOIL_ENCODER_PATH)
    district_encoder = joblib.load(DISTRICT_ENCODER_PATH)
    
    market_history_df = pd.read_csv(MARKET_HISTORY_PATH)
    seed_rates_df = pd.read_csv(SEED_RATES_PATH)
except Exception as e:
    print(f"Error loading models or datasets: {e}")
    crop_model = None
    yield_model = None
    yield_encoder = None
    soil_encoder = None
    district_encoder = None
    market_history_df = None
    seed_rates_df = None

HILL_DISTRICTS = ['Idukki', 'Wayanad', 'Pathanamthitta']
COASTAL_WETLANDS = ['Alappuzha']
HIGH_ALTITUDE = ['Idukki', 'Wayanad']

CROP_SOIL_MAP = {
    'rubber': ['laterite'],
    'coconut': ['laterite', 'sandy', 'loamy'],
    'arecanut': ['laterite', 'clayey', 'loamy'],
    'cashew': ['laterite'],
    'rice': ['alluvial', 'clayey'],
    'sugarcane': ['alluvial', 'loamy'],
    'pepper': ['forest', 'laterite'],
    'black pepper': ['forest', 'laterite'],
    'cardamom': ['forest'],
    'coffee': ['forest', 'loamy'],
    'tea': ['forest', 'laterite'],
    'banana': ['alluvial', 'loamy'],
    'tomato': ['loamy', 'alluvial'],
    'bitter gourd': ['loamy', 'alluvial'],
    'cocoa': ['laterite', 'loamy'],
    'mango': ['laterite', 'alluvial', 'red'],
    'papaya': ['loamy', 'alluvial'],
    'pineapple': ['laterite', 'sandy', 'loamy'],
    'tapioca': ['laterite', 'sandy', 'red'],
    'ginger': ['loamy', 'laterite'],
    'turmeric': ['loamy', 'laterite', 'alluvial'],
    'clove': ['laterite', 'loamy'],
    'nutmeg': ['laterite', 'loamy'],
    'cinnamon': ['laterite', 'sandy'],
    'vanilla': ['laterite', 'loamy']
}

DISTRICT_SOIL_PRIORITY = {
    'Thiruvananthapuram': {'laterite': ['coconut', 'rubber'], 'sandy': ['coconut'], 'loamy': ['coconut', 'banana'], 'alluvial': ['banana', 'papaya', 'sugarcane']},
    'Kollam': {'laterite': ['cashew', 'rubber'], 'sandy': ['coconut'], 'alluvial': ['rice']},
    'Pathanamthitta': {'laterite': ['rubber', 'pepper', 'black pepper'], 'forest': ['pepper', 'black pepper'], 'alluvial': ['banana', 'sugarcane']},
    'Alappuzha': {'alluvial': ['rice', 'sugarcane'], 'clayey': ['rice'], 'sandy': ['coconut']},
    'Kottayam': {'laterite': ['rubber', 'pepper', 'black pepper'], 'sandy': ['coconut'], 'forest': ['pepper', 'cardamom', 'clove', 'nutmeg'], 'alluvial': ['banana']},
    'Idukki': {'forest': ['tea', 'cardamom', 'pepper', 'coffee', 'black pepper']},
    'Ernakulam': {'laterite': ['coconut', 'mango'], 'sandy': ['coconut'], 'alluvial': ['banana', 'bitter gourd'], 'loamy': ['tomato', 'bitter gourd']},
    'Thrissur': {'alluvial': ['rice', 'banana', 'papaya'], 'clayey': ['rice'], 'laterite': ['coconut']},
    'Palakkad': {'alluvial': ['rice', 'sugarcane', 'tomato'], 'clayey': ['rice'], 'loamy': ['sugarcane', 'tomato', 'bitter gourd']},
    'Malappuram': {'laterite': ['coconut', 'arecanut'], 'sandy': ['coconut'], 'alluvial': ['banana', 'turmeric', 'tomato']},
    'Kozhikode': {'laterite': ['coconut', 'pepper', 'black pepper'], 'sandy': ['coconut'], 'alluvial': ['banana', 'mango'], 'forest': ['pepper', 'black pepper']},
    'Wayanad': {'forest': ['tea', 'coffee', 'pepper', 'black pepper'], 'alluvial': ['rice', 'ginger'], 'clayey': ['rice']},
    'Kannur': {'laterite': ['coconut', 'cashew', 'pepper', 'black pepper'], 'sandy': ['coconut'], 'forest': ['pepper', 'black pepper'], 'alluvial': ['rice', 'turmeric']},
    'Kasaragod': {'laterite': ['coconut', 'arecanut', 'pepper', 'black pepper'], 'sandy': ['coconut'], 'forest': ['pepper', 'black pepper'], 'alluvial': ['banana', 'arecanut']}
}

def is_valid_constraint(crop_name, district, soil_type):
    c = str(crop_name).lower().strip()
    d = str(district).title().strip()
    s = str(soil_type).lower().strip()
    
    if c in ['tea', 'coffee', 'cardamom'] and d not in HILL_DISTRICTS: return False
    if c in ['coconut', 'rice'] and d in HIGH_ALTITUDE: return False
    if c == 'rubber' and d in COASTAL_WETLANDS: return False
    
    allowed = CROP_SOIL_MAP.get(c, [])
    if allowed and s not in allowed: return False
    return True

def get_all_crops():
    if seed_rates_df is not None:
        return sorted([c.strip() for c in seed_rates_df['Crop'].unique() if isinstance(c, str)])
    return []

def predict_top_crops(n, p, k, temp, humidity, ph, rainfall, soil_type, district, top_n=3):
    if not crop_model or not soil_encoder or not district_encoder:
        return []
    
    try:
        if soil_type in soil_encoder.classes_:
            soil_encoded = soil_encoder.transform([soil_type])[0]
        else:
            soil_encoded = 0 # fallback
            
        if district in district_encoder.classes_:
            district_encoded = district_encoder.transform([district])[0]
        else:
            district_encoded = 0 # fallback
    except:
        soil_encoded = 0
        district_encoded = 0

    input_data = pd.DataFrame(
        [[n, p, k, temp, humidity, ph, rainfall, soil_encoded, district_encoded]],
        columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'soil_type_encoded', 'district_encoded']
    )
    probs = crop_model.predict_proba(input_data)[0]
    classes = crop_model.classes_
    district_title = str(district).title().strip()
    soil_lower = str(soil_type).lower().strip()
    
    district_map = DISTRICT_SOIL_PRIORITY.get(district_title, {})
    priority_crops = district_map.get(soil_lower, [])
    
    # Enforce strict constraints and apply expert bonuses
    for i, c_name in enumerate(classes):
        c_lower = str(c_name).lower().strip()
        
        if not is_valid_constraint(c_name, district, soil_type):
            probs[i] = 0.0
            continue
            
        if c_lower in priority_crops:
            probs[i] += 2.0  # Massive guaranteed priority boost over everything else
            
    # Get top classes that are valid
    class_indices = probs.argsort()[::-1]
    
    top_crops = []
    seen = set()
    for idx in class_indices:
        if probs[idx] > 0:
            normalized_crop = str(classes[idx]).title().strip()
            if normalized_crop not in seen:
                seen.add(normalized_crop)
                top_crops.append(normalized_crop)
            if len(top_crops) >= top_n:
                break
                
    # If constraints removed everything, fallback smoothly
    if not top_crops and len(classes) > 0:
        top_crops.append(str(classes[probs.argsort()[-1]]).title().strip())
        
    return top_crops

def predict_yield(crop, n, p, k, temp, rain):
    if not yield_model or not yield_encoder:
        return 0.0
    
    try:
        # Yield model was trained with Title Case labels (e.g., 'Rice', 'Tomato')
        # recommendation engine often uses lowercase. Normalize to .title()
        crop_norm = str(crop).strip().title()
        
        # Fallback if title case doesn't match exactly
        if crop_norm not in yield_encoder.classes_:
            crop_norm = str(crop).strip().lower()
            if crop_norm not in yield_encoder.classes_:
                # Use a specific but varied fallback if still missing
                return 2.0 
            
        crop_encoded = yield_encoder.transform([crop_norm])[0]
        input_data = pd.DataFrame(
            [[crop_encoded, n, p, k, temp, rain]],
            columns=['Crop', 'N', 'P', 'K', 'Temperature', 'Rainfall']
        )
        predicted_yield = yield_model.predict(input_data)[0]
        return max(0.2, float(predicted_yield))
    except Exception as e:
        print(f"Yield prediction error: {e}")
        return 0.0

def get_seasonal_market_estimate(crop, harvest_month):
    """
    Returns (avg_price, demand_level) for a crop in a specific month
    by averaging historical data across all years.
    """
    if market_history_df is None:
        return 0.0, 'Unknown'
    
    # Filter by crop (case-insensitive) and month
    mask = (market_history_df['Crop'].str.lower() == crop.strip().lower()) & \
           (market_history_df['Month'] == harvest_month)
    
    subset = market_history_df[mask]
    if subset.empty:
        return 0.0, 'Unknown'
    
    avg_price = subset['Price_Per_Ton'].mean()
    # Most frequent demand level
    demand = subset['Demand_Level'].mode()[0] if not subset['Demand_Level'].mode().empty else 'Medium'
    
    return round(avg_price, 2), demand

def get_seed_details(crop):
    """
    Returns (rate_per_acre, unit, price_per_unit, is_approx)
    """
    if seed_rates_df is None:
        return 20.0, 'kg', 50.0, True
    
    mask = seed_rates_df['Crop'].str.lower() == crop.strip().lower()
    row = seed_rates_df[mask]
    
    if not row.empty:
        return (
            float(row.iloc[0]['Seed_Rate_Per_Acre']),
            row.iloc[0]['Seed_Unit'],
            float(row.iloc[0]['Seed_Price_Per_Unit']),
            False
        )
    
    # Generic Fallback
    return 20.0, 'unit', 50.0, True

def get_seed_cost(crop):
    """Legacy compatibility: returns total seed cost per acre."""
    rate, _, price, _ = get_seed_details(crop)
    return rate * price

def get_advisory(crop):
    # Simple dictionary based advisory
    advisory = {
        'Rice': "Requires flooded fields. Monitor water levels.",
        'Maize': "Ensure good drainage. Watch for stem borer.",
        'Wheat': "Cold weather crop. Needs timely irrigation.",
        'Cotton': "Sensitive to pests (bollworm). Check leaves regularly.",
        'Sugarcane': "Long duration crop. High water requirement.",
    }
    return advisory.get(crop, "Ensure balanced nutrition and timely irrigation. Monitor for local pests.")
