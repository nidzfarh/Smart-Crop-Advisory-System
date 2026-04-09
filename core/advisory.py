import random
from datetime import date, timedelta
import pandas as pd
import os

ADVICE_CSV_PATH = 'ml_training/datasets/crop_advice.csv'
_ADVICE_CACHE = None

def get_advice_db():
    global _ADVICE_CACHE
    if _ADVICE_CACHE is not None:
        return _ADVICE_CACHE
        
    db = {}
    CROP_STAGE_ADVICE_CSV = 'ml_training/datasets/crop_stage_advice.csv'
    if os.path.exists(CROP_STAGE_ADVICE_CSV):
        try:
            df = pd.read_csv(CROP_STAGE_ADVICE_CSV)
            for _, row in df.iterrows():
                crop_name = str(row['crop']).lower().strip()
                if crop_name not in db:
                    db[crop_name] = []
                
                db[crop_name].append({
                    'stage': str(row['stage']).title(),
                    'start_day': int(row['start_day']),
                    'end_day': int(row['end_day']),
                    'description': str(row['description'])
                })
        except Exception as e:
            print(f"Error loading stage advice csv: {e}")
            
    _ADVICE_CACHE = db
    return db


# Typical growing days to harvest for each supported crop
CROP_GROWING_DAYS = {
    'Rice': 120, 'Maize': 90, 'Black gram': 70, 'Green gram': 75, 'Cowpea': 90,
    'Coconut': 365, 'Rubber': 365, 'Coffee': 365, 'Tea': 365, 'Cocoa': 365,
    'Arecanut': 365, 'Black Pepper': 365, 'Cardamom': 365, 'Ginger': 240,
    'Turmeric': 270, 'Clove': 365, 'Nutmeg': 365, 'Cinnamon': 365, 'Vanilla': 365,
    'Banana': 300, 'Mango': 120, 'Papaya': 270, 'Pineapple': 450, 'Jackfruit': 365,
    'Guava': 180, 'Sapota': 180, 'Custard Apple': 180, 'Tapioca': 300,
    'Bitter gourd': 90, 'Snake gourd': 90, 'Ash gourd': 120, 'Brinjal': 120,
    'Okra': 90, 'Tomato': 100, 'Chilli': 150, 'Cucumber': 60
}


def _rain_label(mm):
    if mm < 2:   return "dry conditions"
    if mm < 8:   return "light showers"
    if mm < 20:  return "moderate rainfall"
    return "heavy rainfall"


def _temp_label(t):
    if t < 22: return "cool temperatures"
    if t < 28: return "warm temperatures"
    if t < 33: return "hot conditions"
    return "very high heat"


def generate_crop_rationale(crop, n, p, k, ph, current_weather, harvest_weather=None):
    """
    Builds a 2–3 sentence human-readable explanation for recommending this crop,
    using soil inputs, current weather, and predicted harvest-time weather.
    """
    db = get_advice_db()
    c_lower = str(crop).lower().strip()
    
    if c_lower in db and len(db[c_lower]) > 0:
        days = max([stage['end_day'] for stage in db[c_lower]])
    else:
        days = CROP_GROWING_DAYS.get(crop, 120)
        
    harvest_date = date.today() + timedelta(days=days)
    rationale_parts = []

    # Soil suitability sentence
    soil_note = ""
    if n > 80:
        soil_note = f"high nitrogen content ({n} kg/ha)"
    elif n < 30:
        soil_note = f"low nitrogen soil ({n} kg/ha)"
    else:
        soil_note = f"balanced nitrogen level ({n} kg/ha)"

    ph_note = ""
    if 5.5 <= ph <= 7.0:
        ph_note = f"ideal pH ({ph})"
    elif ph < 5.5:
        ph_note = f"acidic soil (pH {ph})"
    else:
        ph_note = f"alkaline soil (pH {ph})"

    rationale_parts.append(
        f"Your soil has {soil_note} and {ph_note}, which suits {crop} cultivation."
    )

    # Current climate sentence
    if current_weather:
        ct = current_weather.get('temp', 27)
        cr = current_weather.get('rainfall', 150)
        rationale_parts.append(
            f"Current climate in your region shows {_temp_label(ct)} ({ct}°C) "
            f"and {_rain_label(cr)}, favourable for the planting stage of {crop}."
        )

    # Harvest forecast sentence
    if harvest_weather:
        ht = harvest_weather.get('temp', 27)
        hr = harvest_weather.get('rainfall', 150)
        src = harvest_weather.get('source', 'historical data')
        rationale_parts.append(
            f"At expected harvest (~{harvest_date.strftime('%b %Y')}, ~{days} days), "
            f"forecasted conditions (based on {src}) indicate {_temp_label(ht)} ({ht}°C) "
            f"with {_rain_label(hr)} — suitable for {crop} maturation and harvest."
        )
    else:
        rationale_parts.append(
            f"Estimated harvest around {harvest_date.strftime('%b %Y')} (~{days} days from now)."
        )

    return ' '.join(rationale_parts)


def get_crop_advice(crop_name, days):
    crop_name = crop_name.lower().strip()
    db = get_advice_db()
    
    if crop_name in db and len(db[crop_name]) > 0:
        stages = db[crop_name]
        for stage in stages:
            if stage['start_day'] <= days <= stage['end_day']:
                return stage['stage'], stage['description']
        
        if days > stages[-1]['end_day']:
            return stages[-1]['stage'], stages[-1]['description']
            
    # Generic fallback
    if days <= 15: return "Germination", "Start of journey. Ensure optimal soil moisture."
    elif days <= 45: return "Vegetative", "Apply balanced fertilizer and support growth."
    elif days <= 90: return "Flowering", "Critical period. Avoid overhead irrigation."
    else: return "Harvest Ready", "Check maturity criteria and begin picking."

def get_weather_alert(district):
    # Mock simulation
    alerts = [
        "Heavy rainfall expected next 2 days. Secure harvested crops.",
        "High humidity predicted. Watch out for fungal diseases.",
        "Dry spell approaching. Ensure irrigation system is ready.",
        "Clear skies. Good time for fertilizer application.",
        "Moderate winds. Assessment recommended for tall crops."
    ]
    # Deterministic pseudo-random based on district name length for consistency during session
    idx = len(district) % len(alerts)
    return alerts[idx]

def get_crop_timeline(crop_name, planting_date):
    """
    Generates a timeline of events based on planting date and crop type.
    """
    from datetime import timedelta
    crop = crop_name.lower().strip()
    db = get_advice_db()
    
    events = []
    if crop in db and len(db[crop]) > 0:
        for stage in db[crop]:
            events.append({
                'day': stage['start_day'],
                'title': stage['stage'],
                'desc': stage['description']
            })
    else:
        harvest_day = CROP_GROWING_DAYS.get(crop.title(), 120)
        events = [
            {'day': 0, 'title': 'Germination', 'desc': 'Ensure optimal soil moisture for germination.'},
            {'day': int(harvest_day * 0.35), 'title': 'Vegetative', 'desc': 'Apply balanced fertilizer. Support the plants.'},
            {'day': int(harvest_day * 0.65), 'title': 'Flowering / Fruiting', 'desc': 'Critical period for water. Avoid overhead irrigation.'},
            {'day': harvest_day, 'title': 'Harvest Ready', 'desc': 'Check maturity criteria and begin harvest.'}
        ]
    
    # Safety sort to ensure timeline is always chronological
    events = sorted(events, key=lambda x: x['day'])
    
    timeline = []
    today = date.today()
    
    for event in events:
        event_date = planting_date + timedelta(days=event['day'])
        status = 'Done' if event_date < today else ('Today' if event_date == today else 'Upcoming')
        
        timeline.append({
            'date': event_date,
            'days_offset': event['day'],
            'title': event['title'],
            'description': event['desc'],
            'status': status
        })
        
    return timeline


def get_crop_status(crop_name, days_elapsed, crop_id):
    # Fixed seed for consistent behavior for a single crop instance
    seed = (crop_id * 13) % 100
    crop = crop_name.lower().strip()
    
    vegetables = ['cucumber', 'chilli', 'brinjal', 'okra', 'ash gourd', 'snake gourd', 'bitter gourd', 'tapioca', 'tomato']
    pulses = ['black gram', 'green gram', 'cowpea', 'lentil', 'pigeon pea']
    spices = ['turmeric', 'ginger', 'black pepper', 'clove', 'cardamom', 'cinnamon', 'nutmeg', 'pepper']
    perennials = ['coffee', 'coconut', 'tea', 'rubber', 'mango', 'jackfruit', 'guava', 'papaya', 'custard apple', 'sapota', 'pineapple', 'arecanut', 'cocoa', 'vanilla', 'apple']
    
    status_title = "Optimal Growth"
    status_desc = f"Based on current data for your {crop_name}, soil moisture is within target range for this phase."
    actions = []
    
    if days_elapsed < 7:
        status_title = "Establishment Phase"
        if crop in perennials:
            status_title = "Sapling Adaptation"
            status_desc = f"Your {crop_name} sapling is currently adjusting its roots to the new soil environment. Focus on stability."
            actions = [
                {"title": "Check Support", "desc": "Ensure the sapling is firmly staked to prevent wind damage.", "type": "info"},
                {"title": "Drip Check", "desc": "Maintain constant, slow moisture near the root ball.", "type": "success"}
            ]
        elif crop in spices:
            status_title = "Rhizome Sprouting"
            status_desc = f"The buried {crop_name} nodes are starting to break dormancy. Do not disturb the topsoil."
            actions = [
                {"title": "Shade Check", "desc": f"Ensure {crop_name} has 50% shade to prevent rhizome drying.", "type": "warning"},
                {"title": "Moisture depth", "desc": "Check moisture 3 inches deep; only surface-water if dry.", "type": "info"}
            ]
        elif crop in pulses:
            status_title = "Germination Phase"
            status_desc = f"Your {crop_name} seeds are emerging. They are highly sensitive to soil crusting."
            actions = [
                {"title": "Crust Breaking", "desc": "Gently break any hardened topsoil to help sprouts emerge.", "type": "warning"},
                {"title": "Bird Deterrent", "desc": "Seeds are attractive to birds now. Inspect the field hourly.", "type": "info"}
            ]
        else: # Vegetables / Default
            status_desc = f"Your {crop_name} is currently rooting. Maintain surface moisture for uniform emergence."
            actions = [
                {"title": "Check Soil Moisture", "desc": "Surface should be damp but not soggy to prevent damping off.", "type": "info"},
                {"title": "Snail Guard", "desc": "Watch for snails and slugs that eat tender new sprouts.", "type": "warning"}
            ]
    elif seed < 15:
        status_title = "Nearing Phase Shift"
        status_desc = f"Your {crop_name} is approaching its next major growth milestone. Support indices are healthy."
        actions = [
            {"title": "Secondary Manuring", "desc": f"Prepare organic fertilizer for the week ahead.", "type": "warning"},
            {"title": "Clean Drainage", "desc": "Ensure field channels are clear before possible rain.", "type": "info"}
        ]
    elif seed < 35:
        status_title = "Minor Resource Stress"
        status_desc = f"Historical data suggests Day {days_elapsed} might see slight nutrient depletion for {crop_name}."
        actions = [
            {"title": "Nitrogen Boost", "desc": f"Apply a light dose of booster to ensure canopy closure.", "type": "warning"},
            {"title": "Pest Shield", "desc": "Apply Neem oil spray to prevent early borer attacks.", "type": "info"}
        ]
    else:
        status_title = "Healthy Progression"
        status_desc = f"General growth parameters for {crop_name} are stable. No local alerts found."
        actions = [
            {"title": "Routine Inspection", "desc": "Check underside of leaves for early egg-laying by pests.", "type": "info"},
            {"title": "Balanced Watering", "desc": "Maintain the current irrigation schedule for stability.", "type": "success"}
        ]
        
    return status_title, status_desc, actions
# Trigger refresh
