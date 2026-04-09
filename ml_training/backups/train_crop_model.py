import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Ensure model directory exists
os.makedirs('ml_models', exist_ok=True)

# Load dataset
df = pd.read_csv('ml_training/datasets/crop_recommendation.csv')

# Handle column naming variations
if 'pH' in df.columns:
    df.rename(columns={'pH': 'ph'}, inplace=True)

# Geographic and Soil soft-weighting array (Prioritize natively, DO NOT DELETE)
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
    crop = str(row.get('label', '')).lower().strip()
    dist = str(row.get('district', '')).strip()
    soil = str(row.get('soil_type', '')).lower().strip()
    
    # If the user perfectly matched their strict District+Soil mapping, bump weight massively (Primary Constraint)
    if dist in DISTRICT_SOIL_PRIORITY and soil in DISTRICT_SOIL_PRIORITY[dist]:
        if crop in DISTRICT_SOIL_PRIORITY[dist][soil]:
            return 50.0
            
    # Explicit mapping for Tea
    if crop == 'tea' and dist in ['Idukki', 'Wayanad'] and soil in ['forest', 'loamy']:
        return 50.0
        
    # Baseline weight for all other User-Uploaded secondary metrics (DO NOT DELETE)
    return 1.0

df['weight'] = df.apply(get_weight, axis=1)

# Encode Soil Type and District
soil_encoder = LabelEncoder()
df['soil_type_encoded'] = soil_encoder.fit_transform(df['soil_type'])
joblib.dump(soil_encoder, 'ml_models/soil_encoder.pkl')

district_encoder = LabelEncoder()
df['district_encoded'] = district_encoder.fit_transform(df['district'])
joblib.dump(district_encoder, 'ml_models/district_encoder.pkl')

# Force numeric vectors for critical inputs safely
num_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

target_col = 'label' if 'label' in df.columns else 'crop'

# Naturally scrub unconvertible rogue string rows and entirely blank targets/geography
df = df.dropna(subset=num_cols + [target_col, 'district', 'soil_type'])

# Features and Target
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'soil_type_encoded', 'district_encoded']]
y = df[target_col]
w = df['weight']

# Split data
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=0.2, random_state=42)

# Train model natively and heavily
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train, sample_weight=w_train)

# Save model
joblib.dump(model, 'ml_models/crop_model.pkl')

# Result
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Crop model trained data accuracy: {train_acc:.4f} | test data accuracy: {test_acc:.4f}")
