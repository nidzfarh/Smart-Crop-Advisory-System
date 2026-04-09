import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Ensure model directory exists
os.makedirs('ml_models', exist_ok=True)

# Load dataset
df = pd.read_csv('ml_training/datasets/yield_data.csv')

# Encode categorical data (Crop)
le = LabelEncoder()
df['Crop'] = le.fit_transform(df['Crop'])

# Save label encoder for later use
joblib.dump(le, 'ml_models/yield_label_encoder.pkl')

# Features and Target
# Using Crop as a feature along with environmental data
X = df[['Crop', 'N', 'P', 'K', 'Temperature', 'Rainfall']]
y = df['Yield_Tons_Acre']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'ml_models/yield_model.pkl')
print(f"Yield model trained with R2 score: {model.score(X_test, y_test):.2f}")
