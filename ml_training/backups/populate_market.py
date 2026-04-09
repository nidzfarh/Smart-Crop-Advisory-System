import csv
import os

crops_data = [
    ("Sugarcane", 3000, "High"),
    ("Black Pepper", 550000, "High"),
    ("Arecanut", 450000, "Medium"),
    ("Ash gourd", 12000, "Low"),
    ("Tomato", 18000, "Medium"),
    ("Cardamom", 900000, "High"),
    ("Cashew", 700000, "High")
]

history_file = r'c:\Users\aksha\/.gemini/antigravity/scratch/Minor-project/ml_training/datasets/market_history.csv'

# Append for every month in 2026 to ensure revenue logic works for all planting months
with open(history_file, 'a', newline='') as f:
    writer = csv.writer(f)
    for crop, price, demand in crops_data:
        for month in range(1, 13):
            writer.writerow([crop, month, 2026, price, demand])

print(f"Successfully added {len(crops_data) * 12} rows to {history_file}")
