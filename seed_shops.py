import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'crop_project.settings')
django.setup()

from core.models import SeedShop

def seed_data():
    # Clear existing to avoid duplicates during dev
    SeedShop.objects.all().delete()
    print("Cleared existing shops.")

    # Kerala Districts & Approximate Lat/Lon (simplified)
    # Trivandrum: 8.5, 76.9
    # Kochi: 9.9, 76.2
    # Kozhikode: 11.2, 75.7
    # Wayanad: 11.6, 76.1
    # Palakkad: 10.7, 76.6

    shops = [
        {"name": "Travancore Seeds", "location": "Pattom", "district": "Thiruvananthapuram", "crop_available": "Coconut, Banana, Tapioca", "price_range": "Medium", "contact_info": "0471-123456", "latitude": 8.52, "longitude": 76.93},
        {"name": "Cochin Agro Spices", "location": "Edappally", "district": "Ernakulam", "crop_available": "Pepper, Nutmeg, Ginger", "price_range": "High", "contact_info": "0484-987654", "latitude": 10.02, "longitude": 76.31},
        {"name": "Wayanad Organic Hub", "location": "Kalpetta", "district": "Wayanad", "crop_available": "Coffee, Tea, Pepper", "price_range": "Medium", "contact_info": "04936-112233", "latitude": 11.61, "longitude": 76.08},
        {"name": "Malabar Farmers Co-op", "location": "Mananchira", "district": "Kozhikode", "crop_available": "Coconut, Arecanut", "price_range": "Low", "contact_info": "0495-223344", "latitude": 11.25, "longitude": 75.77},
        {"name": "Palakkad Paddy Center", "location": "Chittur", "district": "Palakkad", "crop_available": "Rice, Mango", "price_range": "Low", "contact_info": "0491-334455", "latitude": 10.78, "longitude": 76.65},
        {"name": "Idukki Spice Garden", "location": "Munnar", "district": "Idukki", "crop_available": "Cardamom, Tea", "price_range": "High", "contact_info": "04865-556677", "latitude": 10.08, "longitude": 77.06},
        {"name": "Kottayam Rubber Nursery", "location": "Kanjikuzhy", "district": "Kottayam", "crop_available": "Rubber, Cocoa", "price_range": "Medium", "contact_info": "0481-667788", "latitude": 9.59, "longitude": 76.52},
        {"name": "Alleppey Coir & Seeds", "location": "Kuttanad", "district": "Alappuzha", "crop_available": "Rice, Duck Farming Feed", "price_range": "Low", "contact_info": "0477-778899", "latitude": 9.49, "longitude": 76.33},
    ]

    for shop_data in shops:
        SeedShop.objects.create(**shop_data)
    
    print(f"Added {len(shops)} Kerala seed shops.")

if __name__ == "__main__":
    seed_data()
