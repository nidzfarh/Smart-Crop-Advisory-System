# AI-Driven Crop Lifecycle Decision Support System (Kerala Edition)

A Django-based web application tailored for Kerala farmers to calculate the most profitable crops, find nearby seed shops, and track their farming progress.

## Key Features
- **Kerala-Centric Data**: Optimized for Kerala districts and local crops.
- **User Dashboard**: Personalized hub to manage farm activities.
- **Crop Recommendation**: AI-powered suggestions based on soil (N, P, K, pH) and weather.
- **Yield & Profit Prediction**: Estimates yield and calculates potential profit based on market prices.
- **Market Integration**: Finds nearby seed shops in Kerala and lists them by price and distance.
- **Crop Tracking**: Monitor the growth status of your planted crops.

## Setup Instructions

### 1. Prerequisites
- Python 3.8+
- [Git](https://git-scm.com/)

### 2. Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Initialize System
Run the following commands to set up the database and trained models:

1. **Generate Data & Train Models**:
   ```bash
   python ml_training/generate_data.py
   python ml_training/train_crop_model.py
   python ml_training/train_yield_model.py
   ```
2. **Setup Database**:
   ```bash
   python manage.py makemigrations core
   python manage.py migrate
   ```
3. **Seed Kerala Shop Data**:
   ```bash
   python seed_shops.py
   ```

### 4. Running the Application
Start the server:
```bash
python manage.py runserver
```
Visit: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## User Verification Flow
1.  **Register/Login**: Create a new farmer account.
2.  **Dashboard**:
    - Click **"Analyze Soil"** to find the best crop.
    - Click **"Add New Crop"** to track crops you are already growing.
3.  **Result Analysis**:
    - View top crops ranked by profit.
    - Click **"Find Shops"** to see sellers in Kerala tailored to your crop.
4.  **Market View**:
    - See shops sorted by price and estimated distance from your district.
