from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from .forms import FarmDataInputForm, UserRegisterForm, CropTrackingForm
from .models import SeedShop, UserCrop, UserProfile
from core.advisory import get_crop_advice, get_weather_alert, get_crop_timeline, CROP_GROWING_DAYS, generate_crop_rationale, get_crop_status
from core.ml_utils import (
    predict_top_crops, predict_yield, get_advisory,
    get_seasonal_market_estimate, get_seed_cost, get_seed_details,
    get_all_crops
)
import pandas as pd
import math
import random
from datetime import date, datetime, timedelta
from .weather_utils import get_district_climate, get_weather_forecast, get_historical_climate, get_harvest_forecast

# Kerala District Coordinates (Mock/Estimated centroids)
DISTRICT_COORDS = {
    'Alappuzha': (9.49, 76.33),
    'Ernakulam': (9.98, 76.29),
    'Idukki': (9.85, 77.08),
    'Kannur': (11.87, 75.37),
    'Kasaragod': (12.51, 74.98),
    'Kollam': (8.89, 76.61),
    'Kottayam': (9.58, 76.52),
    'Kozhikode': (11.25, 75.78),
    'Malappuram': (11.07, 76.07),
    'Palakkad': (10.78, 76.65),
    'Pathanamthitta': (9.26, 76.78),
    'Thiruvananthapuram': (8.52, 76.93),
    'Thrissur': (10.52, 76.21),
    'Wayanad': (11.60, 76.08),
    'Kerala': (10.15, 76.50) # Fallback center
}

def calculate_distance(lat1, lon1, lat2, lon2):
    """Haversine formula to calculate the distance between two points on Earth."""
    if not all([lat1, lon1, lat2, lon2]): return 999
    
    R = 6371.0 # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 1)

def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            # Save Profile
            district = form.cleaned_data['district']
            UserProfile.objects.create(user=user, district=district)
            
            login(request, user)
            return redirect('dashboard')
    else:
        form = UserRegisterForm()
    return render(request, 'core/register.html', {'form': form})

def user_login(request):
    # Using simple custom login view or Django's built-in. 
    # For simplicity in this mini-project, manual handling or built-in is fine.
    # Let's use Django's built-in via urls.py but if custom needed:
    return render(request, 'core/login.html') 
    # Note: We will use Django's LoginView in urls.py, this is just a placeholder if needed.

@login_required
def dashboard(request):
    user_crops = UserCrop.objects.filter(user=request.user)
    
    # Enrich crop objects with advisory data
    enriched_crops = []
    for crop in user_crops:
        days = (date.today() - crop.planting_date).days
        if days < 0: days = 0
        stage, advice = get_crop_advice(crop.crop_name, days)
        enriched_crops.append({
            'obj': crop,
            'stage': stage,
            'days': days,
            'advice': advice
        })
    
    # Get Location & Weather
    try:
        profile = request.user.userprofile
        location = profile.district
    except UserProfile.DoesNotExist:
        location = "Kerala"
        
    weather_alert = get_weather_alert(location)
    forecast = get_weather_forecast(location)
        
    return render(request, 'core/dashboard.html', {
        'enriched_crops': enriched_crops, 
        'location': location,
        'weather_alert': weather_alert,
        'forecast': forecast
    })

@login_required
def add_crop(request):
    if request.method == 'POST':
        form = CropTrackingForm(request.POST)
        if form.is_valid():
            crop = form.save(commit=False)
            crop.user = request.user
            crop.save()
            return redirect('crop_detail', crop_id=crop.id)
    else:
        form = CropTrackingForm()
    
    crops = get_all_crops()
    return render(request, 'core/add_crop.html', {'form': form, 'crops': crops})

@login_required
def crop_detail(request, crop_id):
    crop = get_object_or_404(UserCrop, id=crop_id, user=request.user)
    timeline = get_crop_timeline(crop.crop_name, crop.planting_date)
    
    # Calculate current progress
    days_elapsed = (date.today() - crop.planting_date).days
    total_days = timeline[-1]['days_offset'] if timeline else 120
    progress_percentage = min(100, (days_elapsed / total_days) * 100) if total_days > 0 else 100
    
    # Get dynamic status and recommended actions
    stage_title, stage_desc = get_crop_advice(crop.crop_name, days_elapsed)
    status_title = f"{stage_title.title()} Phase"
    status_desc = stage_desc
    _, _, actions = get_crop_status(crop.crop_name, days_elapsed, crop.id)
    
    # Calendar logic: Generate all months from planting to harvest
    import calendar
    today = date.today()
    planting_date = crop.planting_date
    harvest_date = timeline[-1]['date'] if timeline else planting_date + timedelta(days=120)
    
    # Start from the first day of the planting month
    current_date = planting_date.replace(day=1)
    # End at the last month of harvest
    end_date = harvest_date.replace(day=1)
    
    full_calendar = []
    milestone_lookup = {ev['date']: ev for ev in timeline}
    
    while current_date <= end_date:
        cal = calendar.Calendar(calendar.SUNDAY)
        month_days = cal.monthdatescalendar(current_date.year, current_date.month)
        
        month_data = {
            'month_name': current_date.strftime("%B %Y"),
            'days': []
        }
        
        for week in month_days:
            for day in week:
                month_data['days'].append({
                    'date': day,
                    'is_today': day == today,
                    'milestone': milestone_lookup.get(day) if day.month == current_date.month else None,
                    'is_current_month': day.month == current_date.month
                })
        
        full_calendar.append(month_data)
        
        # Advance to next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)

    return render(request, 'core/crop_detail.html', {
        'crop': crop,
        'timeline': timeline,
        'days_elapsed': days_elapsed,
        'progress': progress_percentage,
        'status_title': status_title,
        'status_desc': status_desc,
        'actions': actions,
        'full_calendar': full_calendar,
        'today': today
    })

def index(request):
    # Public landing page or redirect to dashboard if logged in?
    # Requirement: "first should be a sign page... then dash board"
    if request.user.is_authenticated:
        return redirect('dashboard')
    return render(request, 'core/index.html') # This will be the landing/login choice page

@login_required
def predict_view(request):
    # Pre-fill location from profile
    try:
        location = request.user.userprofile.district
    except:
        location = ''
        
    form = FarmDataInputForm(initial={'location': location})
    crops = get_all_crops()
    return render(request, 'core/predict.html', {'form': form, 'crops': crops})

@login_required
def result(request):
    if request.method == 'POST':
        form = FarmDataInputForm(request.POST)
        if form.is_valid():
            # ... (Existing logic) ...
            data = form.cleaned_data
            # ... (Existing logic) ...
            n = data['nitrogen']
            p = data['phosphorus']
            k = data['potassium']
            ph = data['ph']
            soil_type = data['soil_type']
            user_crop = data['crop_name'].strip().title() if data['crop_name'] else None
            location = data['location']

            # 1. Get current/baseline climate (district profile as fallback)
            climate_base = get_district_climate(location)

            # 2. Fetch Live Real-Time Open-Meteo Forecast Data
            live_forecast = get_weather_forecast(location)

            if live_forecast and live_forecast.get('today'):
                today = live_forecast['today']
                temp = round(today['temp_avg'], 1)
                humidity = round(today['humidity'], 1)
                
                # ML model expects Monthly Rainfall metric (~150-350mm). 
                # We project the 9-day live forecast precipitation into a monthly equivalent.
                nine_day_rain = sum(d.get('precip', 0) for d in live_forecast.get('days', []))
                projected_monthly_rain = round(nine_day_rain * 3.33, 1)
                
                # Blend dynamically with historical baseline to prevent severe drought/flood outlier shocks
                rain = round((climate_base['rainfall'] + projected_monthly_rain) / 2, 1)
            else:
                # Fallback to historical averages if API is momentarily disconnected
                hist = get_historical_climate(location)
                if hist:
                    temp = round((climate_base['temp'] + hist['temp']) / 2, 1)
                    humidity = round((climate_base['humidity'] + hist['humidity']) / 2, 1)
                    rain = round((climate_base['rainfall'] + hist['rainfall']) / 2, 1)
                else:
                    temp = climate_base['temp']
                    humidity = climate_base['humidity']
                    rain = climate_base['rainfall']

            current_weather = {'temp': temp, 'humidity': humidity, 'rainfall': rain}

            # Farm Size Calculations
            land_value = form.cleaned_data.get('land_area_value')
            land_unit = form.cleaned_data.get('land_area_unit', 'Acre')
            land_size_acres = None

            if land_value:
                land_value = float(land_value)
                if land_unit == 'Cent':
                    land_size_acres = land_value / 100.0
                elif land_unit == 'Hectare':
                    land_size_acres = land_value * 2.471
                else: # Acre
                    land_size_acres = land_value

            # Multi-Stage Intelligence: Get top 8 compatible candidates for financial vetting
            top_crops_candidates = predict_top_crops(n, p, k, temp, humidity, ph, rain, soil_type, location, top_n=8)

            crops_to_evaluate = list(top_crops_candidates)
            if user_crop and user_crop not in crops_to_evaluate:
                crops_to_evaluate.insert(0, user_crop)

            all_candidates = []
            for crop in crops_to_evaluate:
                # 1. Predict Yield (ML model)
                predicted_yield_acre = predict_yield(crop, n, p, k, temp, rain)
                
                # 2. Determine Harvest Time
                growing_days = CROP_GROWING_DAYS.get(crop, 120)
                harvest_date = datetime.now() + timedelta(days=growing_days)
                harvest_month = harvest_date.month
                
                # 3. Fetch Seasonal Market & Seed Data
                m_price, m_demand = get_seasonal_market_estimate(crop, harvest_month)
                s_rate, s_unit, s_price, s_approx = get_seed_details(crop)
                
                # 4. Economic Calculation per Acre
                seed_cost_acre = s_rate * s_price
                revenue_acre = predicted_yield_acre * m_price
                profit_acre = revenue_acre - seed_cost_acre
                
                # Demand weighting for ranking
                demand_bias = {'High': 1.25, 'Medium': 1.0, 'Low': 0.75}.get(m_demand, 1.0)
                adj_profit_acre = profit_acre * demand_bias
                
                # 5. Personalized Scaling if land size available
                total_yield = total_revenue = total_seed_qty = total_seed_cost = total_profit = 0.0
                if land_size_acres:
                    total_yield = predicted_yield_acre * land_size_acres
                    total_revenue = total_yield * m_price
                    total_seed_qty = s_rate * land_size_acres
                    total_seed_cost = total_seed_qty * s_price
                    total_profit = total_revenue - total_seed_cost
                
                # Profit Viability
                if m_demand == 'High' and (adj_profit_acre > 5000):
                    viability = 'High'
                elif adj_profit_acre > 0:
                    viability = 'Moderate'
                else:
                    viability = 'Low'

                harvest_wx = get_harvest_forecast(location, growing_days)
                rationale = generate_crop_rationale(crop, n, p, k, ph, current_weather, harvest_wx)

                all_candidates.append({
                    'crop':           crop,
                    'yield_acre':     round(predicted_yield_acre, 2),
                    'price':          m_price,
                    'demand':         m_demand,
                    'seed_rate_acre': s_rate,
                    'seed_unit':      s_unit,
                    'seed_price':     s_price,
                    'seed_approx':    s_approx,
                    'seed_cost_acre': seed_cost_acre,
                    'revenue_acre':   round(revenue_acre, 2),
                    'profit_acre':    round(profit_acre, 2),
                    'adj_profit':     adj_profit_acre,
                    'land_size_acres': land_size_acres,
                    'total_yield':     round(total_yield, 2),
                    'total_revenue':   round(total_revenue, 2),
                    'total_seed_qty':  round(total_seed_qty, 1),
                    'total_seed_cost': round(total_seed_cost, 2),
                    'total_profit':    round(total_profit, 2),
                    'viability':       viability,
                    'advisory':       get_advisory(crop),
                    'rationale':      rationale,
                    'harvest_days':   growing_days,
                    'harvest_month_label': harvest_date.strftime('%B'),
                    'harvest_wx':     harvest_wx,
                })

            # HYBRID FILTERING:
            # 1. Always keep the user_crop (requested manually) so they see the analysis even if it's a loss
            # 2. For the rest, filter out non-profitable crops and rank by highest adjusted profit.
            results = []
            if user_crop:
                # Find the user's requested crop in candidates
                user_res = next((c for c in all_candidates if c['crop'].lower() == user_crop.lower()), None)
                if user_res: 
                    results.append(user_res)
                    all_candidates.remove(user_res)

            # Filter remaining for profit and sort
            profitable_candidates = [c for c in all_candidates if c['profit_acre'] > 0]
            profitable_candidates.sort(key=lambda x: x['adj_profit'], reverse=True)
            
            # Combine up to 3 results total
            results.extend(profitable_candidates)
            results = results[:3]
            
            # Fallback: if no profitable crops found at all, show top suitable candidates (even if loss)
            if not results:
                results = all_candidates[:3]

            # Add Rank index
            for i, res in enumerate(results):
                res['rank'] = i + 1
            
            # Save location in session
            request.session['user_location'] = location

            # Fetch shops and calculate REAL distance based on Analyze-page District
            all_shops = SeedShop.objects.all()
            user_coords = DISTRICT_COORDS.get(location, DISTRICT_COORDS['Kerala'])
            
            shops_with_distance = []
            for shop in all_shops:
                # Use shop's district to find its coords
                shop_coords = DISTRICT_COORDS.get(shop.district, (0.0, 0.0))
                
                # If shop district coords aren't in our map, fall back to its lat/lon fields
                if shop_coords == (0.0, 0.0):
                    dist = calculate_distance(user_coords[0], user_coords[1], shop.latitude, shop.longitude)
                else:
                    dist = calculate_distance(user_coords[0], user_coords[1], shop_coords[0], shop_coords[1])
                
                # Small random jitter for realism (avoid exact 0 for same district)
                if dist == 0: dist = round(math.sqrt(random.randint(4, 100)), 1)
                
                shops_with_distance.append({
                    'name': shop.name,
                    'location': shop.location,
                    'crop_available': shop.crop_available,
                    'price_range': shop.price_range,
                    'contact_info': shop.contact_info,
                    'distance': dist
                })
            
            # Sort by distance (Increasing)
            shops_with_distance.sort(key=lambda x: x['distance'])
            
            forecast = get_weather_forecast(location)

            return render(request, 'core/result.html', {
                'results': results,
                'shops': shops_with_distance,
                'location': location,
                'land_info': {
                    'value': land_value,
                    'unit': land_unit,
                    'acres': land_size_acres
                },
                'climate': {
                    'temp': temp,
                    'humidity': humidity,
                    'rainfall': rain,
                    'ph': ph,
                },
                'forecast': forecast,
            })
    else:
        form = FarmDataInputForm() # Create empty form for GET request if needed or redirect

    # If form invalid or GET, show form with errors
    return render(request, 'core/predict.html', {'form': form})

@login_required
def market_view(request, crop_name):
    user_location = request.session.get('user_location', 'Kerala')
    
    # Simple distance mock (Randomized or based on District matching if implemented)
    # Since we don't have user's lat/lon, we just return shops in "Kerala" or match district text if possible.
    # Requirement: "distance of that shop"
    
    all_shops = SeedShop.objects.all()
    user_coords = DISTRICT_COORDS.get(user_location, DISTRICT_COORDS['Kerala'])
    
    shops_with_distance = []
    import random
    for shop in all_shops:
        shop_coords = DISTRICT_COORDS.get(shop.district, (0.0, 0.0))
        
        if shop_coords == (0.0, 0.0):
            dist = calculate_distance(user_coords[0], user_coords[1], shop.latitude, shop.longitude)
        else:
            dist = calculate_distance(user_coords[0], user_coords[1], shop_coords[0], shop_coords[1])
            
        if dist == 0: dist = round(math.sqrt(random.randint(4, 100)), 1)
        
        shops_with_distance.append({
            'shop': shop,
            'distance': dist
        })
    
    # Sort primarily by Distance (Increasing), then Price
    price_map = {'Low': 1, 'Medium': 2, 'High': 3}
    shops_with_distance.sort(key=lambda x: (x['distance'], price_map.get(x['shop'].price_range, 2)))
    
    # Sort by price (mock logic: check price_range or just assume random for "cheaply")
    # Using price_range: Low=1, Medium=2, High=3
    price_map = {'Low': 1, 'Medium': 2, 'High': 3}
    shops_with_distance.sort(key=lambda x: (price_map.get(x['shop'].price_range, 2), x['distance']))

    return render(request, 'core/market.html', {'shops': shops_with_distance, 'crop': crop_name})
@login_required
def delete_crop(request, crop_id):
    crop = get_object_or_404(UserCrop, id=crop_id, user=request.user)
    if request.method == 'POST':
        crop.delete()
        return redirect('dashboard')
    return redirect('dashboard')
