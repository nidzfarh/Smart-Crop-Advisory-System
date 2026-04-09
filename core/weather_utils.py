import requests
from datetime import datetime, timedelta

# Historical monthly climate averages (fallback if API fails)
KERALA_DISTRICT_CLIMATE = {
    'Thiruvananthapuram': {'rainfall': 150, 'temp': 27, 'humidity': 78},
    'Kollam':             {'rainfall': 208, 'temp': 27, 'humidity': 79},
    'Pathanamthitta':     {'rainfall': 241, 'temp': 26, 'humidity': 80},
    'Alappuzha':          {'rainfall': 250, 'temp': 27, 'humidity': 81},
    'Kottayam':           {'rainfall': 258, 'temp': 27, 'humidity': 80},
    'Idukki':             {'rainfall': 275, 'temp': 24, 'humidity': 82},
    'Ernakulam':          {'rainfall': 258, 'temp': 28, 'humidity': 78},
    'Thrissur':           {'rainfall': 258, 'temp': 28, 'humidity': 77},
    'Palakkad':           {'rainfall': 191, 'temp': 29, 'humidity': 70},
    'Malappuram':         {'rainfall': 241, 'temp': 28, 'humidity': 76},
    'Kozhikode':          {'rainfall': 275, 'temp': 28, 'humidity': 78},
    'Wayanad':            {'rainfall': 283, 'temp': 23, 'humidity': 83},
    'Kannur':             {'rainfall': 283, 'temp': 28, 'humidity': 78},
    'Kasaragod':          {'rainfall': 291, 'temp': 28, 'humidity': 78},
}

# Latitude/Longitude for Open-Meteo API
KERALA_DISTRICT_COORDS = {
    'Thiruvananthapuram': (8.5241,  76.9366),
    'Kollam':             (8.8932,  76.6141),
    'Pathanamthitta':     (9.2648,  76.7870),
    'Alappuzha':          (9.4981,  76.3388),
    'Kottayam':           (9.5916,  76.5222),
    'Idukki':             (9.9189,  77.1025),
    'Ernakulam':          (9.9312,  76.2673),
    'Thrissur':           (10.5276, 76.2144),
    'Palakkad':           (10.7867, 76.6548),
    'Malappuram':         (11.0510, 76.0711),
    'Kozhikode':          (11.2588, 75.7804),
    'Wayanad':            (11.6854, 76.1320),
    'Kannur':             (11.8745, 75.3704),
    'Kasaragod':          (12.4996, 74.9869),
}


def get_district_climate(district):
    """Returns historical average climate (monthly) for a Kerala district. Used as fallback."""
    return KERALA_DISTRICT_CLIMATE.get(district, {'rainfall': 240, 'temp': 27, 'humidity': 77})


def get_historical_climate(district, years=5):
    """
    Fetches the same ±15-day window across the past `years` years using
    Open-Meteo archive API, then averages them to smooth seasonal anomalies.
    Returns averaged temp, humidity, rainfall or None if all calls fail.
    """
    coords = KERALA_DISTRICT_COORDS.get(district)
    if not coords:
        return None

    lat, lon = coords
    today = datetime.today().date()

    all_temps, all_humid, all_rain = [], [], []

    for y in range(1, years + 1):
        try:
            mid   = today.replace(year=today.year - y)
            start = mid - timedelta(days=15)
            end   = mid + timedelta(days=15)
        except ValueError:
            continue  # skip leap-year edge cases

        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
            f"relative_humidity_2m_max,relative_humidity_2m_min"
            f"&timezone=Asia/Kolkata"
            f"&start_date={start}&end_date={end}"
        )

        try:
            r = requests.get(url, timeout=8)
            if r.status_code != 200:
                continue
            d     = r.json().get('daily', {})
            t_max = [v for v in d.get('temperature_2m_max', []) if v is not None]
            t_min = [v for v in d.get('temperature_2m_min', []) if v is not None]
            prcp  = [v for v in d.get('precipitation_sum', []) if v is not None]
            rh_mx = [v for v in d.get('relative_humidity_2m_max', []) if v is not None]
            rh_mn = [v for v in d.get('relative_humidity_2m_min', []) if v is not None]
            if not t_max:
                continue

            avg_t  = (sum(t_max)/len(t_max) + sum(t_min)/len(t_min)) / 2
            avg_h  = (sum(rh_mx)/len(rh_mx) + sum(rh_mn)/len(rh_mn)) / 2 if rh_mx else 77
            avg_r  = sum(prcp) / max(len(prcp), 1)

            all_temps.append(avg_t)
            all_humid.append(avg_h)
            all_rain.append(avg_r)

        except Exception as e:
            print(f"Historical weather error (year -{y}): {e}")
            continue

    if not all_temps:
        return None

    return {
        'temp':     round(sum(all_temps) / len(all_temps), 1),
        'humidity': round(sum(all_humid) / len(all_humid), 0),
        'rainfall': round(sum(all_rain)  / len(all_rain),  1),
        'years_used': len(all_temps),
    }


def get_harvest_forecast(district, days_to_harvest):
    """
    Estimates average temperature and rainfall at harvest time using:
     - Open-Meteo forecast (if within 16 days)
     - Open-Meteo archive of same period last year (otherwise)
    Returns {'temp', 'humidity', 'rainfall', 'source'} or None.
    """
    coords = KERALA_DISTRICT_COORDS.get(district)
    if not coords:
        return None

    lat, lon = coords
    today        = datetime.today().date()
    harvest_date = today + timedelta(days=days_to_harvest)
    window_start = harvest_date - timedelta(days=7)
    window_end   = harvest_date + timedelta(days=7)

    # Try short-range forecast (≤ 16 days)
    if days_to_harvest <= 16:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
            f"relative_humidity_2m_max,relative_humidity_2m_min"
            f"&timezone=Asia/Kolkata"
            f"&start_date={today}&end_date={window_end}"
        )
        source = "short-range forecast"
    else:
        # Use archive from same period last year
        ly_start = window_start.replace(year=window_start.year - 1)
        ly_end   = window_end.replace(year=window_end.year - 1)
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
            f"relative_humidity_2m_max,relative_humidity_2m_min"
            f"&timezone=Asia/Kolkata"
            f"&start_date={ly_start}&end_date={ly_end}"
        )
        source = "historical same-period last year"

    try:
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return None
        d = r.json().get('daily', {})
        t_max = [v for v in d.get('temperature_2m_max', []) if v is not None]
        t_min = [v for v in d.get('temperature_2m_min', []) if v is not None]
        prcp  = [v for v in d.get('precipitation_sum', []) if v is not None]
        rh_mx = [v for v in d.get('relative_humidity_2m_max', []) if v is not None]
        rh_mn = [v for v in d.get('relative_humidity_2m_min', []) if v is not None]
        if not t_max:
            return None
        return {
            'temp':     round((sum(t_max)/len(t_max) + sum(t_min)/len(t_min)) / 2, 1),
            'humidity': round((sum(rh_mx)/len(rh_mx) + sum(rh_mn)/len(rh_mn)) / 2, 0) if rh_mx else 77,
            'rainfall': round(sum(prcp) / max(len(prcp), 1), 1),
            'source':   source,
            'harvest_date': str(harvest_date),
        }
    except Exception as e:
        print(f"Harvest forecast error: {e}")
        return None


def _weather_icon(precip, is_day=True):
    """Returns an emoji icon based on precipitation level."""
    if precip is None:
        return '🌡️'
    if precip > 15:
        return '⛈️'
    if precip > 5:
        return '🌧️'
    if precip > 0.5:
        return '🌦️'
    return '☀️' if is_day else '🌙'


def get_weather_forecast(district):
    """
    Fetches today + next 9 days weather using Open-Meteo (free, no API key).
    Returns a rich dict for the unified weather widget.
    Falls back to None on any error.
    """
    coords = KERALA_DISTRICT_COORDS.get(district)
    if not coords:
        return None

    lat, lon = coords
    today = datetime.today().date()
    end_date = today + timedelta(days=9)

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
        f"relative_humidity_2m_max,relative_humidity_2m_min,windspeed_10m_max"
        f"&timezone=Asia/Kolkata"
        f"&start_date={today}&end_date={end_date}"
    )

    try:
        response = requests.get(url, timeout=8)
        if response.status_code != 200:
            return None

        data = response.json().get('daily', {})
        dates    = data.get('time', [])
        t_max    = data.get('temperature_2m_max', [])
        t_min    = data.get('temperature_2m_min', [])
        precip   = data.get('precipitation_sum', [])
        rh_max   = data.get('relative_humidity_2m_max', [])
        rh_min   = data.get('relative_humidity_2m_min', [])
        wind     = data.get('windspeed_10m_max', [])

        days = []
        for i, d in enumerate(dates):
            dt = datetime.strptime(d, "%Y-%m-%d")
            p  = precip[i] if i < len(precip) and precip[i] is not None else 0
            mx = t_max[i]  if i < len(t_max)  and t_max[i]  is not None else 0
            mn = t_min[i]  if i < len(t_min)  and t_min[i]  is not None else 0
            rh = ((rh_max[i] or 0) + (rh_min[i] or 0)) / 2 if i < len(rh_max) else 0
            wn = wind[i]   if i < len(wind)   and wind[i]   is not None else 0

            days.append({
                'date':       d,
                'label':      'Today' if dt.date() == today else dt.strftime('%a %d'),
                'icon':       _weather_icon(p),
                'temp_max':   round(mx, 1),
                'temp_min':   round(mn, 1),
                'temp_avg':   round((mx + mn) / 2, 1),
                'precip':     round(p, 1),
                'humidity':   round(rh, 0),
                'wind':       round(wn, 1),
                'is_today':   dt.date() == today,
            })

        return {
            'days':  days,
            'today': days[0] if days else None,
        }

    except Exception as e:
        print(f"Open-Meteo forecast error: {e}")
        return None
