"""
Weather API endpoint for Horisation.
Pulls current weather from Open-Meteo (free, no API key needed).
Caches results in memory for 10 minutes to avoid redundant API calls.
"""

from flask import Blueprint, jsonify
from urllib.request import urlopen
import json
import time

weather_bp = Blueprint('weather', __name__)

# St. Louis coordinates
LAT = 38.6270
LON = -90.1994

_cache = {'data': None, 'ts': 0}
CACHE_TTL = 600  # 10 minutes

WMO_CODES = {
    0:  {'icon': 'fa-sun',           'label': 'Clear'},
    1:  {'icon': 'fa-sun',           'label': 'Mostly clear'},
    2:  {'icon': 'fa-cloud-sun',     'label': 'Partly cloudy'},
    3:  {'icon': 'fa-cloud',         'label': 'Overcast'},
    45: {'icon': 'fa-smog',          'label': 'Foggy'},
    48: {'icon': 'fa-smog',          'label': 'Depositing rime fog'},
    51: {'icon': 'fa-cloud-rain',    'label': 'Light drizzle'},
    53: {'icon': 'fa-cloud-rain',    'label': 'Moderate drizzle'},
    55: {'icon': 'fa-cloud-rain',    'label': 'Dense drizzle'},
    56: {'icon': 'fa-cloud-rain',    'label': 'Light freezing drizzle'},
    57: {'icon': 'fa-cloud-rain',    'label': 'Dense freezing drizzle'},
    61: {'icon': 'fa-cloud-showers-heavy', 'label': 'Slight rain'},
    63: {'icon': 'fa-cloud-showers-heavy', 'label': 'Moderate rain'},
    65: {'icon': 'fa-cloud-showers-heavy', 'label': 'Heavy rain'},
    66: {'icon': 'fa-cloud-showers-heavy', 'label': 'Light freezing rain'},
    67: {'icon': 'fa-cloud-showers-heavy', 'label': 'Heavy freezing rain'},
    71: {'icon': 'fa-snowflake',     'label': 'Slight snow'},
    73: {'icon': 'fa-snowflake',     'label': 'Moderate snow'},
    75: {'icon': 'fa-snowflake',     'label': 'Heavy snow'},
    77: {'icon': 'fa-snowflake',     'label': 'Snow grains'},
    80: {'icon': 'fa-cloud-rain',    'label': 'Slight rain showers'},
    81: {'icon': 'fa-cloud-rain',    'label': 'Moderate rain showers'},
    82: {'icon': 'fa-cloud-rain',    'label': 'Violent rain showers'},
    85: {'icon': 'fa-snowflake',     'label': 'Slight snow showers'},
    86: {'icon': 'fa-snowflake',     'label': 'Heavy snow showers'},
    95: {'icon': 'fa-bolt',          'label': 'Thunderstorm'},
    96: {'icon': 'fa-bolt',          'label': 'Thunderstorm with slight hail'},
    99: {'icon': 'fa-bolt',          'label': 'Thunderstorm with heavy hail'},
}


def _time_of_day():
    """Return 'morning', 'afternoon', or 'evening' based on current hour."""
    h = time.localtime().tm_hour
    if h < 12:
        return 'morning'
    elif h < 18:
        return 'afternoon'
    else:
        return 'evening'


def _greeting(name):
    t = _time_of_day()
    return {
        'morning':   f'Good morning, {name}',
        'afternoon': f'Good afternoon, {name}',
        'evening':   f'Good evening, {name}',
    }[t]


@weather_bp.route('/api/weather')
def get_weather():
    now = time.time()

    # Return cached data if still fresh
    if _cache['data'] and now - _cache['ts'] < CACHE_TTL:
        return jsonify({'ok': True, ** _cache['data']})

    # Fetch from Open-Meteo
    url = (
        f'https://api.open-meteo.com/v1/forecast?'
        f'latitude={LAT}&longitude={LON}'
        f'&current=temperature_2m,apparent_temperature,weather_code'
        f'&daily=sunrise,sunset'
        f'&timezone=America/Chicago'
    )
    try:
        resp = urlopen(url, timeout=5)
        data = json.loads(resp.read().decode())
    except Exception as e:
        # If fetch fails, return stale cache if we have it
        if _cache['data']:
            return jsonify({'ok': True, ** _cache['data'], 'stale': True})
        return jsonify({'ok': False, 'error': str(e)}), 502

    current = data.get('current', {})
    daily   = data.get('daily', {})

    temp       = current.get('temperature_2m')
    feels_like = current.get('apparent_temperature')
    wmo_code   = current.get('weather_code', 0)
    sunrise    = (daily.get('sunrise') or [''])[0] if daily.get('sunrise') else ''
    sunset     = (daily.get('sunset') or [''])[0] if daily.get('sunset') else ''

    weather_info = WMO_CODES.get(wmo_code, {'icon': 'fa-question', 'label': 'Unknown'})

    result = {
        'city':       'St. Louis',
        'temp':       round(temp) if temp is not None else None,
        'feels_like': round(feels_like) if feels_like is not None else None,
        'icon':       weather_info['icon'],
        'label':      weather_info['label'],
        'sunrise':    sunrise,
        'sunset':     sunset,
        'greeting':   _greeting('{name}'),  # placeholder, frontend replaces {name}
    }

    _cache['data'] = result
    _cache['ts']   = now
    return jsonify({'ok': True, **result})
