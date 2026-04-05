import os
from pathlib import Path

import requests
from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent / ".env"


def get_istanbul_weather():
    load_dotenv(_ENV_PATH)
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        raise ValueError("WEATHER_API_KEY .env dosyasında tanımlı değil.")

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": "Istanbul,TR",
        "appid": api_key,
        "units": "metric",
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    temperature_c = data["main"]["temp"]
    description = data["weather"][0]["description"]

    return temperature_c, description


if __name__ == "__main__":
    temp, desc = get_istanbul_weather()
    print(f"Sıcaklık: {temp:.1f} °C")
    print(f"Hava durumu: {desc}")
