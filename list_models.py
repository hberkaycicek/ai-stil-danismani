import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
URL = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"

response = requests.get(URL)
if response.status_code == 200:
    models = response.json()
    print("✅ Erişebildiğin Modeller:")
    for m in models['models']:
        print(f"- {m['name']}")
else:
    print(f"❌ Hata ({response.status_code}): {response.text}")