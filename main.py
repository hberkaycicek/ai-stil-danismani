import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
from weather_api import get_istanbul_weather

# 1. Hazırlık ve Dosya Yolları
base_path = Path(__file__).parent
load_dotenv(base_path / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-flash-latest"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

def load_json(filename):
    path = base_path / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def main():
    print("🌤️  Hava durumu alınıyor...")
    try:
        temp, desc = get_istanbul_weather()
    except Exception as e:
        print(f"❌ Hava durumu alınamadı: {e}")
        return

    print("👔 Dolap ve geçmiş verileri okunuyor...")
    gardrop = load_json("gardrop.json")
    gecmis = load_json("gecmis.json")

    # Yapay Zekaya Gönderilecek Senaryo (Prompt)
    prompt = f"""
    Sen profesyonel bir stil danışmanısın. Kullanıcının bugün şık ve hava durumuna uygun giyinmesini sağla.
    
    GÜNCEL DURUM:
    - Şehir: İstanbul
    - Sıcaklık: {temp}°C
    - Hava Durumu: {desc}
    
    KULLANICININ DOLABI (JSON formatında):
    {json.dumps(gardrop, ensure_ascii=False)}
    
    DÜN GİYİLENLER (Bunları kesinlikle önerme):
    {json.dumps(gecmis, ensure_ascii=False)}
    
    GÖREV:
    1. Dolaptaki kıyafetlerden (Ayakkabı, Pantolon, Üst Giyim) bir kombin yap.
    2. Neden bu kombini seçtiğini (hava durumuna ve şıklığa göre) 2 cümlede açıkla.
    3. Yanıtını çok samimi ve enerjik bir tonda ver.
    """

    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    print("🤖 Yapay zeka kombin hazırlıyor...\n")
    try:
        response = requests.post(URL, json=payload, headers={'Content-Type': 'application/json'})
        res_data = response.json()
        ai_suggestion = res_data['candidates'][0]['content']['parts'][0]['text']
        
        print("--- GÜNÜN KOMBİN ÖNERİSİ ---")
        print(ai_suggestion)
        print("----------------------------")

    except Exception as e:
        print(f"❌ AI hatası: {e}")

if __name__ == "__main__":
    main()