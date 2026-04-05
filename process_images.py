import os
import json
import base64
import requests
import time
from pathlib import Path
from dotenv import load_dotenv

# 1. Ayarları ve Klasörleri Hazırla
base_path = Path(__file__).parent
load_dotenv(base_path / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

# SENİN LİSTENDEN EN GARANTİ MODEL: gemini-flash-latest
MODEL_NAME = "gemini-flash-latest" 
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

images_dir = base_path / "images"
gardrop_path = base_path / "gardrop.json"

PROMPT = (
    "Sen bir moda asistanısın. Bu fotoğraftaki kıyafeti incele. "
    "Kategori olarak SADECE şunlardan birini seç: ayakkabi, pantolon_kot, gomlek_tshirt, polar_ceket. "
    "Yanıtını SADECE şu JSON formatında ver: "
    '{"kategori": "kategori_adi", "aciklama": "kıyafetin kısa tanımı"}'
)

def get_mime_type(file_path):
    ext = file_path.suffix.lower()
    return 'image/png' if ext == '.png' else 'image/jpeg'

def process_all_images():
    gardrop = {"ayakkabi": [], "pantolon_kot": [], "gomlek_tshirt": [], "polar_ceket": []}
    image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

    if not image_files:
        print("❌ images klasöründe resim bulunamadı!")
        return

    print(f"🚀 İşlem başlıyor... Model: {MODEL_NAME}")

    for img_path in image_files:
        print(f"🔄 {img_path.name} inceleniyor...")
        
        with open(img_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": PROMPT},
                    {"inline_data": {"mime_type": get_mime_type(img_path), "data": base64_image}}
                ]
            }],
            "generationConfig": {"response_mime_type": "application/json"}
        }

        try:
            response = requests.post(URL, json=payload, headers={'Content-Type': 'application/json'})
            
            if response.status_code == 429:
                print("⏳ Kota doldu, 15 saniye bekleniyor...")
                time.sleep(15) # 429 alırsak biraz daha uzun bekleyelim
                continue

            if response.status_code != 200:
                print(f"❌ {img_path.name} Hatası ({response.status_code}): {response.text}")
                continue

            res_data = response.json()
            text_response = res_data['candidates'][0]['content']['parts'][0]['text']
            
            # JSON temizliği
            clean_json = text_response.replace('```json', '').replace('```', '').strip()
            result = json.loads(clean_json)
            
            kat = result.get("kategori")
            if kat in gardrop:
                gardrop[kat].append({"dosya": img_path.name, "aciklama": result.get("aciklama")})
                print(f"✅ Eklendi: {result.get('aciklama')}")
            
            # Her resimden sonra 2 saniye nefes al (Rate Limit koruması)
            time.sleep(2)

        except Exception as e:
            print(f"❌ {img_path.name} işlenirken hata: {e}")

    # Sonuçları Kaydet
    with open(gardrop_path, "w", encoding="utf-8") as f:
        json.dump(gardrop, f, ensure_ascii=False, indent=2)
    
    print("\n✨ BAŞARILI! gardrop.json dosyasını kontrol et, kıyafetlerin orada!")

if __name__ == "__main__":
    process_all_images()