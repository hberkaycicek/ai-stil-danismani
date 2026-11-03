CONFIG = {
    # Hedef
    "HEDEF_TUR": "TUTARSAL",
    "HEDEF_SON": "90_12",

    # Bölme (zaman bazlı)
    "GELISTIRME_ARALIK": (202101, 202409),    # eğitim + walk-forward CV + model seçimi
    "OOT_ARALIK": (202410, 202509),           # seçimde hiç kullanılmaz, sadece ölçüm

    # Olgunlaşma: M kohortunun 12. ay verisi M+12 ayının ilk haftasında gelir (202509 -> 202610)
    "BILGI_AYI": "2026-10",
    "HEDEF_OLGUNLASMA_AY": 12,
    "RAPOR_GECIKME_AY": 0,

    # Walk-forward CV (her bilgi ayı için ayrı fold seti, olgunlaşma boşluklu)
    "N_SPLITS": 5, "TEST_SIZE": 4,
    "MIN_TRAIN": 9,                           # AGORT6 + 2. ayda Fold1 eğitimi 9 kohort (Bölüm 7)
    "IC_CV_SPLITS": 3,

    # Aday havuzu
    "CHAMPION_MAX_AY": 6, "CHALLENGER_MAX_AY": 3,
    "MAX_K": 2, "AYNI_AYDAN_TEK_ONCUL": True,
    "MODEL_AILELERI": ["Linear", "Ridge", "Lasso", "ElasticNet"],
    "ALPHALAR": np.logspace(-6, 3, 61),
    "L1_ORANLARI": [0.1, 0.3, 0.5, 0.7, 0.9],  # 1.0 = Lasso (ayrı aile)
    "MAX_ITER": 100000, "MODEL_TOL": 1e-6,

    # Takvim
    "TREND_DENE": True,                       # rapor için denenir
    "TREND_SECILEBILIR": False,               # DÜZELTME: trend geleceğe doğrusal uzanır -> seçilemez
    "TREND_ORIJIN": "2021-01",
    "MEVSIM_DENE": True,
    "YON_TOLERANSI": 0.0005,

    # Geçmiş kohort ortalamaları (limit ağırlıklı, sadece önceki kohortlar)
    "ORTALAMA_DENE": True,
    "ORTALAMA_PENCERELERI": [3, 6],
    "MAX_ORTALAMA": 1,
    "ORTALAMA_TAM_PENCERE": True,
    "ORTALAMA_GECIKME": 1,
    "ORTALAMA_AGIRLIK_KOLON": "TOPLAM_ONAYLANAN_LIMIT",
    "LIMIT_KOLON": "TOPLAM_ONAYLANAN_LIMIT",
    "LIMIT_PAYDA_KOLON": "BASVURU_ADEDI",
    "TUFE_KOLON": None, "SAMPLE_WEIGHT_KOLON": None,

    # Sıralama ve geçerlilik (naif ortalama modeline göre)
    "AGIRLIK": {"CV_WMAPE": 0.30, "EN_KOTU_WMAPE": 0.12, "FOLD_WMAPE_STD": 0.12,
                "OVERFIT_ORANI": 0.15, "DELTA_MAE": 0.12, "MUTLAK_FOLD_SAPMA": 0.19},
    "GECERLI": {"CV_WMAPE_DUMMY_ORAN": 0.50, "EN_KOTU_DUMMY_ORAN": 0.75,
                "OVERFIT_ORANI_MAX": 2.0, "SAPMA_DUMMY_ORAN": 0.35},

    # Seçim: anlamlılık (HAC, sadece geliştirme verisi) + 1-SE + sadelik
    "HAC_P_ESIK": 0.10,
    "HAC_ILK_N": 1000,
    "CHALLENGER_FARKLI_GIRDI": True,
    "REFERANS": {"girdiler": ["30_2", "KKB"], "aile": "Linear"},   # v4 champion, kıyas için

    # Optuna (sağlamlık kontrolü; doğrusal olmayan aileler raporlanır, seçilemez)
    "OPTUNA_CALISTIR": True, "OPTUNA_TRIALS": 60, "OPTUNA_SET_SAYISI": 20,
    "FORMUL_AILELERI": ["Linear", "Ridge", "Lasso", "ElasticNet", "Ridge*", "Lasso*",
                        "ElasticNet*", "Huber"],

    # Seviye düzeltmesi (overlay)
    "OVERLAY_PENCERE": 12, "OVERLAY_MIN_AYNI_ISARET": 10,
}

KKB_KOL = next((c for c in ["KKB_SKOR_O0120", "KKB_SKOR_00120"] if c in df_raw.columns), "KKB_SKOR_O0120")
ORTAK_ALANLAR = [KKB_KOL, "MM_SKOR", "KALIBRE_PD_AGIRLIKLI", "ORTALAMA_LIMIT"]
ONCU_KODLARI = ["20_2", "30_2", "30_3", "60_3", "30_4", "60_4", "30_5", "60_5", "30_6", "60_6"]
MEVSIM_KOL, TREND_KOL = ["SIN_AY", "COS_AY"], "TREND_IDX"
TAKVIM_KOL = MEVSIM_KOL + [TREND_KOL]

oncul_kolon = lambda kod, tur=None: f"KK_TAKIP_{kod}_{tur or CONFIG['HEDEF_TUR']}"
oncul_ay = lambda kod: int(kod.split("_")[1])
kisa_ad = lambda k: k.replace("KK_TAKIP_", "").replace(f"_{CONFIG['HEDEF_TUR']}", "")
taban = lambda kolonlar: [c for c in kolonlar if c not in TAKVIM_KOL]
if not np.isclose(sum(CONFIG["AGIRLIK"].values()), 1):
    raise ValueError("Sıralama ağırlıkları toplamı 1 olmalı.")

display(pd.DataFrame([
    {"Başlık": "Geliştirme", "Değer": f"{CONFIG['GELISTIRME_ARALIK'][0]}–{CONFIG['GELISTIRME_ARALIK'][1]}"},
    {"Başlık": "Bağımsız OOT", "Değer": f"{CONFIG['OOT_ARALIK'][0]}–{CONFIG['OOT_ARALIK'][1]}"},
    {"Başlık": "Bilgi ayı (bugün)", "Değer": CONFIG["BILGI_AYI"]},
    {"Başlık": "Dış / iç CV", "Değer": f"{CONFIG['N_SPLITS']} × {CONFIG['TEST_SIZE']} ay, bilgi ayına göre "
                                      f"olgunlaşma boşluklu / {CONFIG['IC_CV_SPLITS']} zamansal fold"},
    {"Başlık": "Champion / Challenger", "Değer": f"En fazla {CONFIG['CHAMPION_MAX_AY']} / "
                                                 f"{CONFIG['CHALLENGER_MAX_AY']} aylık öncül"},
    {"Başlık": "Aday yapısı", "Değer": f"1–{CONFIG['MAX_K']} ham girdi + en fazla {CONFIG['MAX_ORTALAMA']} "
                                       f"geçmiş ortalama + takvim"},
    {"Başlık": "Seçim", "Değer": f"HAC p ≤ {CONFIG['HAC_P_ESIK']} + 1-SE + sadelik; trend seçilemez"},
    {"Başlık": "KKB kolonu", "Değer": KKB_KOL},
]).style.hide(axis="index"))
