# %% [1] Kütüphaneler ve veri (USD limit)
# L12M90 v9 – Dataiku. Kaynak: kur kolonları eklenmiş kohort tablosu (df). Olgun VE olgunlaşmamış kohortlar
# aynı tabloda kalır: hedefi henüz belli olmayanlar (202510+) modele girmez, Bölüm 15'te tahmin edilir.
# Tüm limit tutarları USD'ye çevrilir: TOPLAM_ONAYLANAN_LIMIT_USD = TL limit / efektif satış kuru (kohort ayı).
# USD limit hem geçmiş ortalamaların ağırlığı hem de ORTALAMA_LIMIT değişkeni için kullanılır.
import time
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import statsmodels.api as sm
from IPython.display import display
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, enet_path)
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 250)

DATASET_ADI = "BDS_TAHMIN_ALL"
ILK_KOHORT = 202101
LIMIT_TL, KUR_KOL, KUR_TARIH_KOL = "TOPLAM_ONAYLANAN_LIMIT", "EFEKTIF_SATIS", "KUR_TARIHI"
LIMIT_USD = "TOPLAM_ONAYLANAN_LIMIT_USD"

if "df" not in globals():
    try:
        import dataiku
        df = dataiku.Dataset(DATASET_ADI).get_dataframe()
    except ImportError:
        raise RuntimeError("Önce kur kolonları eklenmiş df tablosunu oluştur.")
kaynak = df.copy()
if kaynak.empty or kaynak.columns.duplicated().any():
    raise ValueError("Kaynak boş veya kolon adları tekrarlanıyor.")
eksik = sorted({"DONEM_1", LIMIT_TL, KUR_KOL, KUR_TARIH_KOL} - set(kaynak.columns))
if eksik:
    raise ValueError(f"Eksik kolonlar: {eksik}")

donem = pd.to_numeric(kaynak["DONEM_1"], errors="coerce")
if not np.isfinite(donem).all() or (donem % 1 != 0).any():
    raise ValueError("DONEM_1 sonlu YYYYMM tam sayıları olmalı.")
secim = donem >= ILK_KOHORT
if "DONEM" in kaynak.columns:                                  # kaynakta DONEM boş satırlar (veri yok) atılır
    secim &= kaynak["DONEM"].notna()
df_raw = kaynak.loc[secim].copy()
df_raw["DONEM_1"] = donem.loc[df_raw.index].astype(int)
df_raw = df_raw.sort_values("DONEM_1").reset_index(drop=True)
if df_raw.empty:
    raise ValueError(f"{ILK_KOHORT} sonrası kohort bulunamadı.")

# --- USD dönüşümü (tüm kohortlar: olgun + yeni)
limit_tl = pd.to_numeric(df_raw[LIMIT_TL], errors="raise")
kur = pd.to_numeric(df_raw[KUR_KOL], errors="raise")
if not np.isfinite(limit_tl).all() or (limit_tl < 0).any():
    raise ValueError("TL limitleri sonlu ve negatif olmayan değerler olmalı.")
if not np.isfinite(kur).all() or (kur <= 0).any():
    raise ValueError(f"{KUR_KOL} sonlu ve pozitif olmalı: {df_raw.loc[~(np.isfinite(kur) & (kur > 0)), 'DONEM_1'].tolist()}")
kohort_ayi = pd.to_datetime(df_raw["DONEM_1"].astype(str), format="%Y%m").dt.to_period("M")
kur_ayi = pd.to_datetime(df_raw[KUR_TARIH_KOL], errors="raise").dt.to_period("M")
uyumsuz = kur_ayi.isna() | kur_ayi.ne(kohort_ayi)
if uyumsuz.any():
    raise ValueError(f"Kur ayı kohort ayıyla eşleşmiyor: {df_raw.loc[uyumsuz, 'DONEM_1'].tolist()}")
df_raw[LIMIT_TL] = limit_tl
df_raw[LIMIT_USD] = limit_tl / kur
if not np.isfinite(df_raw[LIMIT_USD]).all():
    raise ValueError("USD limitlerinde geçersiz değer oluştu.")

RENK = {"GERCEK": "#111827", "EGITIM": "#94a3b8", "VALIDASYON": "#2563eb", "OOT": "#ea580c",
        "TAHMIN": "#16a34a", "KISMI": "#6b7280", "CHALLENGER": "#7c3aed", "REFERANS": "#0891b2",
        "DUMMY": "#cbd5e1"}
ARKA = {"EGITIM": "#f1f5f9", "VALIDASYON": "#eff6ff", "OOT": "#fff7ed", "TAHMIN": "#f0fdf4"}
yuzde = lambda v, d=2: "-" if pd.isna(v) else f"%{v * 100:.{d}f}".replace(".", ",")
sayi = lambda v, d=2: "-" if pd.isna(v) else f"{v:.{d}f}".replace(".", ",")
PCT_AX = FuncFormatter(lambda x, _: f"%{x:.2f}".replace(".", ","))

print(f"Kaynak: {DATASET_ADI} | Satır: {len(df_raw)} | Kolon: {df_raw.shape[1]} | "
      f"Dönem: {df_raw['DONEM_1'].min()} – {df_raw['DONEM_1'].max()}")
_kur = df_raw[["DONEM_1", KUR_KOL, LIMIT_TL, LIMIT_USD]].iloc[[0, len(df_raw) // 2, -1]]
display(_kur.style.format({KUR_KOL: "{:.4f}", LIMIT_TL: "{:,.0f}", LIMIT_USD: "{:,.0f}"}).hide(axis="index"))


# %% [2] Konfigürasyon
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

    # Aday havuzu
    "CHAMPION_MAX_AY": 6, "CHALLENGER_MAX_AY": 3,
    "MAX_K": 2, "AYNI_AYDAN_TEK_ONCUL": True,
    "MODEL_AILELERI": ["Linear", "Ridge", "Lasso", "ElasticNet"],
    # Ceza katsayısı (alpha) DIŞ walk-forward CV'de seçilir: her alpha ayrı aday, aynı boşluklu fold'lar.
    # (İç CV'ye olgunlaşma boşluğu konunca ilk fold'larda 0-6 eğitim kohortu kalıyor; uygulanamaz.)
    # Ölçek: X standart, y ham (~0,015). Ridge cezası X ölçeğinde, Lasso/EN cezası hedef ölçeğinde.
    "ALPHA_RIDGE": np.logspace(-2, 2, 9),
    "ALPHA_LASSO": np.logspace(-6, -2.5, 8),
    "L1_ORANLARI": [0.2, 0.5, 0.8],           # ElasticNet (1.0 = Lasso, ayrı aile)
    "MAX_ITER": 100000, "MODEL_TOL": 1e-8,

    # Takvim
    "TREND_DENE": True,                       # rapor için denenir
    "TREND_SECILEBILIR": False,               # DÜZELTME: trend geleceğe doğrusal uzanır -> seçilemez
    "TREND_ORIJIN": "2021-01",
    "MEVSIM_DENE": True,
    "YON_TOLERANSI": 0.0005,

    # Geçmiş kohort ortalamaları (USD limit ağırlıklı, sadece önceki kohortlar)
    "ORTALAMA_DENE": True,
    "ORTALAMA_PENCERELERI": [3, 6],
    "MAX_ORTALAMA": 1,
    "ORTALAMA_TAM_PENCERE": True,
    "ORTALAMA_GECIKME": 1,
    # USD: TL limit kur/enflasyonla şişer ve yeni kohortlara fazla ağırlık verir; USD bu etkiyi temizler.
    "ORTALAMA_AGIRLIK_KOLON": LIMIT_USD,      # geçmiş ortalamaların ağırlığı
    "LIMIT_KOLON": LIMIT_USD,                 # ORTALAMA_LIMIT = USD limit / başvuru adedi
    "LIMIT_PAYDA_KOLON": "BASVURU_ADEDI",
    "TUFE_KOLON": None, "SAMPLE_WEIGHT_KOLON": None,
    # İstersen hacim değişkenleri de aday olur (ör. ["BASVURU_ADEDI"]). Nominal tutarlar enflasyonla
    # trend gibi davranır; eklenirse TUFE ile reelleştir.
    "EK_BASVURU_ALANLARI": [],

    # Sıralama ve geçerlilik (naif ortalama modeline göre)
    "AGIRLIK": {"CV_WMAPE": 0.30, "EN_KOTU_WMAPE": 0.12, "FOLD_WMAPE_STD": 0.12,
                "OVERFIT_ORANI": 0.15, "DELTA_MAE": 0.12, "MUTLAK_FOLD_SAPMA": 0.19},
    "GECERLI": {"CV_WMAPE_DUMMY_ORAN": 0.50, "EN_KOTU_DUMMY_ORAN": 0.75,
                "OVERFIT_ORANI_MAX": 2.0, "SAPMA_DUMMY_ORAN": 0.35},

    # Seçim: anlamlılık (HAC, sadece geliştirme verisi) + 1-SE + sadelik
    "HAC_P_ESIK": 0.10,
    "HAC_ILK_N": 1000,
    "CHALLENGER_FARKLI_GIRDI": True,

    # Optuna (sağlamlık kontrolü; doğrusal olmayan aileler raporlanır, seçilemez)
    "OPTUNA_CALISTIR": True, "OPTUNA_TRIALS": 60, "OPTUNA_SET_SAYISI": 20,
    "FORMUL_AILELERI": ["Linear", "Ridge", "Lasso", "ElasticNet", "Ridge*", "Lasso*",
                        "ElasticNet*", "Huber"],

    # Seviye düzeltmesi (overlay)
    "OVERLAY_PENCERE": 12, "OVERLAY_MIN_AYNI_ISARET": 10,
}

KKB_KOL = next((c for c in ["KKB_SKOR_O0120", "KKB_SKOR_00120"] if c in df_raw.columns), "KKB_SKOR_O0120")
ORTAK_ALANLAR = [KKB_KOL, "MM_SKOR", "KALIBRE_PD_AGIRLIKLI", "ORTALAMA_LIMIT"] + list(CONFIG["EK_BASVURU_ALANLARI"])
ONCU_KODLARI = ["20_2", "30_2", "30_3", "60_3", "30_4", "60_4", "30_5", "60_5", "30_6", "60_6"]
MEVSIM_KOL, TREND_KOL = ["SIN_AY", "COS_AY"], "TREND_IDX"
TAKVIM_KOL = MEVSIM_KOL + [TREND_KOL]

oncul_kolon = lambda kod, tur=None: f"KK_TAKIP_{kod}_{tur or CONFIG['HEDEF_TUR']}"
oncul_ay = lambda kod: int(kod.split("_")[1])
kisa_ad = lambda k: k.replace("KK_TAKIP_", "").replace(f"_{CONFIG['HEDEF_TUR']}", "")
taban = lambda kolonlar: [c for c in kolonlar if c not in TAKVIM_KOL]


def tahmin_ayi_hesapla(kohort_ayi, bilgi_ayi):
    """Kohortun Y. ay verisi, Y. ayın bittiği ayın BİR SONRAKİ ayının ilk haftasında gelir.
    Tek erişim takvimi: öncül, PIT testi, CV fold'ları, OOT ve üretim hep bunu kullanır.
    Bir verinin bu anda bilinmesi için: erişim_ayı (= kohort + Y + gecikme) < tahmin_ayı."""
    return kohort_ayi + int(bilgi_ayi) + CONFIG["RAPOR_GECIKME_AY"] + 1
if not np.isclose(sum(CONFIG["AGIRLIK"].values()), 1):
    raise ValueError("Sıralama ağırlıkları toplamı 1 olmalı.")

display(pd.DataFrame([
    {"Başlık": "Geliştirme", "Değer": f"{CONFIG['GELISTIRME_ARALIK'][0]}–{CONFIG['GELISTIRME_ARALIK'][1]}"},
    {"Başlık": "Bağımsız OOT", "Değer": f"{CONFIG['OOT_ARALIK'][0]}–{CONFIG['OOT_ARALIK'][1]}"},
    {"Başlık": "Bilgi ayı (bugün)", "Değer": CONFIG["BILGI_AYI"]},
    {"Başlık": "Walk-forward CV", "Değer": f"{CONFIG['N_SPLITS']} × {CONFIG['TEST_SIZE']} ay, bilgi ayına göre "
                                          f"olgunlaşma boşluklu; alpha da bu CV'de seçilir"},
    {"Başlık": "Champion / Challenger", "Değer": f"En fazla {CONFIG['CHAMPION_MAX_AY']} / "
                                                 f"{CONFIG['CHALLENGER_MAX_AY']} aylık öncül"},
    {"Başlık": "Aday yapısı", "Değer": f"1–{CONFIG['MAX_K']} ham girdi + en fazla {CONFIG['MAX_ORTALAMA']} "
                                       f"geçmiş ortalama + takvim"},
    {"Başlık": "Seçim", "Değer": f"HAC p ≤ {CONFIG['HAC_P_ESIK']} + 1-SE + sadelik; trend seçilemez"},
    {"Başlık": "KKB kolonu", "Değer": KKB_KOL},
    {"Başlık": "Limit / ağırlık", "Değer": f"USD ({LIMIT_TL} / {KUR_KOL}); ortalama ağırlığı ve ORTALAMA_LIMIT"},
]).style.hide(axis="index"))


# %% [3] Veri hazırlama, olgunlaşma ve bölme
def veri_hazirla(df_raw, config):
    tur = config["HEDEF_TUR"]
    hedef = oncul_kolon(config["HEDEF_SON"], tur)
    onculler = [oncul_kolon(k, tur) for k in ONCU_KODLARI]
    k90 = sorted([c for c in df_raw.columns if c.startswith("KK_TAKIP_90_") and c.endswith(f"_{tur}")
                  and c.split("_")[3].isdigit() and int(c.split("_")[3]) < 12],
                 key=lambda c: int(c.split("_")[3]))
    ham_ortak = [c for c in ORTAK_ALANLAR if c != "ORTALAMA_LIMIT"]
    ek = [config[k] for k in ["LIMIT_KOLON", "LIMIT_PAYDA_KOLON", "ORTALAMA_AGIRLIK_KOLON",
                              "SAMPLE_WEIGHT_KOLON", "TUFE_KOLON"] if config[k]]
    gerekli = list(dict.fromkeys(["DONEM_1", hedef, *onculler, *ham_ortak, *ek, *k90]))
    eksik = sorted(set(gerekli) - set(df_raw.columns))
    if eksik:
        raise ValueError(f"Eksik kolon: {eksik}")
    if any("_90_" in k for k in onculler + ORTAK_ALANLAR):
        raise ValueError("90+ alanı girdi olamaz.")

    d = df_raw[gerekli].copy()
    d["AY"] = pd.to_datetime(d["DONEM_1"].astype(str), format="%Y%m").dt.to_period("M")
    d = d.sort_values("AY").reset_index(drop=True)
    if d["AY"].duplicated().any():
        raise ValueError("Tekrarlanan kohort var.")
    tam = pd.period_range(d["AY"].min(), d["AY"].max(), freq="M")
    if len(tam) != len(d):
        raise ValueError(f"Eksik ay: {sorted(set(tam.astype(str)) - set(d['AY'].astype(str)))}")
    sayisal = [c for c in gerekli if c != "DONEM_1"]
    d[sayisal] = d[sayisal].apply(pd.to_numeric, errors="raise")

    payda = d[config["LIMIT_PAYDA_KOLON"]].where(d[config["LIMIT_PAYDA_KOLON"]] > 0)
    d["ORTALAMA_LIMIT"] = d[config["LIMIT_KOLON"]] / payda
    if config["TUFE_KOLON"]:
        d["ORTALAMA_LIMIT"] = d["ORTALAMA_LIMIT"] / d[config["TUFE_KOLON"]] * 100

    orijin = pd.Period(config["TREND_ORIJIN"], freq="M")
    d[TREND_KOL] = d["AY"].apply(lambda p: (p - orijin).n).astype(float)
    d["SIN_AY"] = np.sin(2 * np.pi * d["AY"].dt.month / 12)
    d["COS_AY"] = np.cos(2 * np.pi * d["AY"].dt.month / 12)

    bilgi = pd.Period(config["BILGI_AYI"], freq="M")
    gec = config["RAPOR_GECIKME_AY"]
    d["HEDEF_ERISIM"] = d["AY"] + config["HEDEF_OLGUNLASMA_AY"] + gec       # hedefin belli olduğu ay
    gecerli_hedef = d[hedef].notna() & np.isfinite(d[hedef]) & d[hedef].between(0, 1)
    d["HEDEF_KULLANILABILIR"] = (d["HEDEF_ERISIM"] < bilgi) & gecerli_hedef
    d["GOZLENEN_AY"] = d["AY"].apply(lambda p: (bilgi - p).n) - 1 - gec

    kismi = {}
    for k, kol in zip(ONCU_KODLARI, onculler):                               # bugün tamamlanmamış öncül -> NaN
        tam_mi = (d["AY"] + oncul_ay(k) + gec) < bilgi
        if (~tam_mi & d[kol].notna()).any():
            kismi[kisa_ad(kol)] = int((~tam_mi & d[kol].notna()).sum())
        d.loc[~tam_mi, kol] = np.nan
    for kol in k90:                                                         # bugün tamamlanmamış 90_m -> NaN
        m_ = int(kol.split("_")[3])                                         # (chain-ladder için, GİRDİ DEĞİL)
        d.loc[(d["AY"] + m_ + gec) >= bilgi, kol] = np.nan
    if kismi:
        print("Kısmi (tamamlanmamış) öncül değeri NaN yapıldı:", kismi)
    if ((d[onculler] < 0) | (d[onculler] > 1)).sum().sum():
        raise ValueError("0-1 dışında oran girdisi var.")
    na = d.loc[d["HEDEF_KULLANILABILIR"], onculler + ORTAK_ALANLAR].isna().sum()
    if na.any():
        print("UYARI - olgun kohortlarda boş girdi:\n", na[na > 0].to_string())
    return d, hedef, onculler, k90


def bolme_uygula(d, config):
    kullan = d[d["HEDEF_KULLANILABILIR"]]
    gel = kullan.index[kullan["DONEM_1"].between(*config["GELISTIRME_ARALIK"])].to_numpy()
    oot = kullan.index[kullan["DONEM_1"].between(*config["OOT_ARALIK"])].to_numpy()
    disarida = kullan.index.difference(np.concatenate([gel, oot]))
    if len(disarida):
        print("UYARI - bölgeye atanmamış olgun kohort:", list(kullan.loc[disarida, "DONEM_1"]))
    if len(gel) == 0 or len(oot) == 0 or len(set(gel) & set(oot)):
        raise ValueError("Geliştirme/OOT bölmesi hatalı.")
    if d.loc[gel, "AY"].max() >= d.loc[oot, "AY"].min():
        raise ValueError("OOT geliştirmeden sonra başlamalı.")
    return gel, oot


d, HEDEF, ONCULLER, K90 = veri_hazirla(df_raw, CONFIG)
gel_idx, oot_idx = bolme_uygula(d, CONFIG)
HK = kisa_ad(HEDEF)
BILGI = pd.Period(CONFIG["BILGI_AYI"], "M")
d["BOLGE"] = np.select([d.index.isin(gel_idx), d.index.isin(oot_idx), ~d["HEDEF_KULLANILABILIR"]],
                       ["GELISTIRME", "OOT", "TAHMIN"], "DISARIDA")
VERI_OZET = (d.groupby("BOLGE", sort=False)
              .agg(N=("AY", "size"), BAS=("AY", "min"), SON=("AY", "max"), HEDEF_ORT=(HEDEF, "mean"))
              .reindex(["GELISTIRME", "OOT", "TAHMIN"]))
print(f"Hedef: {HEDEF} | Bilgi ayı: {BILGI} | 90+ olgunluk kolonları: {[kisa_ad(k) for k in K90] or 'yok'}")
display(VERI_OZET.style.format({"HEDEF_ORT": yuzde}, na_rep="-"))
if not (d["BOLGE"] == "TAHMIN").any():
    print("Bilgi: veride olgunlaşmamış kohort yok -> Bölüm 15'te gelecek tahmini üretilmez.")


# %% [4] Geçmiş ortalamalar ve aday havuzu (point-in-time)
# Kural: kohort c'nin her girdisi, c'nin tahmin anında sistemde olan veriyle üretilir.
#   - X_Y öncülü, kohort j için ancak j + Y < T ise bilinir (T = verinin geldiği ay)
#   - Ortalama: sadece ÖNCEKİ kohortlar (shift(1)), USD limit ağırlıklı, tam pencere. Lead YASAK.
HAM_HAVUZ = ONCULLER + ORTAK_ALANLAR
KOLON_AYI = {c: 0 for c in ORTAK_ALANLAR + TAKVIM_KOL}
KOLON_AYI.update({oncul_kolon(k): oncul_ay(k) for k in ONCU_KODLARI})


def ortalama_uret(df, T):
    """T ayının başında bilinen veriyle geçmiş kohort ortalamaları. Dönüş: DataFrame."""
    x = df.sort_values("AY").copy()
    for k in ONCU_KODLARI:                                                   # T'de tamamlanmamış öncül -> NaN
        x.loc[x["AY"] + oncul_ay(k) + CONFIG["RAPOR_GECIKME_AY"] >= T, oncul_kolon(k)] = np.nan
    w = x[CONFIG["ORTALAMA_AGIRLIK_KOLON"]].astype(float)
    g = CONFIG["ORTALAMA_GECIKME"]
    out = {}
    for kol in HAM_HAVUZ:
        s = x[kol].astype(float)
        ws = w.where(s.notna())                                              # boş değerin ağırlığı da sayılmaz
        for p in CONFIG["ORTALAMA_PENCERELERI"]:
            mp = p if CONFIG["ORTALAMA_TAM_PENCERE"] else 1
            pay = (s * ws).shift(g).rolling(p, min_periods=mp).sum()
            payda = ws.shift(g).rolling(p, min_periods=mp).sum()
            out[f"{kol}_AGORT{p}"] = pay / payda.where(payda > 0)
    return pd.DataFrame(out, index=x.index)


ORTALAMA_KOL, ORTALAMA_KAYNAK, ORTALAMA_PENCERE = [], {}, {}
if CONFIG["ORTALAMA_DENE"]:
    g_, pencereler = CONFIG["ORTALAMA_GECIKME"], CONFIG["ORTALAMA_PENCERELERI"]
    if not isinstance(g_, int) or g_ < 1:
        raise ValueError("Ortalama gecikmesi pozitif tam sayı olmalı (shift(0) kohortun kendisini içerir).")
    ORT_DF = ortalama_uret(d, BILGI)
    d = d.drop(columns=[c for c in ORT_DF.columns if c in d.columns]).join(ORT_DF)
    for kol in HAM_HAVUZ:
        for p in pencereler:
            ad = f"{kol}_AGORT{p}"
            ORTALAMA_KOL.append(ad)
            ORTALAMA_KAYNAK[ad], ORTALAMA_PENCERE[ad] = kol, p
            KOLON_AYI[ad] = KOLON_AYI[kol]

    # Point-in-time testi: her kohort için ortalamayı, gelecek satırlar silinmiş veriyle ve kendi
    # tahmin anında yeniden üret; pipeline'daki değerle aynı değilse sızıntı vardır.
    pit = []
    for i in d.index:
        c_ay = d.loc[i, "AY"]
        gecmis = d[d["AY"] <= c_ay]
        for Y in sorted({KOLON_AYI[a] for a in ORTALAMA_KOL}):
            yeniden = ortalama_uret(gecmis, tahmin_ayi_hesapla(c_ay, Y))      # Y. ay verisinin geldiği ay
            for a in [a for a in ORTALAMA_KOL if KOLON_AYI[a] == Y]:
                v1, v2 = d.loc[i, a], yeniden.loc[i, a]
                if not ((pd.isna(v1) and pd.isna(v2)) or (pd.notna(v1) and pd.notna(v2) and np.isclose(v1, v2))):
                    pit.append({"KOHORT": str(c_ay), "DEGISKEN": kisa_ad(a), "PIPELINE": v1, "O_GUN": v2})
    print(f"Point-in-time testi: {len(pit)} uyumsuzluk", "-> sızıntı yok" if not pit else "-> KONTROL ET")
    if pit:
        display(pd.DataFrame(pit).head(20))

HAVUZ = HAM_HAVUZ + ORTALAMA_KOL + TAKVIM_KOL
if len(HAVUZ) != len(set(HAVUZ)) or any("_90_" in c or c == HEDEF for c in HAVUZ):
    raise ValueError("Tekrarlanan veya yasak model girdisi var.")
bilgi_ayi_hesapla = lambda kolonlar: max((KOLON_AYI[c] for c in kolonlar), default=0)


def kombinasyon_uygun(kolonlar):
    aylar = [KOLON_AYI[c] for c in kolonlar if c in ONCULLER]
    return not CONFIG["AYNI_AYDAN_TEK_ONCUL"] or len(aylar) == len(set(aylar))


TEMEL_KOMBS = [alt for adet in range(1, CONFIG["MAX_K"] + 1)
               for alt in combinations(HAM_HAVUZ, adet) if kombinasyon_uygun(alt)]
VARYANTLAR = [trend + mevsim
              for trend in ([(), (TREND_KOL,)] if CONFIG["TREND_DENE"] else [()])
              for mevsim in ([(), tuple(MEVSIM_KOL)] if CONFIG["MEVSIM_DENE"] else [()])]


def ortalama_ekleri(temel):
    uygun = [k for k in ORTALAMA_KOL if ORTALAMA_KAYNAK[k] in temel]       # sadece modeldeki girdinin ortalaması
    return [()] + [(k,) for k in uygun]


KOMBS = [temel + ort + takvim
         for temel in TEMEL_KOMBS for ort in ortalama_ekleri(temel) for takvim in VARYANTLAR
         if bilgi_ayi_hesapla(temel + ort) <= CONFIG["CHAMPION_MAX_AY"]]
ADAY_TABLO = pd.DataFrame([{
    "ADAY_ID": n, "GIRDILER": ", ".join(map(kisa_ad, taban(k))),
    "TEMEL_K": sum(c in HAM_HAVUZ for c in k), "ORTALAMA_K": sum(c in ORTALAMA_KOL for c in k),
    "TOPLAM_K": len(k), "BILGI_AY": bilgi_ayi_hesapla(k),
    "TREND": int(TREND_KOL in k), "MEVSIM": int("SIN_AY" in k),
    "ORTALAMA": next((f"AGORT{ORTALAMA_PENCERE[c]}" for c in k if c in ORTALAMA_KOL), "Yok"),
    "SECILEBILIR": TREND_KOL not in k or CONFIG["TREND_SECILEBILIR"],
    "CHALLENGER": bilgi_ayi_hesapla(k) <= CONFIG["CHALLENGER_MAX_AY"],
    "KOLONLAR": tuple(k)} for n, k in enumerate(KOMBS, 1)])
print(f"Havuz: {len(HAM_HAVUZ)} ham + {len(ORTALAMA_KOL)} ortalama + takvim | aday: {len(ADAY_TABLO):,}")
display(pd.DataFrame([{"Model": "CHAMPION", "Aday": len(ADAY_TABLO)},
                      {"Model": "CHALLENGER", "Aday": int(ADAY_TABLO["CHALLENGER"].sum())}])
        .style.hide(axis="index").format({"Aday": "{:,.0f}"}))
print("Bilgi ayı dağılımı:", ADAY_TABLO["BILGI_AY"].value_counts().sort_index().to_dict())


# %% [5] Walk-forward CV ve metrik motoru (olgunlaşma boşluklu)
# Her bilgi ayı için ayrı fold seti. Bloğun ilk kohortu tahmin_ayi_hesapla(AY, b) anında tahmin edilir;
# eğitimde sadece hedefi o anda BELLİ olan kohortlar (AY + 12 + gecikme < tahmin anı) kalır.
# Boşluk = 11 − bilgi ayı: 2. ay modelinde 9, 3. ayda 8, 6. ayda 5 kohort eğitimden çıkar.
MIN_TRAIN = CONFIG["MIN_TRAIN"]
cv_df = d.loc[gel_idx].sort_values("AY").reset_index(drop=True)
Y_CV = cv_df[HEDEF].to_numpy(dtype=float)
AY_CV = pd.PeriodIndex(cv_df["AY"])
AYNO_CV = AY_CV.asi8
TOL = float(CONFIG["YON_TOLERANSI"])


def fold_olustur(bilgi_ayi):
    foldlar, ozet = [], []
    tscv = TimeSeriesSplit(n_splits=CONFIG["N_SPLITS"], test_size=CONFIG["TEST_SIZE"])
    for i, (tr, val) in enumerate(tscv.split(np.arange(len(cv_df))), 1):
        ilk_tahmin = tahmin_ayi_hesapla(AY_CV[val[0]], bilgi_ayi)
        tr = tr[(AY_CV[tr] + CONFIG["HEDEF_OLGUNLASMA_AY"] + CONFIG["RAPOR_GECIKME_AY"]) < ilk_tahmin]
        foldlar.append((tr, val))
        ozet.append({"FOLD": i, "EGITIM_N": len(tr), "EGITIM": f"{AY_CV[tr[0]]} – {AY_CV[tr[-1]]}",
                     "BOSLUK (olgunlaşmamış)": int(val[0] - tr[-1] - 1),
                     "VALIDASYON": f"{AY_CV[val[0]]} – {AY_CV[val[-1]]}", "TAHMIN_ANI": str(ilk_tahmin)})
    return foldlar, pd.DataFrame(ozet)


FOLD_CACHE = {}
def fold_al(b):
    if b not in FOLD_CACHE:
        FOLD_CACHE[b] = fold_olustur(b)[0]
    return FOLD_CACHE[b]


def wmape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.abs(p - y).sum() / max(np.abs(y).sum(), 1e-12) * 100)


def agirlik_al(df):
    kol = CONFIG["SAMPLE_WEIGHT_KOLON"]
    return None if kol is None else df[kol].to_numpy(dtype=float)


def pipe_fit(estimator, X, y, w=None):
    p = make_pipeline(StandardScaler(), clone(estimator))
    return p.fit(X, y) if w is None else p.fit(X, y, **{f"{p.steps[-1][0]}__sample_weight": w})


def fold_hazirla(kolonlar, foldlar=None):
    """Aday kolonlarına göre fold'ları hazırlar (ortalamaların boş olduğu ilk kohortlar eğitimden çıkar)."""
    kolonlar = list(kolonlar)
    if foldlar is None:
        foldlar = fold_al(bilgi_ayi_hesapla(kolonlar))
    X = cv_df[kolonlar].to_numpy(dtype=float)
    dolu = ~np.isnan(X).any(axis=1)
    hazir = []
    for tr, val in foldlar:
        tr = tr[dolu[tr]]
        if len(tr) < MIN_TRAIN:
            raise ValueError(f"eğitim yetersiz ({len(tr)} < {MIN_TRAIN})")
        if not dolu[val].all():
            raise ValueError("validasyonda boş değer")
        hazir.append((tr, val))
    return X, hazir


def metrik_ozet(kayitlar):
    """kayitlar: her fold için (tr, val, p_val, p_tr). Bütün adaylar AYNI fonksiyonla ölçülür."""
    y = Y_CV
    hata, gercek, f_wmape, f_sapma, f_oran = [], [], [], [], []
    d_g, d_t, aralik_disi = [], [], 0
    for tr, val, p, p_tr in kayitlar:
        f_wmape.append(wmape(y[val], p))
        f_sapma.append(float(np.mean(p - y[val])))
        f_oran.append(f_wmape[-1] / max(wmape(y[tr], p_tr), 1e-12))
        aralik_disi += int(((p < 0) | (p > 2 * y[tr].max())).sum())
        bit = np.diff(AYNO_CV[val]) == 1
        d_g.extend(np.diff(y[val])[bit]); d_t.extend(np.diff(p)[bit])
        hata.extend(p - y[val]); gercek.extend(y[val])
    hata, gercek, d_g, d_t = map(np.asarray, (hata, gercek, d_g, d_t))
    anlamli = np.abs(d_g) > TOL
    dogru = np.sign(np.where(np.abs(d_t) <= TOL, 0, d_t))[anlamli] == np.sign(d_g[anlamli])
    buy = np.abs(d_g[anlamli])
    return {
        "CV_WMAPE": float(np.abs(hata).sum() / gercek.sum() * 100),
        "CV_MAE": float(np.mean(np.abs(hata))), "CV_SAPMA": float(np.mean(hata)),
        "MUTLAK_FOLD_SAPMA": float(np.mean(np.abs(f_sapma))),
        "EN_KOTU_WMAPE": float(np.max(f_wmape)), "FOLD_WMAPE_STD": float(np.std(f_wmape)),
        "OVERFIT_ORANI": float(np.median(f_oran)),
        "DELTA_MAE": float(np.mean(np.abs(d_t - d_g))) if len(d_g) else np.nan,
        "YON_AGIRLIKLI": float((buy * dogru).sum() / buy.sum()) if anlamli.any() else np.nan,
        "ARALIK_DISI": aralik_disi, "FOLD1_EGITIM_N": len(kayitlar[0][0]),
    }


def cv_degerlendir(kolonlar, estimator, foldlar=None):
    X, hazir = fold_hazirla(kolonlar, foldlar)
    kayitlar = []
    for tr, val in hazir:
        w = None if CONFIG["SAMPLE_WEIGHT_KOLON"] is None else cv_df.iloc[tr][CONFIG["SAMPLE_WEIGHT_KOLON"]].to_numpy()
        m = pipe_fit(estimator, X[tr], Y_CV[tr], w)
        kayitlar.append((tr, val, m.predict(X[val]), m.predict(X[tr])))
    return metrik_ozet(kayitlar)


DUMMY = cv_degerlendir([HAM_HAVUZ[0]], DummyRegressor(strategy="mean"), fold_al(0))
print(f"Naif referans (geçmiş ortalama): CV_WMAPE {DUMMY['CV_WMAPE']:.2f} | "
      f"EN_KOTU {DUMMY['EN_KOTU_WMAPE']:.2f} | MUTLAK_FOLD_SAPMA {DUMMY['MUTLAK_FOLD_SAPMA']:.4f}")


# %% [6] Eğitim, validasyon ve OOT takvimi
def ts(p):
    return p.to_timestamp() if hasattr(p, "to_timestamp") else pd.Period(p, "M").to_timestamp()


def takvim_grafigi(b, tahmin_araligi=None):
    """Veri bölmesi + bilgi ayı b için walk-forward fold yapısı (taralı alan = olgunlaşma boşluğu)."""
    def span(p0, p1):
        a, c = mdates.date2num(ts(p0)), mdates.date2num(ts(p1 + 1))
        return (a, c - a)
    foldlar = fold_al(b)
    val_bas = AY_CV[foldlar[0][1][0]]
    olgun = d[d["HEDEF_KULLANILABILIR"]]
    ust = [(span(olgun["AY"].min(), val_bas - 1), ARKA["EGITIM"], "Eğitim"),
           (span(val_bas, AY_CV.max()), RENK["VALIDASYON"], "Validasyon"),
           (span(d.loc[oot_idx, "AY"].min(), d.loc[oot_idx, "AY"].max()), RENK["OOT"], "OOT")]
    gelecek = d.loc[d["BOLGE"] == "TAHMIN", "AY"]
    if len(gelecek):
        ust.append((span(gelecek.min(), gelecek.max()), RENK["TAHMIN"], "Tahmin"))
    satir = [("Kohortlar", ust)]
    for i, (tr, val) in enumerate(foldlar, 1):
        parca = [(span(AY_CV[tr[0]], AY_CV[tr[-1]]), "#bfdbfe", None)]
        if val[0] - tr[-1] > 1:
            parca.append((span(AY_CV[tr[-1]] + 1, AY_CV[val[0]] - 1), "#e5e7eb", "hatch"))
        parca.append((span(AY_CV[val[0]], AY_CV[val[-1]]), RENK["VALIDASYON"], None))
        satir.append((f"CV Fold {i}", parca))
    fig, ax = plt.subplots(figsize=(16, 4.2))
    for yy, (ad, parcalar) in enumerate(reversed(satir)):
        for (x, gen), renk, ek in parcalar:
            ax.broken_barh([(x, gen)], (yy - 0.35, 0.7), facecolors=renk, edgecolor="white",
                           hatch="///" if ek == "hatch" else None)
            if isinstance(ek, str) and ek != "hatch":
                ax.text(x + gen / 2, yy, ek, ha="center", va="center", fontsize=9, fontweight="bold",
                        color="white" if renk != ARKA["EGITIM"] else "#334155")
    ax.set_yticks(range(len(satir))); ax.set_yticklabels([s[0] for s in reversed(satir)])
    ax.xaxis_date(); ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); ax.grid(axis="x", alpha=0.3)
    ax.legend(handles=[Patch(color="#bfdbfe", label="Fold eğitimi"),
                       Patch(facecolor="#e5e7eb", hatch="///", label="Boşluk: hedef o tarihte henüz belli değil"),
                       Patch(color=RENK["VALIDASYON"], label="Validasyon"), Patch(color=RENK["OOT"], label="OOT")],
              loc="upper left", fontsize=8, ncol=4, bbox_to_anchor=(0, -0.12))
    ax.set_title(f"Walk-forward yapı | bilgi ayı = {b} (öncül {b}. ayda belli, tahmin anı = kohort + {b} ay)",
                 fontweight="bold", loc="left")
    plt.tight_layout(); plt.show()


for b in [2, 3, 6]:
    print(f"=== Fold yapısı | bilgi ayı = {b} ===")
    display(fold_olustur(b)[1])
takvim_grafigi(2)
takvim_grafigi(6)


# %% [7] Girdi uygunluğu ve veri kapsaması
olgun_maske = d["HEDEF_KULLANILABILIR"]
KAPSAMA = pd.DataFrame([{
    "DEGISKEN": kisa_ad(c), "GOZLEM_AYI": KOLON_AYI[c],
    "TUR": "Ortalama" if c in ORTALAMA_KOL else "Takvim" if c in TAKVIM_KOL
           else "Öncül" if c in ONCULLER else "Başvuru anı",
    "OLGUN_DOLU": f"{int(d.loc[olgun_maske, c].notna().sum())}/{int(olgun_maske.sum())}",
    "ILK_DOLU_KOHORT": str(d.loc[d[c].notna(), "AY"].min()),
    "TAHMIN_DOLU": f"{int(d.loc[d['BOLGE'] == 'TAHMIN', c].notna().sum())}/{int((d['BOLGE'] == 'TAHMIN').sum())}",
} for c in HAVUZ])
display(KAPSAMA)

ilk_dolu = {"Ham": cv_df["AY"].min()}
for p in CONFIG["ORTALAMA_PENCERELERI"] if CONFIG["ORTALAMA_DENE"] else []:
    ilk_dolu[f"AGORT{p}"] = cv_df.loc[cv_df[f"{ONCULLER[1]}_AGORT{p}"].notna(), "AY"].min()
F1 = pd.DataFrame({tur: {b: int((AY_CV[fold_al(b)[0][0]] >= bas).sum()) for b in range(0, CONFIG["CHAMPION_MAX_AY"] + 1)}
                   for tur, bas in ilk_dolu.items()})
F1.index.name = "BILGI_AY"
print(f"Fold 1 eğitim kohort sayısı (MIN_TRAIN = {MIN_TRAIN}; altı kalan aday elenir):")
display(F1.style.apply(lambda s: ["background-color:#fee2e2" if v < MIN_TRAIN else "" for v in s]))


# %% [8] Hızlı regresyon motoru (alpha yolu, dış CV'de)
# Ridge / Lasso / ElasticNet: her fold'da tüm alpha yolu TEK seferde çözülür; her alpha ayrı aday olarak
# AYNI boşluklu walk-forward CV ile ölçülür. Standartlaştırma her fold'un SADECE eğitim kısmıyla yapılır.
# Çözümler sklearn Ridge / Lasso / ElasticNet(+StandardScaler) ile birebir aynıdır (alt kısımda test).
def _std(X, y):
    mx, sx = X.mean(axis=0), X.std(axis=0)
    sx = np.where(sx > 0, sx, 1.0)
    return (X - mx) / sx, y - y.mean(), mx, sx, y.mean()


def alpha_yolu(Xs, yc, aile, alphalar, l1=None):
    """Standart X, merkezlenmiş y -> (n_alpha x p) katsayı matrisi."""
    if aile == "Ridge":
        U, s_, Vt = np.linalg.svd(Xs, full_matrices=False)
        Uy = U.T @ yc
        return np.stack([Vt.T @ (s_ / (s_ ** 2 + a) * Uy) for a in alphalar])
    Xf = np.asfortranarray(Xs)
    _, coefs, _ = enet_path(Xf, yc, l1_ratio=1.0 if aile == "Lasso" else l1, alphas=alphalar,
                            max_iter=CONFIG["MAX_ITER"], tol=CONFIG["MODEL_TOL"],
                            check_input=False, precompute=Xf.T @ Xf, Xy=Xf.T @ yc)
    return coefs.T


def yol_varyantlari(aile):
    if aile == "Ridge":
        return [(np.asarray(CONFIG["ALPHA_RIDGE"], float), None)]
    if aile == "Lasso":
        return [(np.asarray(CONFIG["ALPHA_LASSO"], float), None)]
    return [(np.asarray(CONFIG["ALPHA_LASSO"], float), l1) for l1 in CONFIG["L1_ORANLARI"]]


def cv_yol(kolonlar, aile, foldlar=None):
    """Bir aday + bir aile için tüm alpha'ların CV sonuçları: [(params, metrikler), ...]."""
    X, hazir = fold_hazirla(kolonlar, foldlar)
    sonuc = []
    for alphalar, l1 in yol_varyantlari(aile):
        alphalar = np.sort(alphalar)[::-1]
        fold_P = []
        for tr, val in hazir:
            Xs, yc, mx, sx, my = _std(X[tr], Y_CV[tr])
            B = alpha_yolu(Xs, yc, aile, alphalar, l1)                 # n_alpha x p
            fold_P.append((tr, val, ((X[val] - mx) / sx) @ B.T + my, Xs @ B.T + my))
        for j, a in enumerate(alphalar):
            kay = [(tr, val, P[:, j], Ptr[:, j]) for tr, val, P, Ptr in fold_P]
            p = {"alpha": float(a)} if l1 is None else {"alpha": float(a), "l1": float(l1)}
            sonuc.append((p, metrik_ozet(kay)))
    return sonuc


def model_fabrika(aile, p=None):
    p = p or {}
    return {
        "Dummy":       lambda: DummyRegressor(strategy="mean"),
        "Linear":      lambda: LinearRegression(),
        "Ridge":       lambda: Ridge(alpha=p["alpha"]),
        "Lasso":       lambda: Lasso(alpha=p["alpha"], max_iter=CONFIG["MAX_ITER"], tol=CONFIG["MODEL_TOL"]),
        "ElasticNet":  lambda: ElasticNet(alpha=p["alpha"], l1_ratio=p["l1"], max_iter=CONFIG["MAX_ITER"],
                                          tol=CONFIG["MODEL_TOL"]),
        "Ridge*":      lambda: Ridge(alpha=p["alpha"]),
        "Lasso*":      lambda: Lasso(alpha=p["alpha"], max_iter=100000),
        "ElasticNet*": lambda: ElasticNet(alpha=p["alpha"], l1_ratio=p["l1"], max_iter=100000),
        "Huber":       lambda: HuberRegressor(epsilon=p["eps"], alpha=p["alpha"], max_iter=2000),
        "SVR":         lambda: SVR(kernel=p["kernel"], C=p["C"], epsilon=p["eps"]),
        "ExtraTrees":  lambda: ExtraTreesRegressor(n_estimators=300, max_depth=p["depth"],
                                                   min_samples_leaf=p["leaf"], random_state=42),
        "GBoost":      lambda: GradientBoostingRegressor(n_estimators=p["n"], learning_rate=p["lr"],
                                                         max_depth=p["depth"], random_state=42),
    }[aile]()


# Doğrulama: alpha yolu = sklearn (aynı alpha, aynı aday, aynı fold'lar)
_kol = [ONCULLER[1], KKB_KOL]
_kontrol = []
for aile in ["Ridge", "Lasso", "ElasticNet"]:
    p_, m_yol = cv_yol(_kol, aile)[3]
    m_sk = cv_degerlendir(_kol, model_fabrika(aile, p_))
    _kontrol.append({"AILE": aile, "PARAMS": p_, "CV_WMAPE_YOL": m_yol["CV_WMAPE"], "CV_WMAPE_SKLEARN": m_sk["CV_WMAPE"],
                     "FARK": abs(m_yol["CV_WMAPE"] - m_sk["CV_WMAPE"])})
display(pd.DataFrame(_kontrol).style.format({"CV_WMAPE_YOL": "{:.6f}", "CV_WMAPE_SKLEARN": "{:.6f}", "FARK": "{:.1e}"}))
if max(k["FARK"] for k in _kontrol) > 1e-3:
    raise AssertionError("Alpha yolu sklearn ile uyuşmuyor.")
print("Alpha ızgarası | Ridge:", np.round(CONFIG["ALPHA_RIDGE"], 4).tolist(),
      "| Lasso/EN:", [f"{a:.1e}" for a in CONFIG["ALPHA_LASSO"]], "| EN L1:", CONFIG["L1_ORANLARI"])


# %% [9] Grid araması + sıralama
def kayit_ekle(s, kolonlar, aile, params, satir=None):
    kolonlar = list(kolonlar)
    trend = int(TREND_KOL in kolonlar)
    s.update({"GIRDILER": ", ".join(kisa_ad(c) for c in taban(kolonlar)), "MODEL": aile, "PARAMS": params,
              "BILGI_AY": bilgi_ayi_hesapla(kolonlar), "TOPLAM_K": len(kolonlar),
              "TREND": trend, "MEVSIM": int("SIN_AY" in kolonlar),
              "ORTALAMA": next((f"AGORT{ORTALAMA_PENCERE[c]}" for c in kolonlar if c in ORTALAMA_KOL), "Yok"),
              "KOLONLAR": tuple(kolonlar),
              "SECILEBILIR": bool((not trend or CONFIG["TREND_SECILEBILIR"]) and aile in CONFIG["FORMUL_AILELERI"])})
    return s


def rank_ekle(df):
    g = CONFIG["GECERLI"]
    df = df.copy()
    df["GECERLI"] = ((df["CV_WMAPE"] <= g["CV_WMAPE_DUMMY_ORAN"] * DUMMY["CV_WMAPE"])
                     & (df["EN_KOTU_WMAPE"] <= g["EN_KOTU_DUMMY_ORAN"] * DUMMY["EN_KOTU_WMAPE"])
                     & (df["OVERFIT_ORANI"] <= g["OVERFIT_ORANI_MAX"])
                     & (df["MUTLAK_FOLD_SAPMA"] <= g["SAPMA_DUMMY_ORAN"] * DUMMY["MUTLAK_FOLD_SAPMA"])
                     & (df["ARALIK_DISI"] == 0))
    df["RANK_SKORU"] = np.nan
    m = df["GECERLI"]
    if m.any():
        skor = 0.0
        for kol, w in CONFIG["AGIRLIK"].items():
            s = df.loc[m, kol].astype(float)
            s = s.clip(lower=1.0) if kol == "OVERFIT_ORANI" else s
            skor = skor + w * s.rank(pct=True).fillna(1.0)
        df.loc[m, "RANK_SKORU"] = skor
    return df.sort_values(["GECERLI", "RANK_SKORU", "CV_WMAPE"], ascending=[False, True, True]).reset_index(drop=True)


def varyant_etiketi(df):
    return np.select([(df["TREND"] == 1) & (df["MEVSIM"] == 1), df["TREND"] == 1, df["MEVSIM"] == 1],
                     ["Trend+Mevsim", "Trend", "Mevsim"], "Yok")


N_JOBS = 2          # paralel çekirdek sayısı (sorun çıkarsa 1 yap)
GRUP_BOYUTU = 50


def grup_calistir(grup):
    warnings.filterwarnings("ignore")                   # paralel işçilerde de uyarıları kapat
    kayit_, hata_ = [], []
    for kolonlar in grup:
        foldlar = fold_al(bilgi_ayi_hesapla(kolonlar))
        for aile in CONFIG["MODEL_AILELERI"]:
            try:
                if aile == "Linear":
                    kayit_.append(kayit_ekle(cv_degerlendir(kolonlar, model_fabrika(aile), foldlar),
                                             kolonlar, aile, None))
                else:                                    # her alpha ayrı satır
                    for p_, s_ in cv_yol(kolonlar, aile, foldlar):
                        kayit_.append(kayit_ekle(s_, kolonlar, aile, p_))
            except Exception as e:
                hata_.append({"GIRDILER": ", ".join(map(kisa_ad, kolonlar)), "MODEL": aile, "HATA": str(e)[:80]})
    return kayit_, hata_


for b in range(0, CONFIG["CHAMPION_MAX_AY"] + 1):        # fold'lar önceden hazır olsun (paralel işçiler için)
    fold_al(b)
adaylar = list(ADAY_TABLO["KOLONLAR"])
gruplar = [adaylar[i:i + GRUP_BOYUTU] for i in range(0, len(adaylar), GRUP_BOYUTU)]
t0, kayit, hatalar = time.time(), [], []
if N_JOBS > 1:
    from joblib import Parallel, delayed
    parca = N_JOBS * 4
    for i in range(0, len(gruplar), parca):
        for k_, h_ in Parallel(n_jobs=N_JOBS)(delayed(grup_calistir)(g) for g in gruplar[i:i + parca]):
            kayit += k_; hatalar += h_
        print(f"  {min((i + parca) * GRUP_BOYUTU, len(adaylar)):,}/{len(adaylar):,} aday | {time.time() - t0:.0f} sn")
else:
    for i, g in enumerate(gruplar, 1):
        k_, h_ = grup_calistir(g)
        kayit += k_; hatalar += h_
        if i % 10 == 0:
            print(f"  {i * GRUP_BOYUTU:,}/{len(adaylar):,} aday | {time.time() - t0:.0f} sn")
grid_ham = pd.DataFrame(kayit)
grid = rank_ekle(grid_ham)
grid["VARYANT"] = varyant_etiketi(grid)
print(f"{len(grid):,} değerlendirme | geçerli {int(grid['GECERLI'].sum()):,} | elenen {len(hatalar)} | "
      f"{(time.time() - t0) / 60:.1f} dk")
if hatalar:
    print("Elenme nedenleri:", pd.DataFrame(hatalar)["HATA"].value_counts().to_dict())

GOSTER = ["GIRDILER", "MODEL", "PARAMS", "BILGI_AY", "TOPLAM_K", "TREND", "MEVSIM", "ORTALAMA", "CV_WMAPE",
          "EN_KOTU_WMAPE", "FOLD_WMAPE_STD", "OVERFIT_ORANI", "DELTA_MAE", "CV_SAPMA", "MUTLAK_FOLD_SAPMA",
          "YON_AGIRLIKLI", "FOLD1_EGITIM_N", "GECERLI", "SECILEBILIR", "RANK_SKORU"]
FMT = {**{c: "{:.2f}" for c in ["CV_WMAPE", "EN_KOTU_WMAPE", "FOLD_WMAPE_STD"]},
       **{c: "{:.4f}" for c in ["OVERFIT_ORANI", "DELTA_MAE", "CV_SAPMA", "MUTLAK_FOLD_SAPMA",
                                "YON_AGIRLIKLI", "RANK_SKORU"]}}
print("=== Bileşik skora göre ilk 15 ===")
display(grid[GOSTER].head(15).style.format(FMT, decimal=","))


def grup_ozet(df, by):
    out = df.groupby(by).agg(VARYANT=("CV_WMAPE", "size"), EN_IYI_CV_WMAPE=("CV_WMAPE", "min"),
                             MEDYAN_CV_WMAPE=("CV_WMAPE", "median"))
    out["GECERLI"] = df[df["GECERLI"]].groupby(by).size().reindex(out.index).fillna(0).astype(int)
    return out[["VARYANT", "GECERLI", "EN_IYI_CV_WMAPE", "MEDYAN_CV_WMAPE"]]


OZET_VARYANT = grup_ozet(grid, "VARYANT").reindex(["Yok", "Mevsim", "Trend", "Trend+Mevsim"]).dropna(how="all")
OZET_BILGI_AY = grup_ozet(grid, "BILGI_AY")
OZET_AILE = grup_ozet(grid, "MODEL")
OZET_ORTALAMA = grup_ozet(grid, "ORTALAMA")
OZET_K = grup_ozet(grid, "TOPLAM_K")
for ad, t in [("Takvim (trend / mevsim)", OZET_VARYANT), ("Bilgi ayı (en geç öncül)", OZET_BILGI_AY),
              ("Model ailesi", OZET_AILE), ("Geçmiş ortalama", OZET_ORTALAMA), ("Toplam girdi sayısı", OZET_K)]:
    print(f"=== {ad} ===")
    display(t.style.format({"EN_IYI_CV_WMAPE": "{:.2f}", "MEDYAN_CV_WMAPE": "{:.2f}"}, decimal=",", na_rep="-"))


# %% [10] Optuna: geniş model aileleri (sağlamlık kontrolü)
try:
    import optuna
except ImportError:
    optuna = None
    print("optuna kurulu değil (pip install optuna) -> Bölüm 10 atlanıyor.")
if CONFIG["OPTUNA_CALISTIR"] and optuna is not None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    setler = (grid[grid["GECERLI"] & grid["SECILEBILIR"]].drop_duplicates("KOLONLAR")["KOLONLAR"]
              .head(CONFIG["OPTUNA_SET_SAYISI"]).tolist())
    opt_kayit = []

    def amac(trial):
        kolonlar = list(setler[trial.suggest_int("set", 0, len(setler) - 1)])
        aile = trial.suggest_categorical("aile", ["Ridge*", "Lasso*", "ElasticNet*", "Huber",
                                                  "SVR", "ExtraTrees", "GBoost"])
        p = {}
        if aile in ("Ridge*", "Huber"):                       # standart X'te ceza
            p["alpha"] = trial.suggest_float(f"{aile}_alpha", 1e-3, 1e3, log=True)
        if aile in ("Lasso*", "ElasticNet*"):                 # DÜZELTME: hedef ölçeğinde ceza (alpha_max ~1e-3)
            p["alpha"] = trial.suggest_float(f"{aile}_alpha", 1e-7, 1e-2, log=True)
        if aile == "ElasticNet*":
            p["l1"] = trial.suggest_float("l1", 0.05, 0.95)
        if aile == "Huber":
            p["eps"] = trial.suggest_float("h_eps", 1.1, 2.5)
        if aile == "SVR":
            p.update(kernel=trial.suggest_categorical("kernel", ["linear", "rbf"]),
                     C=trial.suggest_float("C", 1e-2, 1e2, log=True),
                     eps=trial.suggest_float("svr_eps", 1e-5, 1e-3, log=True))
        if aile == "ExtraTrees":
            p.update(depth=trial.suggest_int("et_depth", 2, 4), leaf=trial.suggest_int("leaf", 2, 6))
        if aile == "GBoost":
            p.update(n=trial.suggest_int("n", 50, 300), lr=trial.suggest_float("lr", 0.01, 0.2, log=True),
                     depth=trial.suggest_int("gb_depth", 1, 3))
        s = cv_degerlendir(kolonlar, model_fabrika(aile, p))
        opt_kayit.append(kayit_ekle(s, kolonlar, aile, p))
        return s["CV_WMAPE"], (s["DELTA_MAE"] if np.isfinite(s["DELTA_MAE"]) else 1.0)

    study = optuna.create_study(directions=["minimize", "minimize"], sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(amac, n_trials=CONFIG["OPTUNA_TRIALS"])
    grid = rank_ekle(pd.concat([grid_ham, pd.DataFrame(opt_kayit)], ignore_index=True))
    grid["VARYANT"] = varyant_etiketi(grid)
    OZET_AILE = grup_ozet(grid, "MODEL")
    print(f"Optuna sonrası: {len(grid):,} değerlendirme | geçerli {int(grid['GECERLI'].sum()):,}")
    display(OZET_AILE.sort_values("EN_IYI_CV_WMAPE").style.format(
        {"EN_IYI_CV_WMAPE": "{:.2f}", "MEDYAN_CV_WMAPE": "{:.2f}"}, decimal=",", na_rep="-"))
else:
    print("Optuna atlandı.")


# %% [11] Champion / Challenger aday havuzları
AILE_SIRA = {"Linear": 0, "Ridge": 1, "Ridge*": 1, "Lasso": 2, "Lasso*": 2,
             "ElasticNet": 3, "ElasticNet*": 3, "Huber": 4}


def hac_p(kolonlar, df):
    df = df.dropna(subset=list(kolonlar))
    X = sm.add_constant(df[list(kolonlar)].astype(float).to_numpy(), has_constant="add")
    y = df[HEDEF].astype(float).to_numpy()
    lag = int(np.ceil(4 * (len(y) / 100) ** (2 / 9)))
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag}).pvalues[1:]


def kisa_liste(havuz, baslik, ilk_n=CONFIG["HAC_ILK_N"]):
    """1) CV'de en iyi ilk_n aday içinde TÜM katsayıları anlamlı olanlar (HAC, sadece geliştirme verisi)
       2) 1-SE: en iyi anlamlı modelin CV_WMAPE + fold std/sqrt(N) içindekiler istatistiksel olarak eşit
       3) bunlar arasında: az girdi > erken bilgi ayı > basit aile > bileşik skor."""
    h = havuz.nsmallest(ilk_n, "CV_WMAPE").drop_duplicates(["KOLONLAR", "MODEL"]).copy()
    h["HAC_MAKS_P"] = h["KOLONLAR"].apply(lambda k: float(np.max(hac_p(k, cv_df))))
    a = h[h["HAC_MAKS_P"] <= CONFIG["HAC_P_ESIK"]]
    n_anl = len(a)
    if a.empty:
        print(f"UYARI ({baslik}): tüm katsayıları anlamlı model yok, filtre uygulanmadı.")
        a = h
    en_iyi = a.nsmallest(1, "CV_WMAPE").iloc[0]
    se = en_iyi["FOLD_WMAPE_STD"] / np.sqrt(CONFIG["N_SPLITS"])
    esik = en_iyi["CV_WMAPE"] + se
    k_ = a[a["CV_WMAPE"] <= esik].copy()
    k_["AILE_SIRA"] = k_["MODEL"].map(AILE_SIRA)
    k_ = k_.sort_values(["TOPLAM_K", "BILGI_AY", "AILE_SIRA", "RANK_SKORU"]).reset_index(drop=True)
    print(f"=== {baslik}: anlamlı {n_anl}/{len(h)} | en iyi{' anlamlı' if n_anl else ''} {en_iyi['CV_WMAPE']:.2f} + 1 SE ({se:.2f}) "
          f"= eşik {esik:.2f} | eşik içinde {len(k_)} model ===")
    print(f"    (anlamlılık filtresi olmadan en iyi CV_WMAPE {h['CV_WMAPE'].min():.2f}, "
          f"o modelin HAC_MAKS_P {h.iloc[0]['HAC_MAKS_P']:.3f})")
    display(k_[["GIRDILER", "MODEL", "BILGI_AY", "MEVSIM", "ORTALAMA", "TOPLAM_K", "CV_WMAPE", "EN_KOTU_WMAPE",
                "FOLD_WMAPE_STD", "HAC_MAKS_P", "RANK_SKORU"]].head(15).style.format(
        {"CV_WMAPE": "{:.2f}", "EN_KOTU_WMAPE": "{:.2f}", "FOLD_WMAPE_STD": "{:.2f}",
         "HAC_MAKS_P": "{:.3f}", "RANK_SKORU": "{:.4f}"}, decimal=","))
    return k_


aday = grid[grid["GECERLI"] & grid["SECILEBILIR"]].copy()
if aday.empty:
    raise ValueError("Seçilebilir geçerli model yok: GECERLI eşiklerini kontrol et.")
erken = aday[aday["BILGI_AY"] <= CONFIG["CHALLENGER_MAX_AY"]]
if erken.empty:
    raise ValueError(f"<= {CONFIG['CHALLENGER_MAX_AY']}. ay bilgisiyle geçerli model yok.")
print(f"Seçilebilir geçerli: champion havuzu {len(aday):,} | challenger havuzu {len(erken):,}")
kisa = kisa_liste(aday, f"CHAMPION (<= {CONFIG['CHAMPION_MAX_AY']}. ay)")
kisa_c = kisa_liste(erken, f"CHALLENGER (<= {CONFIG['CHALLENGER_MAX_AY']}. ay)")


# %% [12] Model seçimi: anlamlılık + 1-SE + sadelik, Champion / Challenger
def spec_yap(satir, rol):
    return {"rol": rol, "kolonlar": list(satir["KOLONLAR"]), "aile": satir["MODEL"], "params": satir["PARAMS"],
            "bilgi_ay": int(satir["BILGI_AY"]), "girdiler": satir["GIRDILER"], "mevsim": int(satir["MEVSIM"]),
            "ad": f"{rol}: {satir['GIRDILER']}{' + mevsim' if satir['MEVSIM'] else ''} | {satir['MODEL']} | "
                  f"{int(satir['BILGI_AY'])}. ay"}


CHAMP_SATIR = kisa.iloc[0]
ham_set = lambda k: {c for c in k if c in HAM_HAVUZ}             # ortalama/takvim hariç ham girdiler
farkli = lambda df: df[df["KOLONLAR"].apply(ham_set) != ham_set(CHAMP_SATIR["KOLONLAR"])] \
    if CONFIG["CHALLENGER_FARKLI_GIRDI"] else df
# Challenger: <= CHALLENGER_MAX_AY, champion'dan farklı ham girdi. Yedek yollar da anlamlılık kuralına uyar.
ch = farkli(kisa_c)
CHAL_KAYNAK = "1-SE + anlamlılık"
if ch.empty:
    h_ = erken.nsmallest(CONFIG["HAC_ILK_N"], "CV_WMAPE").drop_duplicates(["KOLONLAR", "MODEL"]).copy()
    h_["HAC_MAKS_P"] = h_["KOLONLAR"].apply(lambda k: float(np.max(hac_p(k, cv_df))))
    ch = farkli(h_[h_["HAC_MAKS_P"] <= CONFIG["HAC_P_ESIK"]]).sort_values("CV_WMAPE")
    CHAL_KAYNAK = "anlamlı erken modeller (1-SE dışı)"
if ch.empty:
    ch = kisa_c
    CHAL_KAYNAK = "1-SE listesi (champion ile aynı ham girdiler)"
print(f"Challenger kaynağı: {CHAL_KAYNAK}")
CHAL_SATIR = ch.iloc[0]

CHAMP, CHAL = spec_yap(CHAMP_SATIR, "CHAMPION"), spec_yap(CHAL_SATIR, "CHALLENGER")
SECIMLER = {"CHAMPION": CHAMP, "CHALLENGER": CHAL}
MODELLER = dict(SECIMLER)
for s_ in MODELLER.values():
    print(s_["ad"])
SECIM_TABLO = pd.DataFrame([CHAMP_SATIR, CHAL_SATIR], index=list(MODELLER))
SECIM_TABLO["HAC_MAKS_P"] = SECIM_TABLO["KOLONLAR"].apply(lambda k: float(np.max(hac_p(k, cv_df))))
display(SECIM_TABLO[GOSTER[:-3] + ["HAC_MAKS_P"]].style.format({**FMT, "HAC_MAKS_P": "{:.3f}"}, decimal=","))

# Şeffaflık: challenger yedek yoldan geldiyse aday listesi (seçimi değiştirmez)
if CHAL_KAYNAK != "1-SE + anlamlılık":
    print(f"1-SE challenger listesindeki ham girdi setleri: "
          f"{kisa_c['KOLONLAR'].apply(lambda k: ', '.join(sorted(map(kisa_ad, ham_set(k))))).value_counts().to_dict()}"
          f" | champion: {', '.join(sorted(map(kisa_ad, ham_set(CHAMP_SATIR['KOLONLAR']))))}")
    print("-> hepsi champion ile aynı ham girdi; challenger = anlamlı + farklı girdili erken modellerin en iyisi:")
    display(ch[["GIRDILER", "MODEL", "PARAMS", "BILGI_AY", "TOPLAM_K", "CV_WMAPE", "EN_KOTU_WMAPE",
                "FOLD_WMAPE_STD", "HAC_MAKS_P"]].head(8).style.format(
        {"CV_WMAPE": "{:.2f}", "EN_KOTU_WMAPE": "{:.2f}", "FOLD_WMAPE_STD": "{:.2f}", "HAC_MAKS_P": "{:.3f}"},
        decimal=","))


# %% [13] OOT walk-forward, performans tablosu, overlay
# DÜZELTME: her olgun kohort (validasyon ve OOT), KENDİ tahmin anında (AY + bilgi ayı) hedefi belli olan
# kohortlarla yeniden eğitilen modelle tahmin edilir. Tek fit yok.
olgun_df = d[d["HEDEF_KULLANILABILIR"]].copy()


def wf_tahmin(spec):
    kol, kayit = spec["kolonlar"], []
    for i in olgun_df.index:
        c = olgun_df.loc[i]
        tahmin_ay = tahmin_ayi_hesapla(c["AY"], spec["bilgi_ay"])
        eg = olgun_df[olgun_df["HEDEF_ERISIM"] < tahmin_ay].dropna(subset=kol)
        if len(eg) < MIN_TRAIN or c[kol].isna().any():
            continue
        m = pipe_fit(model_fabrika(spec["aile"], spec["params"]), eg[kol], eg[HEDEF], agirlik_al(eg))
        kayit.append({"DONEM_1": int(c["DONEM_1"]), "AY": c["AY"], "BOLGE": c["BOLGE"], "TAHMIN_AY": tahmin_ay,
                      "HEDEF_ERISIM": c["HEDEF_ERISIM"], "GERCEK": float(c[HEDEF]),
                      "TAHMIN": float(m.predict(olgun_df.loc[[i], kol])[0]),
                      "DUMMY": float(eg[HEDEF].mean()), "EGITIM_N": len(eg), "EGITIM_SON": str(eg["AY"].max())})
    w = pd.DataFrame(kayit)
    w["ARTIK"] = w["TAHMIN"] - w["GERCEK"]
    return w


def metrik(g, p):
    g, p = np.asarray(g, float), np.asarray(p, float)
    if len(g) < 3:
        return {}
    dg, dp = np.diff(g), np.diff(p)
    return {"N": len(g), "WMAPE": wmape(g, p), "MAE_yp": float(np.mean(np.abs(p - g)) * 100),
            "RMSE_yp": float(np.sqrt(np.mean((p - g) ** 2)) * 100), "SAPMA_yp": float(np.mean(p - g) * 100),
            "R2": float(1 - np.sum((g - p) ** 2) / np.sum((g - g.mean()) ** 2)),
            "KORELASYON": float(np.corrcoef(g, p)[0, 1]) if np.std(p) > 0 else np.nan,
            "YON": f"{int((np.sign(dg) == np.sign(dp)).sum())}/{len(dg)}",
            "POZITIF_SAPMA": f"{int((p > g).sum())}/{len(g)}"}


VAL_BAS = AY_CV[fold_al(0)[0][1][0]]
WF, OVERLAY, PERF, SIZINTI = {}, {}, [], []
for rol, spec in MODELLER.items():
    w = wf_tahmin(spec)
    WF[rol] = w
    for bolge, maske in [("VALIDASYON", (w["BOLGE"] == "GELISTIRME") & (w["AY"] >= VAL_BAS)),
                         ("OOT", w["BOLGE"] == "OOT")]:
        s = w[maske]
        mm, dm = metrik(s["GERCEK"], s["TAHMIN"]), metrik(s["GERCEK"], s["DUMMY"])
        PERF.append({"MODEL": rol, "BOLGE": bolge, **mm, "DUMMY_WMAPE": dm["WMAPE"],
                     "BECERI": 1 - mm["WMAPE"] / dm["WMAPE"]})
    o1 = w[w["BOLGE"] == "OOT"].iloc[0]
    SIZINTI.append({"MODEL": rol, "BILGI_AY": spec["bilgi_ay"], "ILK_OOT_KOHORT": str(o1["AY"]),
                    "TAHMIN_ANI": str(o1["TAHMIN_AY"]), "EGITIM_SON_KOHORT": o1["EGITIM_SON"],
                    "EGITIM_N": o1["EGITIM_N"]})
    son = w[w["HEDEF_ERISIM"] < BILGI].sort_values("HEDEF_ERISIM").tail(CONFIG["OVERLAY_PENCERE"])
    ayni = int(max((son["ARTIK"] > 0).sum(), (son["ARTIK"] < 0).sum()))
    OVERLAY[rol] = {"deger": float(son["ARTIK"].mean()), "ayni_isaret": ayni, "n": len(son),
                    "donem": f"{son['DONEM_1'].min()}-{son['DONEM_1'].max()}",
                    "uygula": ayni >= CONFIG["OVERLAY_MIN_AYNI_ISARET"]}

PERF = pd.DataFrame(PERF)
SIZINTI = pd.DataFrame(SIZINTI)
SIZINTI["KONTROL"] = np.where([pd.Period(e, "M") + 12 + CONFIG["RAPOR_GECIKME_AY"] < pd.Period(t, "M")
                               for e, t in zip(SIZINTI["EGITIM_SON_KOHORT"], SIZINTI["TAHMIN_ANI"])], "OK", "SIZINTI")
print("Sızıntı kontrolü (eğitimin son kohortu + 12 < tahmin anı olmalı):")
display(SIZINTI)
print(f"Validasyon: {VAL_BAS} – {AY_CV.max()} | OOT: {d.loc[oot_idx, 'AY'].min()} – {d.loc[oot_idx, 'AY'].max()}")
display(PERF.style.format({"WMAPE": "{:.2f}", "MAE_yp": "{:.3f}", "RMSE_yp": "{:.3f}", "SAPMA_yp": "{:+.3f}",
                           "R2": "{:.3f}", "KORELASYON": "{:.3f}", "DUMMY_WMAPE": "{:.2f}", "BECERI": "{:.3f}"},
                          decimal=","))
for rol, o in OVERLAY.items():
    print(f"{rol}: overlay {o['deger'] * 100:+.3f} yp | son {o['n']} olgun kohortun {o['ayni_isaret']}'i aynı "
          f"işaret ({o['donem']}) -> {'UYGULANIR' if o['uygula'] else 'uygulanmaz'}")
_p = PERF.set_index(["MODEL", "BOLGE"])
print("\nÖZET (rakamlar bu çalıştırmadan, elle yazılmaz):")
for rol in MODELLER:
    v, o = _p.loc[(rol, "VALIDASYON")], _p.loc[(rol, "OOT")]
    print(f"  {rol}: validasyon WMAPE {sayi(v['WMAPE'])} | OOT WMAPE {sayi(o['WMAPE'])} "
          f"(naif {sayi(o['DUMMY_WMAPE'])}, beceri {sayi(o['BECERI'])}) | OOT sapma {o['SAPMA_yp']:+.3f} yp")

# Karar metni (bu çalıştırmanın sonuçlarından; elle yazılmaz)
_c, _h = _p.loc[("CHAMPION", "OOT")], _p.loc[("CHALLENGER", "OOT")]
KARAR_GEREKCE = (f"Champion CV kuralıyla seçildi ({CHAMP['ad']}). OOT WMAPE: champion {sayi(_c['WMAPE'])} "
                 f"(naif {sayi(_c['DUMMY_WMAPE'])}, beceri {_c['BECERI']:+.2f}), challenger {sayi(_h['WMAPE'])} "
                 f"(naif {sayi(_h['DUMMY_WMAPE'])}, beceri {_h['BECERI']:+.2f}). OOT seçimde kullanılmadı.")
print("\nKARAR:", KARAR_GEREKCE)

# Train (in-sample) / Validasyon / OOT metrik tablosu
#   EGITIM     : ilk validasyon kohortundan önceki olgun kohortlarla kurulan model, aynı kohortlarda (in-sample)
#   VALIDASYON : walk-forward tahminler (Bölüm 13, WF)
#   OOT        : walk-forward tahminler (Bölüm 13, WF)
# yp = yüzde puan. SAPMA = ort(tahmin − gerçek): + ise model yukarı, − ise aşağı tahmin ediyor.
METRIK_TABLO = []
for rol, spec in MODELLER.items():
    kol = spec["kolonlar"]
    egit = olgun_df[olgun_df["AY"] < VAL_BAS].dropna(subset=kol)
    m_eg = pipe_fit(model_fabrika(spec["aile"], spec["params"]), egit[kol], egit[HEDEF])
    w = WF[rol]
    parcalar = {
        "EGITIM": (egit[HEDEF].to_numpy(), m_eg.predict(egit[kol]), np.full(len(egit), egit[HEDEF].mean())),
        "VALIDASYON": tuple(w.loc[(w["BOLGE"] == "GELISTIRME") & (w["AY"] >= VAL_BAS), c].to_numpy()
                            for c in ["GERCEK", "TAHMIN", "DUMMY"]),
        "OOT": tuple(w.loc[w["BOLGE"] == "OOT", c].to_numpy() for c in ["GERCEK", "TAHMIN", "DUMMY"]),
    }
    for bolge, (g, p, nv) in parcalar.items():
        mm = metrik(g, p)
        METRIK_TABLO.append({"MODEL": rol, "BOLGE": bolge, "N": mm["N"],
                             "MAE_yp": mm["MAE_yp"], "RMSE_yp": mm["RMSE_yp"], "SAPMA_yp": mm["SAPMA_yp"],
                             "WMAPE": mm["WMAPE"], "NAIF_WMAPE": wmape(g, nv),
                             "BECERI": 1 - mm["WMAPE"] / wmape(g, nv),
                             "R2": mm["R2"], "KORELASYON": mm["KORELASYON"], "YON": mm["YON"],
                             "POZITIF_SAPMA": mm["POZITIF_SAPMA"]})
METRIK_TABLO = pd.DataFrame(METRIK_TABLO)
print("Train (in-sample) / Validasyon (walk-forward) / OOT (walk-forward) | yp = yüzde puan")
display(METRIK_TABLO.style.format(
    {"MAE_yp": "{:.3f}", "RMSE_yp": "{:.3f}", "SAPMA_yp": "{:+.3f}", "WMAPE": "{:.2f}", "NAIF_WMAPE": "{:.2f}",
     "BECERI": "{:+.2f}", "R2": "{:.3f}", "KORELASYON": "{:.3f}"}, decimal=",")
    .apply(lambda r: ["background-color:#fff7ed" if r["BOLGE"] == "OOT" else
                      "background-color:#eff6ff" if r["BOLGE"] == "VALIDASYON" else "" for _ in r], axis=1))


# %% [14] İstatistiksel testler (champion + challenger) + girdi kayması + dönüşüm oranı
# Aynı test seti iki modele birebir uygulanır. Ridge/Lasso için anlamlılık (HAC), RESET, VIF aynı kolonlarla
# OLS üzerinden; artık testleri (DW, Ljung-Box, BP, JB) seçilen modelin KENDİ artıklarıyla yapılır.
# Chow: OOT başında (rejim kırılması) katsayılar değişiyor mu? (otokorelasyonda iyimser; dönüşüm oranıyla birlikte oku)
from scipy import stats
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor

OOT_BAS = d.loc[oot_idx, "AY"].min()


def chow_p(t_df, kol, kirilma):
    """Kırılma öncesi ve sonrası iki ayrı OLS, tek OLS'ten anlamlı derecede iyi mi? (p küçük = kırılma var)"""
    X = sm.add_constant(t_df[kol].astype(float).to_numpy(), has_constant="add")
    y = t_df[HEDEF].astype(float).to_numpy()
    m = (t_df["AY"] >= kirilma).to_numpy()
    k, n = X.shape[1], len(y)
    if m.sum() <= k or (~m).sum() <= k:
        return np.nan
    ssr = lambda a: sm.OLS(y[a], X[a]).fit().ssr
    s_tek, s_iki = sm.OLS(y, X).fit().ssr, ssr(~m) + ssr(m)
    F = ((s_tek - s_iki) / k) / (s_iki / (n - 2 * k))
    return float(stats.f.sf(F, k, n - 2 * k))


def orijinal_katsayi(pipe, kol):
    """StandardScaler + model pipeline'ının katsayılarını orijinal ölçeğe çevirir."""
    sc, est = pipe.steps[0][1], pipe.steps[-1][1]
    b = np.ravel(est.coef_) / sc.scale_
    return float(est.intercept_ - np.sum(b * sc.mean_)), dict(zip(kol, b))


def test_seti(spec):
    kol = spec["kolonlar"]
    t_df = olgun_df.dropna(subset=kol)
    Xc = sm.add_constant(t_df[kol].astype(float).to_numpy(), has_constant="add")
    yv = t_df[HEDEF].astype(float).to_numpy()
    ols = sm.OLS(yv, Xc).fit()
    m = pipe_fit(model_fabrika(spec["aile"], spec["params"]), t_df[kol], t_df[HEDEF])
    resid = m.predict(t_df[kol]) - yv
    rho = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    jb = jarque_bera(resid)
    p_gel, p_tum = hac_p(kol, cv_df), hac_p(kol, olgun_df)
    r2_model = 1 - np.sum(resid ** 2) / np.sum((yv - yv.mean()) ** 2)
    pf = lambda ps: ", ".join(f"{kisa_ad(k)}={p:.3f}" for k, p in zip(kol, ps))
    sonuc = {
        "HAC p (geliştirme, seçimde kullanılan)": pf(p_gel),
        "HAC p (tüm olgun veri, OOT dahil)": pf(p_tum),
        "R² model / Düz. R² OLS (tüm olgun)": f"{r2_model:.3f} / {ols.rsquared_adj:.3f}",
        "Durbin-Watson": f"{durbin_watson(resid):.3f}",
        "Efektif gözlem": f"{len(yv) * (1 - rho) / (1 + rho):.1f} / {len(yv)} (rho {rho:.2f})",
        "Ljung-Box p (lag 1/3/6)": " / ".join(f"{p:.3f}" for p in acorr_ljungbox(resid, lags=[1, 3, 6])["lb_pvalue"]),
        "Breusch-Pagan p": f"{het_breuschpagan(resid, Xc)[1]:.3f}",
        "Jarque-Bera p": f"{jb[1]:.3f} (çarp. {jb[2]:.2f}, bas. {jb[3]:.2f})",
        "Ramsey RESET p": f"{linear_reset(ols, power=2, use_f=True).pvalue:.3f}",
        "VIF": ", ".join(f"{kisa_ad(k)}={variance_inflation_factor(Xc, i + 1):.2f}" for i, k in enumerate(kol))
               if len(kol) > 1 else "-",
        f"Chow p (kırılma {OOT_BAS})": f"{chow_p(t_df, kol, OOT_BAS):.3f}",
    }
    b0, b = orijinal_katsayi(m, kol)
    kats = pd.DataFrame({"MODEL": spec["rol"], "DEGISKEN": [kisa_ad(k) for k in kol],
                         "OLS_KATSAYI": ols.params[1:], "SECILEN_MODEL_KATSAYI": [b[k] for k in kol],
                         "HAC_p_GELISTIRME": p_gel, "HAC_p_TUM": p_tum})
    return sonuc, kats, b0


YORUM = {
    "HAC p (geliştirme, seçimde kullanılan)": f"<= {CONFIG['HAC_P_ESIK']} anlamlı",
    "HAC p (tüm olgun veri, OOT dahil)": "OOT eklenince anlamlılık korunuyor mu?",
    "R² model / Düz. R² OLS (tüm olgun)": "in-sample açıklama gücü",
    "Durbin-Watson": "2 = otokorelasyon yok, <1,5 pozitif otokorelasyon",
    "Efektif gözlem": "otokorelasyon sonrası bağımsız bilgi",
    "Ljung-Box p (lag 1/3/6)": "<0,05 otokorelasyon var",
    "Breusch-Pagan p": ">0,05 sabit varyans",
    "Jarque-Bera p": ">0,05 normal",
    "Ramsey RESET p": ">0,05 doğrusal form yeterli",
    "VIF": "<5 iyi, >10 sorunlu",
    f"Chow p (kırılma {OOT_BAS})": "<0,05 OOT başında katsayılar değişmiş (otokorelasyonda iyimser)",
}

TEST_SONUC, KATSAYI_TEST, SABIT = {}, [], {}
for rol, spec in MODELLER.items():
    TEST_SONUC[rol], k_, SABIT[rol] = test_seti(spec)
    KATSAYI_TEST.append(k_)
TEST_KARSI = pd.DataFrame(TEST_SONUC)
TEST_KARSI["YORUM"] = TEST_KARSI.index.map(YORUM)
TEST_KARSI.index.name = "TEST"
KATSAYI_TEST = pd.concat(KATSAYI_TEST, ignore_index=True)

for rol, spec in MODELLER.items():
    print(spec["ad"], f"| params {spec['params']}")
print("=== İstatistiksel testler: Champion vs Challenger (tüm olgun veri) ===")
display(TEST_KARSI)
print("=== Katsayılar: OLS vs seçilen model (orijinal ölçek) + HAC p ===")
print("    Ridge'de SECILEN_MODEL_KATSAYI, OLS'e göre sıfıra büzülmüş halidir; işaretler aynı olmalı.")
display(KATSAYI_TEST.style.format({"OLS_KATSAYI": "{:.6g}", "SECILEN_MODEL_KATSAYI": "{:.6g}",
                                   "HAC_p_GELISTIRME": "{:.3f}", "HAC_p_TUM": "{:.3f}"}, decimal=","))
isaret_ayni = (np.sign(KATSAYI_TEST["OLS_KATSAYI"]) == np.sign(KATSAYI_TEST["SECILEN_MODEL_KATSAYI"])).all()
print(f"İşaret tutarlılığı (OLS = seçilen model): {'OK' if isaret_ayni else 'FARKLI İŞARET VAR'}")
TESTLER = TEST_KARSI.reset_index()[["TEST", "CHAMPION"]].rename(columns={"CHAMPION": "SONUC"})

drift_kol = list(dict.fromkeys([c for s_ in MODELLER.values() for c in taban(s_["kolonlar"])] + [HEDEF]))
DRIFT = pd.DataFrame([{"DEGISKEN": kisa_ad(c), "GEL_ORT": d.loc[gel_idx, c].mean(), "OOT_ORT": d.loc[oot_idx, c].mean(),
                       "DEGISIM_%": (d.loc[oot_idx, c].mean() / d.loc[gel_idx, c].mean() - 1) * 100,
                       "DRIFT_Z": (d.loc[oot_idx, c].mean() - d.loc[gel_idx, c].mean()) / d.loc[gel_idx, c].std()}
                      for c in drift_kol])
print("=== Girdi / hedef kayması (geliştirme -> OOT) ===")
display(DRIFT.style.format({"GEL_ORT": "{:.4f}", "OOT_ORT": "{:.4f}", "DEGISIM_%": "{:+.1f}", "DRIFT_Z": "{:+.2f}"}, decimal=","))

oran_onc = [c for c in drift_kol if c in ONCULLER]
r = olgun_df.assign(DONEM=olgun_df["AY"].dt.year.astype(str) + "-H" + ((olgun_df["AY"].dt.month > 6) + 1).astype(str))
DONUSUM = r.groupby("DONEM").agg(N=(HEDEF, "size"), HEDEF_ORT=(HEDEF, "mean"), **{f"ORT_{kisa_ad(c)}": (c, "mean") for c in oran_onc})
for c in oran_onc:
    DONUSUM[f"HEDEF/{kisa_ad(c)}"] = DONUSUM["HEDEF_ORT"] / DONUSUM[f"ORT_{kisa_ad(c)}"]
print("=== Erken gecikmenin batığa dönüşüm oranı (yarıyıl) ===")
display(DONUSUM.style.format({c: ("{:.2%}" if c.startswith(("HEDEF_ORT", "ORT_")) else "{:.2f}")
                              for c in DONUSUM.columns if c != "N"}, decimal=","))


# %% [15] Gelecek tahmini: olgunlaşmamış kohortlar veriden (manuel giriş yok)
# Yeni kohortlar (BOLGE = TAHMIN) Bölüm 1-4'te olgun kohortlarla AYNI kurallarla hazırlandı:
#   - USD limit ve USD ağırlıklı geçmiş ortalamalar, bilgi ayında tamamlanmamış öncül / 90_m -> NaN.
# Final modeller tüm olgun veriyle kurulur; bir kohortun gerekli girdisi henüz yoksa tahmini boş kalır.
URETIM, FORMUL = {}, {}
gelecek = d[d["BOLGE"] == "TAHMIN"].sort_values("AY")
TAHMIN = gelecek[["DONEM_1", "AY", "GOZLENEN_AY"]].reset_index(drop=True)

# Girdi durumu: hangi kohortta hangi model girdisi hazır?
GIRDI_DURUM = gelecek[["DONEM_1", "GOZLENEN_AY"]].reset_index(drop=True)
for c in dict.fromkeys(c for s_ in MODELLER.values() for c in taban(s_["kolonlar"])):
    GIRDI_DURUM[kisa_ad(c)] = np.where(gelecek[c].notna().to_numpy(), "✓", "–")
print(f"Olgunlaşmamış kohort: {len(gelecek)} ({gelecek['AY'].min()} – {gelecek['AY'].max()}) | bilgi ayı {BILGI}")
display(GIRDI_DURUM.style.hide(axis="index"))

for rol, spec in MODELLER.items():
    kol_ = spec["kolonlar"]
    uret_df = olgun_df.dropna(subset=kol_)
    m = pipe_fit(model_fabrika(spec["aile"], spec["params"]), uret_df[kol_], uret_df[HEDEF], agirlik_al(uret_df))
    URETIM[rol] = m
    if hasattr(m[-1], "coef_"):
        coef = np.ravel(m[-1].coef_) / m[0].scale_
        b0 = float(np.ravel([m[-1].intercept_])[0] - np.dot(coef, m[0].mean_))
        FORMUL[rol] = f"{HK} = {b0:.6f} " + " ".join(f"{c:+.6g}·{kisa_ad(k)}" for k, c in zip(kol_, coef))
    X = gelecek[kol_]
    ok = X.notna().all(axis=1).to_numpy()
    ham = np.full(len(X), np.nan)
    if ok.any():
        ham[ok] = m.predict(X[ok])
    ov = OVERLAY[rol]["deger"] if OVERLAY[rol]["uygula"] else 0.0
    TAHMIN[f"{rol}_HAM"] = np.clip(ham, 0, 1)
    TAHMIN[f"{rol}_DUZ"] = np.clip(ham - ov, 0, 1)
    print(f"{spec['ad']} | {int(ok.sum())}/{len(X)} kohort tahmin edildi | overlay "
          f"{ov * 100:+.3f} yp{'' if ov else ' (uygulanmıyor)'}")

# Chain-ladder (modelden bağımsız): bilinen en geç 90_m / olgunlaşma oranı (son 12 olgun kohort)
OLGUNLASMA, OLG_STD = {}, {}
for k in K90:
    m_ = int(k.split("_")[3])
    r_ = (olgun_df[k] / olgun_df[HEDEF]).dropna().tail(12)
    if len(r_) >= 6:
        OLGUNLASMA[m_], OLG_STD[m_] = float(r_.mean()), float(r_.std())
cl, kay = np.full(len(gelecek), np.nan), [None] * len(gelecek)
for j, (_, r_) in enumerate(gelecek.iterrows()):
    for m_ in sorted(OLGUNLASMA, reverse=True):
        v = r_.get(oncul_kolon(f"90_{m_}"))
        if pd.notna(v):
            cl[j], kay[j] = v / OLGUNLASMA[m_], f"90_{m_}"
            break
TAHMIN["CHAIN_LADDER"], TAHMIN["CL_KAYNAK"] = cl, kay
if OLGUNLASMA:
    print("Olgunlaşma oranı (90_m / 90_12, son 12 olgun kohort):",
          {f"90_{m_}": f"{OLGUNLASMA[m_]:.3f} ± {OLG_STD[m_]:.3f}" for m_ in sorted(OLGUNLASMA)})
else:
    print("Bilgi: olgun kohortlarda 90_m kolonu yok -> chain-ladder üretilmedi.")
PCT_KOL = [c for c in TAHMIN.columns if c.endswith(("_HAM", "_DUZ")) or c == "CHAIN_LADDER"]
display(TAHMIN.drop(columns="AY").style.format({**{c: "{:.2%}" for c in PCT_KOL}, "DONEM_1": "{:.0f}"},
                                               decimal=",", na_rep="-"))


# %% [16] Sunum görselleri (tahminler Bölüm 15'ten)
def tahmin_tablosu():
    """Sunum için: olgunlaşmamış kohortların tahmin tablosu (görsel)."""
    if not len(TAHMIN):
        print("Olgunlaşmamış kohort yok -> tahmin tablosu yok.")
        return
    kolonlar = [("DONEM_1", "Kohort"), ("GOZLENEN_AY", "Gözlenen ay")]
    for rol in MODELLER:
        kolonlar += [(f"{rol}_HAM", f"{rol.title()} ham"), (f"{rol}_DUZ", f"{rol.title()} overlay")]
    if "CHAIN_LADDER" in TAHMIN:
        kolonlar += [("CHAIN_LADDER", "Chain-ladder"), ("CL_KAYNAK", "CL kaynak")]
    hucre = []
    for _, r in TAHMIN.iterrows():
        satir = []
        for k, _a in kolonlar:
            v = r[k]
            satir.append(str(int(v)) if k in ("DONEM_1", "GOZLENEN_AY") else
                         (v if isinstance(v, str) else "-") if k == "CL_KAYNAK" else yuzde(v))
        hucre.append(satir)
    fig, ax = plt.subplots(figsize=(16, 0.42 * len(hucre) + 1.4)); ax.axis("off")
    tbl = ax.table(cellText=hucre, colLabels=[a for _, a in kolonlar], cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)
    for (i, j), c in tbl.get_celld().items():
        c.set_edgecolor("#e5e7eb")
        if i == 0:
            c.set_facecolor("#1f2937"); c.set_text_props(color="white", fontweight="bold")
        else:
            c.set_facecolor(ARKA["TAHMIN"] if i % 2 else "#ffffff")
    ov_ = " | ".join(f"{r}: overlay {o['deger'] * 100:+.3f} yp ({'uygulanıyor' if o['uygula'] else 'yok'})"
                     for r, o in OVERLAY.items())
    ax.set_title(f"{HK} tahminleri | bilgi ayı {BILGI} | {ov_}", fontweight="bold", loc="left")
    plt.show()


def bolge_boya(ax, alanlar):
    for ad, (p0, p1) in alanlar.items():
        x0, x1 = ts(p0) - pd.Timedelta(days=15), ts(p1) + pd.Timedelta(days=15)
        ax.axvspan(x0, x1, color=ARKA[ad], zorder=0)
        ax.text(x0 + (x1 - x0) / 2, 1.01, {"EGITIM": "EĞİTİM", "VALIDASYON": "VALİDASYON (CV)", "OOT": "OOT",
                                           "TAHMIN": "TAHMİN"}[ad], transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=RENK[ad] if ad != "EGITIM" else "#475569")


def ana_grafik(rol):
    spec, w = MODELLER[rol], WF[rol]
    kol = spec["kolonlar"]
    egit = olgun_df[olgun_df["AY"] < VAL_BAS].dropna(subset=kol)
    m_eg = pipe_fit(model_fabrika(spec["aile"], spec["params"]), egit[kol], egit[HEDEF])
    seg = {"EGITIM": pd.DataFrame({"AY": egit["AY"], "GERCEK": egit[HEDEF], "TAHMIN": m_eg.predict(egit[kol])}),
           "VALIDASYON": w[(w["BOLGE"] == "GELISTIRME") & (w["AY"] >= VAL_BAS)],
           "OOT": w[w["BOLGE"] == "OOT"]}
    alanlar = {k: (v["AY"].min(), v["AY"].max()) for k, v in seg.items()}
    if len(TAHMIN):
        alanlar["TAHMIN"] = (TAHMIN["AY"].min(), TAHMIN["AY"].max())

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.25], hspace=0.12)
    ax = fig.add_subplot(gs[0])
    bolge_boya(ax, alanlar)
    hist = olgun_df.sort_values("AY")
    ax.plot(hist["AY"].dt.to_timestamp(), hist[HEDEF] * 100, "o-", color=RENK["GERCEK"], lw=2.2, ms=4,
            label="Gerçekleşen", zorder=5)
    etiket = {"EGITIM": "Eğitim uyumu (in-sample)", "VALIDASYON": "Validasyon (walk-forward)",
              "OOT": "OOT (walk-forward)"}
    for k, s in seg.items():
        ax.plot(s["AY"].dt.to_timestamp(), s["TAHMIN"] * 100, "o-" if k != "EGITIM" else "-",
                color=RENK[k], lw=2, ms=4, label=etiket[k], zorder=4)
    if len(TAHMIN):
        tx = TAHMIN["AY"].dt.to_timestamp()
        ax.plot(tx, TAHMIN[f"{rol}_DUZ"] * 100, "D-", color=RENK["TAHMIN"], lw=2.4, ms=6,
                label="Tahmin (overlay sonrası)", zorder=6)
        ax.plot(tx, TAHMIN[f"{rol}_HAM"] * 100, "D--", color=RENK["TAHMIN"], lw=1.2, ms=4, alpha=0.6,
                label="Tahmin (ham)", zorder=6)
        if "CHAIN_LADDER" in TAHMIN:
            ax.plot(tx, TAHMIN["CHAIN_LADDER"] * 100, "s:", color=RENK["KISMI"], lw=1.4, ms=5,
                    label="Chain-ladder (90_m'den, modelden bağımsız)", zorder=5)
    ax.yaxis.set_major_formatter(PCT_AX)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.25); ax.set_ylabel(f"{HK} (12 ayda 90+ gecikme oranı)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.07), fontsize=8.5, ncol=4, frameon=False)
    ax.set_title(f"{spec['ad']}\n{FORMUL.get(rol, '')}", fontsize=11.5, fontweight="bold", loc="left", pad=62)

    axt = fig.add_subplot(gs[1]); axt.axis("off")
    satirlar, renkler = [], []
    for k in ["EGITIM", "VALIDASYON", "OOT"]:
        s = seg[k]
        mm = metrik(s["GERCEK"], s["TAHMIN"])
        dm = metrik(s["GERCEK"], s["DUMMY"]) if "DUMMY" in s else {}
        satirlar.append([{"EGITIM": "Eğitim (in-sample)", "VALIDASYON": "Validasyon (walk-forward)",
                          "OOT": "OOT (walk-forward)"}[k], f"{alanlar[k][0]} – {alanlar[k][1]}", mm["N"],
                         sayi(mm["WMAPE"]), sayi(mm["MAE_yp"], 3), f"{mm['SAPMA_yp']:+.3f}".replace(".", ","),
                         sayi(mm["R2"], 3), sayi(mm["KORELASYON"], 3), mm["YON"],
                         sayi(1 - mm["WMAPE"] / dm["WMAPE"], 2) if dm else "-"])
        renkler.append(ARKA[k])
    if len(TAHMIN) and TAHMIN[f"{rol}_HAM"].notna().any():
        t_ = TAHMIN.dropna(subset=[f"{rol}_HAM"])
        satirlar.append(["Tahmin (olgunlaşmamış)", f"{t_['AY'].min()} – {t_['AY'].max()}", len(t_),
                         "-", "-", f"ham ort. {yuzde(t_[f'{rol}_HAM'].mean())}",
                         f"ovl {yuzde(t_[f'{rol}_DUZ'].mean())}",
                         f"CL {yuzde(t_['CHAIN_LADDER'].mean()) if 'CHAIN_LADDER' in t_ else '-'}", "-", "-"])
        renkler.append(ARKA["TAHMIN"])
    tbl = axt.table(cellText=satirlar, colLabels=["Bölge", "Kohortlar", "N", "WMAPE", "MAE (yp)", "Sapma (yp)",
                                                 "R²", "Korelasyon", "Yön isabeti", "Beceri*"],
                    cellLoc="center", loc="center", colWidths=[.17, .15, .05, .07, .08, .12, .07, .09, .09, .08])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.6)
    for (i, j), c in tbl.get_celld().items():
        c.set_edgecolor("#e5e7eb")
        if i == 0:
            c.set_facecolor("#1f2937"); c.set_text_props(color="white", fontweight="bold")
        else:
            c.set_facecolor(renkler[i - 1])
    axt.text(0, -0.02, "* Beceri = 1 − WMAPE / naif ortalama WMAPE. yp = yüzde puan. Walk-forward: her kohort, kendi "
                       "tahmin anında hedefi belli olan kohortlarla eğitilen modelle tahmin edilir.",
             transform=axt.transAxes, fontsize=8, color="#6b7280")
    plt.show()


def model_karsilastirma():
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.3))
    p = PERF.set_index(["MODEL", "BOLGE"])
    x = np.arange(2)
    renkler = {"CHAMPION": RENK["TAHMIN"], "CHALLENGER": RENK["CHALLENGER"]}
    roller = list(MODELLER)
    gen = 0.8 / (len(roller) + 1)
    for i, rol in enumerate(roller + ["DUMMY"]):
        v = [p.loc[(roller[0], b), "DUMMY_WMAPE"] if rol == "DUMMY" else p.loc[(rol, b), "WMAPE"]
             for b in ["VALIDASYON", "OOT"]]
        bars = axs[0].bar(x + (i - len(roller) / 2) * gen, v, gen, color=renkler.get(rol, RENK["DUMMY"]),
                          label="Naif ortalama" if rol == "DUMMY" else rol.replace("_", " ").title())
        axs[0].bar_label(bars, labels=[sayi(t, 1) for t in v], fontsize=8)
    axs[0].set_xticks(x, ["Validasyon", "OOT"]); axs[0].set_title("WMAPE (düşük = iyi)", fontweight="bold")
    axs[0].legend(fontsize=8)
    for j, (t, baslik, vurgu) in enumerate([(OZET_VARYANT, "Takvim etkisi", "Yok"),
                                             (OZET_BILGI_AY, "Öncül olgunluğu (ay)", CHAMP["bilgi_ay"])], 1):
        t = t.dropna(subset=["EN_IYI_CV_WMAPE"])
        renk = [RENK["TAHMIN"] if str(k) == str(vurgu) else "#cbd5e1" for k in t.index]
        bars = axs[j].bar([str(k) for k in t.index], t["EN_IYI_CV_WMAPE"], color=renk)
        axs[j].bar_label(bars, labels=[sayi(v) for v in t["EN_IYI_CV_WMAPE"]], fontsize=8)
        axs[j].set_title(f"{baslik}: en iyi CV WMAPE", fontweight="bold")
        axs[j].set_ylim(0, t["EN_IYI_CV_WMAPE"].max() * 1.2)
    for a in axs:
        a.grid(axis="y", alpha=0.25)
    plt.tight_layout(); plt.show()


def donusum_grafigi():
    if not oran_onc:
        return
    fig, ax = plt.subplots(figsize=(16, 4))
    ref_ = DONUSUM.loc[DONUSUM.index < str(VAL_BAS.year + 1)]
    for c, renk in zip(oran_onc, [RENK["TAHMIN"], RENK["OOT"], RENK["CHALLENGER"], RENK["VALIDASYON"]]):
        kol_ = f"HEDEF/{kisa_ad(c)}"
        endeks = DONUSUM[kol_] / ref_[kol_].mean() * 100
        ax.plot(DONUSUM.index, endeks, "o-", lw=2.2, color=renk, label=f"{HK} / {kisa_ad(c)}")
        for xx, v in zip(DONUSUM.index, endeks):
            ax.annotate(f"{v:.0f}", (xx, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=renk)
    ax.axhline(100, color="#9ca3af", ls="--", lw=1)
    ax.set_title(f"Erken gecikmenin batığa dönüşüm oranı (endeks, {DONUSUM.index[0][:4]}–{VAL_BAS.year} ort. = 100)",
                 fontweight="bold", loc="left")
    ax.grid(alpha=0.25); ax.legend(fontsize=9)
    plt.tight_layout(); plt.show()


takvim_grafigi(CHAMP["bilgi_ay"])
for rol in MODELLER:
    ana_grafik(rol)
tahmin_tablosu()
model_karsilastirma()
donusum_grafigi()


# %% [17] Katsayılar, tahmin kaydı ve sunum özeti
KATSAYI = []
for rol, spec in MODELLER.items():
    m, kol = URETIM[rol], spec["kolonlar"]
    reg, sc = m[-1], m[0]
    if hasattr(reg, "coef_"):
        coef = np.ravel(reg.coef_) / sc.scale_
        b0 = float(np.ravel([reg.intercept_])[0] - np.dot(coef, sc.mean_))
        for k, c in zip(kol, coef):
            KATSAYI.append({"MODEL": rol, "DEGISKEN": kisa_ad(k), "KATSAYI_HAM": c,
                            "1_STD_ETKI_yp": c * olgun_df[k].std() * 100})
        KATSAYI.append({"MODEL": rol, "DEGISKEN": "Sabit", "KATSAYI_HAM": b0, "1_STD_ETKI_yp": np.nan})
    print(f"{spec['ad']} | N={len(olgun_df.dropna(subset=kol))}\n  {FORMUL.get(rol, 'formül yok')}")
KATSAYI = pd.DataFrame(KATSAYI)
display(KATSAYI.style.format({"KATSAYI_HAM": "{:.6g}", "1_STD_ETKI_yp": "{:+.3f}"}, decimal=",", na_rep="-"))

KAYIT = TAHMIN.drop(columns="AY").copy()
KAYIT.insert(0, "KAYIT_TARIHI", pd.Timestamp.today().normalize())
KAYIT.insert(1, "BILGI_AYI", CONFIG["BILGI_AYI"])
KAYIT.insert(2, "CHAMPION", CHAMP["ad"])
KAYIT.insert(3, "CHALLENGER", CHAL["ad"])
# dataiku.Dataset("L12M90_TAHMIN_KAYIT").write_with_schema(KAYIT)   # izleme için (aylık)

p = PERF.set_index(["MODEL", "BOLGE"])
print("=" * 100)
print(f"L12M90 ERKEN UYARI MODELİ v9 (USD) | hedef {HEDEF} | bilgi ayı {BILGI}")
print("=" * 100)
print(f"Veri          : {d['AY'].min()} – {d['AY'].max()} ({len(d)} kohort) | olgun {len(olgun_df)} | tahmin {len(TAHMIN)}")
for b, r_ in VERI_OZET.dropna(subset=["N"]).iterrows():
    print(f"  {b:<11}: {r_['BAS']} – {r_['SON']} | N={int(r_['N'])} | hedef ort. {yuzde(r_['HEDEF_ORT'])}")
print(f"Limit         : USD ({LIMIT_TL} / {KUR_KOL}); geçmiş ortalama ağırlığı ve ORTALAMA_LIMIT USD")
print(f"Validasyon    : {VAL_BAS} – {AY_CV.max()} ({CONFIG['N_SPLITS']} fold x {CONFIG['TEST_SIZE']} kohort, "
      f"olgunlaşma boşluklu walk-forward)")
print(f"Aday havuzu   : {len(HAM_HAVUZ)} ham + {len(ORTALAMA_KOL)} ortalama | {len(ADAY_TABLO):,} aday | "
      f"{len(grid):,} değerlendirme | geçerli {int(grid['GECERLI'].sum()):,}")
print("Takvim        : " + " | ".join(f"{k}: {sayi(v)}" for k, v in OZET_VARYANT["EN_IYI_CV_WMAPE"].items()))
print("Bilgi ayı     : " + " | ".join(f"{k}. ay: {sayi(v)}" for k, v in OZET_BILGI_AY["EN_IYI_CV_WMAPE"].dropna().items()))
print(f"Naif referans : CV WMAPE {sayi(DUMMY['CV_WMAPE'])}")
for rol, spec in MODELLER.items():
    print("-" * 100)
    print(spec["ad"] + (f"  [kaynak: {CHAL_KAYNAK}]" if rol == "CHALLENGER" else ""))
    print(f"  Formül      : {FORMUL.get(rol, '-')}")
    for b in ["VALIDASYON", "OOT"]:
        q = p.loc[(rol, b)]
        print(f"  {b:<11}: WMAPE {sayi(q['WMAPE'])} | sapma {q['SAPMA_yp']:+.3f} yp | korelasyon {sayi(q['KORELASYON'], 3)} "
              f"| yön {q['YON']} | beceri {sayi(q['BECERI'])} (naif {sayi(q['DUMMY_WMAPE'])})")
    o = OVERLAY[rol]
    print(f"  Overlay     : {o['deger'] * 100:+.3f} yp | {o['ayni_isaret']}/{o['n']} aynı işaret | "
          f"{'uygulanıyor' if o['uygula'] else 'uygulanmıyor'}")
    print(f"  Testler     : " + " | ".join(f"{t.split(' (')[0]} {TEST_KARSI.loc[t, rol]}" for t in
                                       ["HAC p (tüm olgun veri, OOT dahil)", "Durbin-Watson", "Jarque-Bera p", "VIF"]))
if len(TAHMIN):
    print("-" * 100)
    print("Tahminler (champion ham -> overlay | challenger ham -> overlay | chain-ladder):")
    for _, t in TAHMIN.iterrows():
        print(f"  {t['DONEM_1']}: {yuzde(t['CHAMPION_HAM'])} -> {yuzde(t['CHAMPION_DUZ'])} | "
              f"{yuzde(t['CHALLENGER_HAM'])} -> {yuzde(t['CHALLENGER_DUZ'])} | CL {yuzde(t.get('CHAIN_LADDER', np.nan))}")
print("=" * 100)


# %% [18A] İnce ayar altyapısı: zaman ağırlığı, ağırlıklı hızlı motor, seçim ve walk-forward fonksiyonları
# İSTEĞE BAĞLI (sağlamlık analizi). Bölüm 1–17'ye DOKUNMAZ, onların değişkenlerinin üzerine yazmaz (tüm yeni nesneler IA_ / ia_ önekli).
# Bu hücre sadece altyapıyı kurar ve 3 kontrol yapar:
#   K1) Ağırlıklı hızlı motor = sklearn (sample_weight ile)          -> fark ~0 olmalı
#   K2) ia_secim(2024-09, 5 fold, ağırlıksız) = Bölüm 12 seçimi       -> aynı champion / challenger
#   K3) ia_wf(ağırlıksız) = Bölüm 13 OOT                             -> 23,98 / 15,41
# Deneyler (yarı ömür, birleştirme, log-log, BF) 18B'den itibaren bu fonksiyonlarla yapılır.
import time as _time

IA = {
    "AILELER": ["Linear", "Ridge", "Lasso", "ElasticNet"],
    "BT_ORIJINLER": ["2023-09", "2024-03", "2024-09"],    # backtest: geliştirme sonu (seçim bu tarihe kadar)
    "BT_N_SPLITS": 3, "BT_TEST_SIZE": 4,                  # backtest validasyonu: son 12 kohort (3 x 4)
    "BT_TEST_N": 12,                                      # her orijinden sonraki 12 kohort = o orijinin testi
}
IA_GEC = CONFIG["RAPOR_GECIKME_AY"]
IA_OLG = CONFIG["HEDEF_OLGUNLASMA_AY"]


# ---------- 1) Zaman ağırlığı (gözlem ağırlığı; değişkenlere değil kohortlara) ----------
def zaman_agirligi(aylar, yari_omur):
    """Eğitimdeki en yeni kohort = 1; yari_omur ay önceki = 0,5. Ortalama 1'e ölçeklenir. None = eşit ağırlık."""
    if yari_omur is None:
        return None
    a = pd.PeriodIndex(aylar, freq="M")
    uzaklik = (a.max() - a).map(lambda x: x.n).to_numpy(float)
    w = 0.5 ** (uzaklik / yari_omur)
    return w / w.mean()


# ---------- 2) Ağırlıklı hızlı motor (pipe_fit(estimator, X, y, w) ile birebir aynı çözüm) ----------
def _ia_coz(Xtr, ytr, w, aile, alphalar=None, l1=None):
    mx, sx = Xtr.mean(axis=0), Xtr.std(axis=0)
    sx = np.where(sx > 0, sx, 1.0)                                   # StandardScaler (ağırlıksız, pipe_fit gibi)
    Xs = (Xtr - mx) / sx
    w = np.ones(len(ytr)) if w is None else np.asarray(w, float)
    xw, yw = (w @ Xs) / w.sum(), (w @ ytr) / w.sum()                  # ağırlıklı merkezleme (sklearn ile aynı)
    sw = np.sqrt(w)
    Xc, yc = (Xs - xw) * sw[:, None], (ytr - yw) * sw
    if aile == "Linear":
        B = np.linalg.lstsq(Xc, yc, rcond=None)[0][None, :]
    elif aile == "Ridge":
        U, s_, Vt = np.linalg.svd(Xc, full_matrices=False)
        Uy = U.T @ yc
        B = np.stack([Vt.T @ (s_ / (s_ ** 2 + a) * Uy) for a in alphalar])
    else:
        Xf = np.asfortranarray(Xc)
        _, coefs, _ = enet_path(Xf, yc, l1_ratio=1.0 if aile == "Lasso" else l1, alphas=alphalar,
                                max_iter=CONFIG["MAX_ITER"], tol=CONFIG["MODEL_TOL"], check_input=False,
                                precompute=Xf.T @ Xf, Xy=Xf.T @ yc)
        B = coefs.T
    return mx, sx, B, yw - B @ xw


def _ia_varyantlar(aile):
    if aile == "Linear":
        return [(None, None)]
    if aile == "Ridge":
        return [(np.sort(np.asarray(CONFIG["ALPHA_RIDGE"], float))[::-1], None)]
    a = np.sort(np.asarray(CONFIG["ALPHA_LASSO"], float))[::-1]
    return [(a, None)] if aile == "Lasso" else [(a, l1) for l1 in CONFIG["L1_ORANLARI"]]


def ia_metrik(kayitlar, y, ayno):
    """Bölüm 5 metrik_ozet ile aynı formüller; y ve ay numaraları parametre (farklı geliştirme setleri için)."""
    hata, gercek, f_w, f_s, f_o, d_g, d_t, ad = [], [], [], [], [], [], [], 0
    for tr, val, p, p_tr in kayitlar:
        f_w.append(wmape(y[val], p)); f_s.append(float(np.mean(p - y[val])))
        f_o.append(f_w[-1] / max(wmape(y[tr], p_tr), 1e-12))
        ad += int(((p < 0) | (p > 2 * y[tr].max())).sum())
        bit = np.diff(ayno[val]) == 1
        d_g.extend(np.diff(y[val])[bit]); d_t.extend(np.diff(p)[bit])
        hata.extend(p - y[val]); gercek.extend(y[val])
    hata, gercek, d_g, d_t = map(np.asarray, (hata, gercek, d_g, d_t))
    anl = np.abs(d_g) > TOL
    dogru = np.sign(np.where(np.abs(d_t) <= TOL, 0, d_t))[anl] == np.sign(d_g[anl])
    buy = np.abs(d_g[anl])
    return {"CV_WMAPE": float(np.abs(hata).sum() / gercek.sum() * 100), "CV_SAPMA": float(np.mean(hata)),
            "MUTLAK_FOLD_SAPMA": float(np.mean(np.abs(f_s))), "EN_KOTU_WMAPE": float(np.max(f_w)),
            "FOLD_WMAPE_STD": float(np.std(f_w)), "OVERFIT_ORANI": float(np.median(f_o)),
            "DELTA_MAE": float(np.mean(np.abs(d_t - d_g))) if len(d_g) else np.nan,
            "YON_AGIRLIKLI": float((buy * dogru).sum() / buy.sum()) if anl.any() else np.nan,
            "ARALIK_DISI": ad, "FOLD1_EGITIM_N": len(kayitlar[0][0])}


# ---------- 3) Herhangi bir geliştirme seti için fold'lar ve grid ----------
def ia_cv_tablo(dev_son):
    """Olgun kohortlardan AY <= dev_son olanlar (point-in-time girdiler Bölüm 3-4'ten, d'de hazır)."""
    t = d[d["HEDEF_KULLANILABILIR"] & (d["AY"] <= pd.Period(dev_son, "M"))]
    return t.sort_values("AY").reset_index(drop=True)


def ia_foldlar(cv_t, b, n_splits, test_size):
    ay = pd.PeriodIndex(cv_t["AY"])
    out = []
    for tr, val in TimeSeriesSplit(n_splits=n_splits, test_size=test_size).split(np.arange(len(cv_t))):
        out.append((tr[(ay[tr] + IA_OLG + IA_GEC) < tahmin_ayi_hesapla(ay[val[0]], b)], val))
    return out


def ia_grid(cv_t, n_splits, test_size, yari_omur=None, kombs=None, aileler=None):
    """Bölüm 9'un aynısı (trend hariç, çünkü seçilemez), farklı geliştirme seti + opsiyonel zaman ağırlığı."""
    kombs = kombs if kombs is not None else [k for k in KOMBS if TREND_KOL not in k]
    aileler = aileler or IA["AILELER"]
    y, ay = cv_t[HEDEF].to_numpy(float), pd.PeriodIndex(cv_t["AY"])
    ayno = ay.asi8
    F = {b: ia_foldlar(cv_t, b, n_splits, test_size) for b in range(0, CONFIG["CHAMPION_MAX_AY"] + 1)}
    W = {b: [zaman_agirligi(ay[tr], yari_omur) for tr, _ in F[b]] for b in F}
    satirlar = []
    for kol in kombs:
        kol = list(kol); b = bilgi_ayi_hesapla(kol)
        X = cv_t[kol].to_numpy(float)
        dolu = ~np.isnan(X).any(axis=1)
        hazir = []
        for (tr, val), w in zip(F[b], W[b]):
            m = dolu[tr]
            if m.sum() < MIN_TRAIN or not dolu[val].all():
                hazir = None; break
            ww = None if w is None else w[m] / w[m].mean()
            hazir.append((tr[m], val, ww))
        if hazir is None:
            continue
        for aile in aileler:
            for alphalar, l1 in _ia_varyantlar(aile):
                n_p = 1 if alphalar is None else len(alphalar)
                fold_P = []
                for tr, val, ww in hazir:
                    mx, sx, B, b0 = _ia_coz(X[tr], y[tr], ww, aile, alphalar, l1)
                    fold_P.append((tr, val, ((X[val] - mx) / sx) @ B.T + b0, ((X[tr] - mx) / sx) @ B.T + b0))
                for j in range(n_p):
                    s = ia_metrik([(tr, val, P[:, j], Pt[:, j]) for tr, val, P, Pt in fold_P], y, ayno)
                    p = None if alphalar is None else (
                        {"alpha": float(alphalar[j])} if l1 is None else {"alpha": float(alphalar[j]), "l1": float(l1)})
                    s.update({"GIRDILER": ", ".join(kisa_ad(c) for c in taban(kol)), "MODEL": aile, "PARAMS": p,
                              "BILGI_AY": b, "TOPLAM_K": len(kol), "MEVSIM": int("SIN_AY" in kol),
                              "ORTALAMA": next((f"AGORT{ORTALAMA_PENCERE[c]}" for c in kol if c in ORTALAMA_KOL), "Yok"),
                              "KOLONLAR": tuple(kol)})
                    satirlar.append(s)
    g = pd.DataFrame(satirlar)
    # Naif referans: aynı fold'larda (bilgi ayı 0) eğitim ortalaması
    kay = [(tr, val, np.full(len(val), y[tr].mean()), np.full(len(tr), y[tr].mean())) for tr, val in F[0]]
    dummy = ia_metrik(kay, y, ayno)
    gc = CONFIG["GECERLI"]
    g["GECERLI"] = ((g["CV_WMAPE"] <= gc["CV_WMAPE_DUMMY_ORAN"] * dummy["CV_WMAPE"])
                    & (g["EN_KOTU_WMAPE"] <= gc["EN_KOTU_DUMMY_ORAN"] * dummy["EN_KOTU_WMAPE"])
                    & (g["OVERFIT_ORANI"] <= gc["OVERFIT_ORANI_MAX"])
                    & (g["MUTLAK_FOLD_SAPMA"] <= gc["SAPMA_DUMMY_ORAN"] * dummy["MUTLAK_FOLD_SAPMA"])
                    & (g["ARALIK_DISI"] == 0))
    g["RANK_SKORU"] = np.nan
    if g["GECERLI"].any():
        m, skor = g["GECERLI"], 0.0
        for k_, w_ in CONFIG["AGIRLIK"].items():
            s_ = g.loc[m, k_].astype(float)
            s_ = s_.clip(lower=1.0) if k_ == "OVERFIT_ORANI" else s_
            skor = skor + w_ * s_.rank(pct=True).fillna(1.0)
        g.loc[m, "RANK_SKORU"] = skor
    return g, dummy


# ---------- 4) Seçim (Bölüm 11-12'nin aynısı, parametrik) ----------
_IA_HAC = {}
def ia_hac_p(kol, cv_t, yari_omur=None):
    anahtar = (tuple(kol), str(cv_t["AY"].max()), len(cv_t), yari_omur)
    if anahtar not in _IA_HAC:
        t = cv_t.dropna(subset=list(kol))
        X = sm.add_constant(t[list(kol)].astype(float).to_numpy(), has_constant="add")
        yv = t[HEDEF].astype(float).to_numpy()
        lag = int(np.ceil(4 * (len(yv) / 100) ** (2 / 9)))
        w = zaman_agirligi(t["AY"], yari_omur)
        mod = sm.OLS(yv, X) if w is None else sm.WLS(yv, X, weights=w)
        _IA_HAC[anahtar] = float(np.max(mod.fit(cov_type="HAC", cov_kwds={"maxlags": lag}).pvalues[1:]))
    return _IA_HAC[anahtar]


def _ia_kisa(havuz, cv_t, n_splits, yari_omur):
    h = havuz.nsmallest(CONFIG["HAC_ILK_N"], "CV_WMAPE").drop_duplicates(["KOLONLAR", "MODEL"]).copy()
    h["HAC_MAKS_P"] = h["KOLONLAR"].apply(lambda k: ia_hac_p(k, cv_t, yari_omur))
    a = h[h["HAC_MAKS_P"] <= CONFIG["HAC_P_ESIK"]]
    a = h if a.empty else a
    en_iyi = a.nsmallest(1, "CV_WMAPE").iloc[0]
    esik = en_iyi["CV_WMAPE"] + en_iyi["FOLD_WMAPE_STD"] / np.sqrt(n_splits)
    k_ = a[a["CV_WMAPE"] <= esik].copy()
    k_["AILE_SIRA"] = k_["MODEL"].map(AILE_SIRA)
    return k_.sort_values(["TOPLAM_K", "BILGI_AY", "AILE_SIRA", "RANK_SKORU"]).reset_index(drop=True), h


def ia_secim(g, cv_t, n_splits, yari_omur=None):
    """Bölüm 11-12 kuralı: geçerli -> HAC -> 1-SE -> sadelik; challenger <= 3. ay ve farklı ham girdi (yedek yollu)."""
    aday = g[g["GECERLI"]]
    erken = aday[aday["BILGI_AY"] <= CONFIG["CHALLENGER_MAX_AY"]]
    kisa, _ = _ia_kisa(aday, cv_t, n_splits, yari_omur)
    champ = kisa.iloc[0]
    hs = lambda k: {c for c in k if c in HAM_HAVUZ}
    farkli = lambda df: df[df["KOLONLAR"].apply(hs) != hs(champ["KOLONLAR"])]
    kisa_c, h_c = _ia_kisa(erken, cv_t, n_splits, yari_omur)
    ch, kaynak = farkli(kisa_c), "1-SE + anlamlılık"
    if ch.empty:
        ch = farkli(h_c[h_c["HAC_MAKS_P"] <= CONFIG["HAC_P_ESIK"]]).sort_values("CV_WMAPE")
        kaynak = "anlamlı erken modeller (1-SE dışı)"
    if ch.empty:
        ch, kaynak = kisa_c, "1-SE listesi (aynı ham girdi)"
    spec = lambda r, rol: {"rol": rol, "kolonlar": list(r["KOLONLAR"]), "aile": r["MODEL"], "params": r["PARAMS"],
                           "bilgi_ay": int(r["BILGI_AY"]), "girdiler": r["GIRDILER"], "CV_WMAPE": r["CV_WMAPE"]}
    return {"CHAMPION": spec(champ, "CHAMPION"), "CHALLENGER": spec(ch.iloc[0], "CHALLENGER"), "CHAL_KAYNAK": kaynak}


# ---------- 5) Walk-forward tahmin (Bölüm 13 wf_tahmin'in ağırlıklı hali) ----------
def ia_wf(spec, yari_omur=None, aylar=None):
    """Her kohort KENDİ tahmin anında, o gün hedefi belli kohortlarla (ve opsiyonel zaman ağırlığıyla) eğitilir."""
    kol, kayit = spec["kolonlar"], []
    hedefler = olgun_df if aylar is None else olgun_df[olgun_df["AY"].isin(pd.PeriodIndex(aylar, freq="M"))]
    for i in hedefler.index:
        c = olgun_df.loc[i]
        t_ay = tahmin_ayi_hesapla(c["AY"], spec["bilgi_ay"])
        eg = olgun_df[olgun_df["HEDEF_ERISIM"] < t_ay].dropna(subset=kol)
        if len(eg) < MIN_TRAIN or c[kol].isna().any():
            continue
        m = pipe_fit(model_fabrika(spec["aile"], spec["params"]), eg[kol], eg[HEDEF], zaman_agirligi(eg["AY"], yari_omur))
        kayit.append({"AY": c["AY"], "BOLGE": c["BOLGE"], "GERCEK": float(c[HEDEF]),
                      "TAHMIN": float(m.predict(olgun_df.loc[[i], kol])[0]), "NAIF": float(eg[HEDEF].mean())})
    return pd.DataFrame(kayit)


def ia_ozet(w, etiket=""):
    """Bir walk-forward çıktısının metrikleri (Bölüm 13 metrik() ile aynı) + naif ve beceri."""
    if w.empty or len(w) < 3:
        return {}
    m = metrik(w["GERCEK"], w["TAHMIN"])
    nw = wmape(w["GERCEK"], w["NAIF"])
    return {"ETIKET": etiket, **m, "NAIF_WMAPE": nw, "BECERI": 1 - m["WMAPE"] / nw}


# ---------- 6) Backtest planı (her orijin: seçim <= orijin, test = sonraki 12 kohort) ----------
def ia_backtest_plani(orijinler=None, n_splits=None):
    orijinler, n_splits = orijinler or IA["BT_ORIJINLER"], n_splits or IA["BT_N_SPLITS"]
    satir = []
    for o in orijinler:
        cv_t = ia_cv_tablo(o)
        f2 = ia_foldlar(cv_t, 2, n_splits, IA["BT_TEST_SIZE"])
        ay = pd.PeriodIndex(cv_t["AY"])
        o_p = pd.Period(o, "M")
        test = olgun_df.loc[olgun_df["AY"] > o_p, "AY"].sort_values().head(IA["BT_TEST_N"])
        satir.append({"ORIJIN (gel. sonu)": o, "GELISTIRME_N": len(cv_t),
                      "VALIDASYON": f"{ay[f2[0][1][0]]} – {ay[f2[-1][1][-1]]}",
                      "FOLD1_EGITIM_2AY": f"{ay[f2[0][0][0]]} – {ay[f2[0][0][-1]]} ({len(f2[0][0])})",
                      "TEST": f"{test.min()} – {test.max()} ({len(test)})"})
    return pd.DataFrame(satir)


# =================== KONTROLLER ===================
t0_ = _time.time()

# K1) Ağırlıklı hızlı motor = sklearn (yarı ömür 12, challenger girdileri, 3. ay fold'ları)
_kol = CHAL["kolonlar"]
_cvt = ia_cv_tablo(str(AY_CV.max()))
_F = ia_foldlar(_cvt, 3, CONFIG["N_SPLITS"], CONFIG["TEST_SIZE"])
_y = _cvt[HEDEF].to_numpy(float)
_K1 = []
for aile, p_ in [("Linear", None), ("Ridge", {"alpha": 10.0}), ("Lasso", {"alpha": 1e-4}),
                 ("ElasticNet", {"alpha": 1e-4, "l1": 0.5})]:
    for yo in [None, 12]:
        P1, P2 = [], []
        for tr, val in _F:
            w = zaman_agirligi(_cvt["AY"].iloc[tr], yo)
            al = None if p_ is None else np.array([p_["alpha"]])
            mx, sx, B, b0 = _ia_coz(_cvt[_kol].to_numpy(float)[tr], _y[tr], w, aile, al, (p_ or {}).get("l1"))
            P1.extend(((_cvt[_kol].to_numpy(float)[val] - mx) / sx) @ B[0] + b0[0])
            m_ = pipe_fit(model_fabrika(aile, p_), _cvt[_kol].iloc[tr], _y[tr], w)
            P2.extend(m_.predict(_cvt[_kol].iloc[val]))
        _K1.append({"AILE": aile, "PARAMS": p_, "YARI_OMUR": yo or "yok",
                    "WMAPE_MOTOR": wmape(_y[np.concatenate([v for _, v in _F])], P1),
                    "WMAPE_SKLEARN": wmape(_y[np.concatenate([v for _, v in _F])], P2),
                    "MAKS_TAHMIN_FARKI": float(np.max(np.abs(np.array(P1) - np.array(P2))))})
_K1 = pd.DataFrame(_K1)
print("K1) Ağırlıklı hızlı motor vs sklearn (sample_weight):")
display(_K1.style.format({"WMAPE_MOTOR": "{:.6f}", "WMAPE_SKLEARN": "{:.6f}", "MAKS_TAHMIN_FARKI": "{:.1e}"}))
if _K1["MAKS_TAHMIN_FARKI"].max() > 1e-6:
    raise AssertionError("Ağırlıklı motor sklearn ile uyuşmuyor.")

# K2) a) Seçim fonksiyonu: Bölüm 9 grid'i ia_secim'e verilince Bölüm 12 seçimi birebir çıkmalı
#     b) Grid motoru: ia_grid (2024-09, 5 x 4, ağırlıksız) metrikleri Bölüm 9 ile aynı olmalı
#     Not: ia_grid trend varyantlarını (seçilemez) ve Optuna satırlarını içermez; bileşik skor (RANK_SKORU)
#     yüzdelik sıralama olduğu için havuza bağlıdır -> eşit sadelikteki adaylarda (ör. AGORT3 / AGORT6)
#     backtest seçimi Bölüm 12'den farklı çıkabilir. Bu bir hata değil, kuralın hassasiyetidir.
IA_S_ANA = ia_secim(grid[grid["SECILEBILIR"]], _cvt, CONFIG["N_SPLITS"])
IA_G0, IA_D0 = ia_grid(_cvt, CONFIG["N_SPLITS"], CONFIG["TEST_SIZE"])
IA_S0 = ia_secim(IA_G0, _cvt, CONFIG["N_SPLITS"])
_ayni = lambda a, b: a["kolonlar"] == b["kolonlar"] and a["aile"] == b["aile"] and a["params"] == b["params"]
_et = lambda s_: f"{s_['girdiler']} | {s_['aile']} | {s_['params']}"
_K2 = pd.DataFrame([{"ROL": r, "BOLUM_12": _et(s_), "IA_SECIM(Bölüm 9 grid)": _et(IA_S_ANA[r]),
                     "AYNI": _ayni(s_, IA_S_ANA[r]), "IA_SECIM(ia_grid, trendsiz havuz)": _et(IA_S0[r])}
                    for r, s_ in [("CHAMPION", CHAMP), ("CHALLENGER", CHAL)]])
_anah = lambda g: [f"{k}|{m}|{sorted(p.items()) if isinstance(p, dict) else None}"
                   for k, m, p in zip(g["KOLONLAR"], g["MODEL"], g["PARAMS"])]
_m = (IA_G0.assign(K=_anah(IA_G0)).set_index("K")[["CV_WMAPE"]]
      .join(grid.assign(K=_anah(grid)).drop_duplicates("K").set_index("K")[["CV_WMAPE"]], rsuffix="_B9", how="inner"))
_K2_FARK = float((_m["CV_WMAPE"] - _m["CV_WMAPE_B9"]).abs().max())
print(f"K2) a) Seçim fonksiyonu | b) ia_grid {len(IA_G0):,} satır, Bölüm 9 ile ortak {len(_m):,} satırda "
      f"maks CV_WMAPE farkı {_K2_FARK:.1e} | naif CV {IA_D0['CV_WMAPE']:.2f}")
display(_K2)

# K3) Ağırlıksız walk-forward = Bölüm 13 OOT
_K3 = []
for r, s_ in [("CHAMPION", CHAMP), ("CHALLENGER", CHAL)]:
    w_ = ia_wf(s_)
    _K3.append({"ROL": r, "OOT_WMAPE_IA": ia_ozet(w_[w_["BOLGE"] == "OOT"])["WMAPE"],
                "OOT_WMAPE_BOLUM13": _p.loc[(r, "OOT"), "WMAPE"]})
_K3 = pd.DataFrame(_K3)
_K3["FARK"] = (_K3["OOT_WMAPE_IA"] - _K3["OOT_WMAPE_BOLUM13"]).abs()
print("K3) Walk-forward yeniden üretimi:")
display(_K3.style.format({"OOT_WMAPE_IA": "{:.4f}", "OOT_WMAPE_BOLUM13": "{:.4f}", "FARK": "{:.1e}"}))

print("Backtest planı (18B'de kullanılacak; her orijinde seçim sadece o tarihe kadarki veriyle):")
display(ia_backtest_plani())
print(f"18A tamam | {_time.time() - t0_:.0f} sn | K1 {'OK' if _K1['MAKS_TAHMIN_FARKI'].max() < 1e-6 else 'HATA'} | "
      f"K2 {'OK' if _K2['AYNI'].all() and _K2_FARK < 1e-6 else 'FARKLI'} | K3 {'OK' if _K3['FARK'].max() < 1e-6 else 'FARKLI'}")


# %% [18B] Yakın döneme ağırlık: yarı ömür x 3 orijin backtest (seçim sürecinin TAMAMI her seferinde yeniden)
# Her orijinde: grid + seçim SADECE o tarihe kadarki veriyle; test = sonraki 12 kohort (her biri kendi tahmin anında).
# Karar kuralı (sonuçlara bakmadan önce sabit): OOT HARİÇ orijinlerin (2023-09, 2024-03) ortalama test WMAPE'si
# en düşük yarı ömür; fark < 1 puan ise daha uzun yarı ömür (daha az oynak) tercih edilir. OOT orijini sadece raporlanır.
IA["YARI_OMURLER"] = [None, 24, 18, 12]
t0_ = _time.time()
IA_BT = []
for o in IA["BT_ORIJINLER"]:
    cv_t = ia_cv_tablo(o)
    test_aylar = olgun_df.loc[olgun_df["AY"] > pd.Period(o, "M"), "AY"].sort_values().head(IA["BT_TEST_N"])
    for yo in IA["YARI_OMURLER"]:
        g, dm = ia_grid(cv_t, IA["BT_N_SPLITS"], IA["BT_TEST_SIZE"], yo)
        sec = ia_secim(g, cv_t, IA["BT_N_SPLITS"], yo)
        for rol in ["CHAMPION", "CHALLENGER"]:
            s_ = sec[rol]
            w = ia_wf(s_, yo, test_aylar)
            oz = ia_ozet(w)
            IA_BT.append({"ORIJIN": o, "TEST": f"{test_aylar.min()}–{test_aylar.max()}",
                          "YARI_OMUR": "yok" if yo is None else yo, "ROL": rol,
                          "MODEL": f"{s_['girdiler']} | {s_['aile']}{'' if not s_['params'] else ' ' + str(s_['params'])}",
                          "BILGI_AY": s_["bilgi_ay"], "CV_WMAPE": s_["CV_WMAPE"], "TEST_WMAPE": oz["WMAPE"],
                          "TEST_SAPMA_yp": oz["SAPMA_yp"], "TEST_KOR": oz["KORELASYON"], "NAIF_WMAPE": oz["NAIF_WMAPE"],
                          "BECERI": oz["BECERI"], "GECERLI_N": int(g["GECERLI"].sum())})
        print(f"  orijin {o} | yarı ömür {yo or 'yok':>3} | {_time.time() - t0_:.0f} sn")
IA_BT = pd.DataFrame(IA_BT)
IA_BT["OOT_MU"] = IA_BT["ORIJIN"].eq(IA["BT_ORIJINLER"][-1])

fmt = {"CV_WMAPE": "{:.2f}", "TEST_WMAPE": "{:.2f}", "TEST_SAPMA_yp": "{:+.3f}", "TEST_KOR": "{:.2f}",
       "NAIF_WMAPE": "{:.2f}", "BECERI": "{:+.2f}"}
print("=== Backtest detay: her orijin ve yarı ömür için seçilen modeller ve test performansı ===")
display(IA_BT.drop(columns="OOT_MU").style.format(fmt, decimal=","))


def _ozet(df):
    return df.groupby(["YARI_OMUR", "ROL"], sort=False).agg(
        ORT_TEST_WMAPE=("TEST_WMAPE", "mean"), EN_KOTU_TEST=("TEST_WMAPE", "max"),
        ORT_MUTLAK_SAPMA=("TEST_SAPMA_yp", lambda s: s.abs().mean()), ORT_BECERI=("BECERI", "mean"),
        ORT_KOR=("TEST_KOR", "mean")).reset_index()


IA_BT_OZET = _ozet(IA_BT[~IA_BT["OOT_MU"]]).merge(
    _ozet(IA_BT[IA_BT["OOT_MU"]])[["YARI_OMUR", "ROL", "ORT_TEST_WMAPE", "ORT_BECERI"]]
    .rename(columns={"ORT_TEST_WMAPE": "OOT_TEST_WMAPE", "ORT_BECERI": "OOT_BECERI"}), on=["YARI_OMUR", "ROL"])
print("=== Özet | karar sütunları: OOT HARİÇ 2 orijin ortalaması | OOT sütunları sadece bilgi ===")
display(IA_BT_OZET.style.format({c: "{:.2f}" for c in IA_BT_OZET.columns if c not in ("YARI_OMUR", "ROL")}, decimal=","))

# Karar (önceden sabit kural): rol bazında
IA_YO_KARAR = {}
for rol in ["CHAMPION", "CHALLENGER"]:
    t = IA_BT_OZET[IA_BT_OZET["ROL"] == rol].copy()
    t["UZUNLUK"] = t["YARI_OMUR"].map(lambda v: 999 if v == "yok" else v)
    en_iyi = t["ORT_TEST_WMAPE"].min()
    aday = t[t["ORT_TEST_WMAPE"] <= en_iyi + 1.0].sort_values("UZUNLUK", ascending=False)
    IA_YO_KARAR[rol] = aday.iloc[0]["YARI_OMUR"]
    print(f"{rol}: OOT hariç en iyi ort. test WMAPE {en_iyi:.2f} | 1 puan içinde en uzun yarı ömür -> "
          f"{IA_YO_KARAR[rol]} (ort. test {aday.iloc[0]['ORT_TEST_WMAPE']:.2f}, OOT {aday.iloc[0]['OOT_TEST_WMAPE']:.2f})")
print(f"18B tamam | {_time.time() - t0_:.0f} sn")


# %% [18C] Tahmin birleştirme + seviye düzeltmesi (overlay) + "60_3, MM, MM_AGORT6" ailesi | 3 orijin backtest
# 18B kararı: yarı ömür YOK (ağırlıksız). Her orijinde seçim SADECE o tarihe kadarki veriyle (3 x 4 validasyon).
# Stratejiler (her biri ham + seviye düzeltmeli):
#   CHAMPION, CHALLENGER      : o orijinde seçilen tek modeller
#   ORT_CC                    : champion + challenger ortalaması
#   ORT_1SE                   : champion 1-SE listesindeki farklı ham girdili ilk 5 modelin ortalaması
#   AILE_60_3_MM              : 60_3 + MM_SKOR + MM_SKOR_AGORT6; aile/alpha o orijinin CV'siyle seçilir.
#                               DİKKAT: aile 18B sonuçlarına bakılarak bulundu (sonradan) -> temkinli oku.
# Seviye düzeltmesi (walk-forward): her test kohortunun tahmin anında, hedefi o gün belli olan son 12 kohortun
#   ortalama hatası; Bölüm 13 kuralı (12'nin >= 10'u aynı işaret) sağlanırsa tahminden düşülür.
# Karar kuralı (önceden sabit): OOT HARİÇ 2 orijinin ort. test WMAPE'si en düşük strateji; 1 puan içinde
#   olanlardan en sade olan (tek model < ORT_CC < ORT_1SE; AILE sonradan bulunduğu için karar dışı, bilgi).
IA["ORT_1SE_N"] = 5
IA["AILE_KOL"] = [oncul_kolon("60_3"), "MM_SKOR", "MM_SKOR_AGORT6"]
IA_SADELIK = {"CHAMPION": 0, "CHALLENGER": 0, "ORT_CC": 1, "ORT_1SE": 2, "AILE_60_3_MM": 9}
t0_ = _time.time()

_IA_WF = {}
def ia_wf_cache(spec):
    """Sabit bir modelin walk-forward tahminleri orijinden bağımsızdır (her kohort kendi tahmin anında eğitilir)."""
    k = (tuple(spec["kolonlar"]), spec["aile"], str(spec["params"]))
    if k not in _IA_WF:
        _IA_WF[k] = ia_wf(spec).set_index("AY")[["GERCEK", "TAHMIN"]]
    return _IA_WF[k]


def ia_seri(specler):
    """Üyelerin walk-forward tahminlerinin ortalaması; tahmin anı = üyelerin en geç bilgi ayı."""
    parcalar = [ia_wf_cache(s).rename(columns={"TAHMIN": f"T{i}"}) for i, s in enumerate(specler)]
    s = parcalar[0][["GERCEK"]].join([p.drop(columns="GERCEK") for p in parcalar], how="inner")
    s["TAHMIN"] = s.filter(like="T").mean(axis=1)
    s = s[["GERCEK", "TAHMIN"]].reset_index()
    s["HEDEF_ERISIM"] = s["AY"] + IA_OLG + IA_GEC
    return s, max(sp["bilgi_ay"] for sp in specler)


def ia_overlay(s, bilgi_ay):
    """Walk-forward seviye düzeltmesi: sadece tahmin anında hedefi belli olan kohortların hatası kullanılır."""
    s = s.copy()
    s["ARTIK"] = s["TAHMIN"] - s["GERCEK"]
    duz = []
    for _, r in s.iterrows():
        T = tahmin_ayi_hesapla(r["AY"], bilgi_ay)
        son = s[s["HEDEF_ERISIM"] < T].sort_values("HEDEF_ERISIM").tail(CONFIG["OVERLAY_PENCERE"])
        ayni = max((son["ARTIK"] > 0).sum(), (son["ARTIK"] < 0).sum()) if len(son) else 0
        duz.append(son["ARTIK"].mean() if len(son) == CONFIG["OVERLAY_PENCERE"]
                   and ayni >= CONFIG["OVERLAY_MIN_AYNI_ISARET"] else 0.0)
    s["DUZELTME"] = duz
    s["TAHMIN_DUZ"] = s["TAHMIN"] - s["DUZELTME"]
    return s


def _pencere_metrik(s, aylar, kolon):
    t = s[s["AY"].isin(aylar)]
    m = metrik(t["GERCEK"], t[kolon])
    return {"N": len(t), "WMAPE": m["WMAPE"], "SAPMA_yp": m["SAPMA_yp"], "KOR": m["KORELASYON"],
            "DUZ_UYGULANAN": int((t["DUZELTME"] != 0).sum()) if kolon == "TAHMIN_DUZ" else 0}


IA_C, IA_C_MODELLER = [], []
for o in IA["BT_ORIJINLER"]:
    cv_t = ia_cv_tablo(o)
    test_aylar = pd.PeriodIndex(olgun_df.loc[olgun_df["AY"] > pd.Period(o, "M"), "AY"].sort_values().head(IA["BT_TEST_N"]))
    g, _ = ia_grid(cv_t, IA["BT_N_SPLITS"], IA["BT_TEST_SIZE"])
    sec = ia_secim(g, cv_t, IA["BT_N_SPLITS"])
    kisa, _ = _ia_kisa(g[g["GECERLI"]], cv_t, IA["BT_N_SPLITS"], None)
    hs = lambda k: frozenset(c for c in k if c in HAM_HAVUZ)
    uyeler = kisa.assign(HS=kisa["KOLONLAR"].apply(hs)).drop_duplicates("HS").head(IA["ORT_1SE_N"])
    sp = lambda r: {"kolonlar": list(r["KOLONLAR"]), "aile": r["MODEL"], "params": r["PARAMS"],
                    "bilgi_ay": int(r["BILGI_AY"]), "girdiler": r["GIRDILER"]}
    a_ = g[g["KOLONLAR"].apply(lambda k: k == tuple(IA["AILE_KOL"])) & g["MODEL"].isin(["Linear", "Ridge"])]
    if len(a_):                                      # aile/alpha: o orijinin CV'sine göre (point-in-time)
        aile_sp = sp(a_.sort_values(["GECERLI", "CV_WMAPE"], ascending=[False, True]).iloc[0])
    else:                                            # o orijinde fold eğitimi yetersiz -> ayarsız Linear
        aile_sp = {"kolonlar": IA["AILE_KOL"], "aile": "Linear", "params": None,
                   "bilgi_ay": bilgi_ayi_hesapla(IA["AILE_KOL"]), "girdiler": "60_3, MM_SKOR, MM_SKOR_AGORT6 (CV yok)"}
    STRATEJI = {"CHAMPION": [sec["CHAMPION"]], "CHALLENGER": [sec["CHALLENGER"]],
                "ORT_CC": [sec["CHAMPION"], sec["CHALLENGER"]],
                "ORT_1SE": [sp(r) for _, r in uyeler.iterrows()], "AILE_60_3_MM": [aile_sp]}
    for ad, specler in STRATEJI.items():
        IA_C_MODELLER.append({"ORIJIN": o, "STRATEJI": ad,
                              "UYELER": " + ".join(f"[{s_['girdiler']} | {s_['aile']}"
                                                   f"{'' if not s_['params'] else ' a=' + format(s_['params']['alpha'], '.3g')}]"
                                                   for s_ in specler)})
        s, b = ia_seri(specler)
        s = ia_overlay(s, b)
        for tur, kol in [("HAM", "TAHMIN"), ("DUZ", "TAHMIN_DUZ")]:
            IA_C.append({"ORIJIN": o, "TEST": f"{test_aylar.min()}–{test_aylar.max()}", "STRATEJI": ad, "TUR": tur,
                         "BILGI_AY": b, **_pencere_metrik(s, test_aylar, kol)})
    print(f"  orijin {o} tamam | {_time.time() - t0_:.0f} sn")
IA_C, IA_C_MODELLER = pd.DataFrame(IA_C), pd.DataFrame(IA_C_MODELLER)
IA_C["OOT_MU"] = IA_C["ORIJIN"].eq(IA["BT_ORIJINLER"][-1])

print("=== Her orijinde stratejilerin üyeleri ===")
display(IA_C_MODELLER)
print("=== Test WMAPE (satır: strateji x ham/düzeltilmiş, sütun: orijin) ===")
display(IA_C.pivot_table(index=["STRATEJI", "TUR"], columns="ORIJIN", values="WMAPE", sort=False)
        .style.format("{:.2f}", decimal=",").background_gradient(cmap="RdYlGn_r", axis=None))
print("=== Test sapması (yp) ===")
display(IA_C.pivot_table(index=["STRATEJI", "TUR"], columns="ORIJIN", values="SAPMA_yp", sort=False)
        .style.format("{:+.3f}", decimal=","))

oz = IA_C[~IA_C["OOT_MU"]].groupby(["STRATEJI", "TUR"], sort=False).agg(
    ORT_TEST_WMAPE=("WMAPE", "mean"), EN_KOTU=("WMAPE", "max"),
    ORT_MUTLAK_SAPMA=("SAPMA_yp", lambda x: x.abs().mean()), ORT_KOR=("KOR", "mean")).reset_index()
oz = oz.merge(IA_C[IA_C["OOT_MU"]][["STRATEJI", "TUR", "WMAPE", "SAPMA_yp"]]
              .rename(columns={"WMAPE": "OOT_WMAPE", "SAPMA_yp": "OOT_SAPMA"}), on=["STRATEJI", "TUR"])
oz["SADELIK"] = oz["STRATEJI"].map(IA_SADELIK) * 2 + oz["TUR"].eq("DUZ").astype(int)
IA_C_OZET = oz.sort_values("ORT_TEST_WMAPE").reset_index(drop=True)
print("=== Özet | karar: OOT HARİÇ 2 orijin | OOT sütunları sadece bilgi ===")
display(IA_C_OZET.drop(columns="SADELIK").style.format(
    {"ORT_TEST_WMAPE": "{:.2f}", "EN_KOTU": "{:.2f}", "ORT_MUTLAK_SAPMA": "{:.3f}", "ORT_KOR": "{:.2f}",
     "OOT_WMAPE": "{:.2f}", "OOT_SAPMA": "{:+.3f}"}, decimal=","))

karar_havuz = IA_C_OZET[IA_C_OZET["STRATEJI"] != "AILE_60_3_MM"]
en_iyi = karar_havuz["ORT_TEST_WMAPE"].min()
sec_ = karar_havuz[karar_havuz["ORT_TEST_WMAPE"] <= en_iyi + 1.0].sort_values(["SADELIK", "ORT_TEST_WMAPE"]).iloc[0]
IA_C_KARAR = {"STRATEJI": sec_["STRATEJI"], "TUR": sec_["TUR"]}
print(f"KARAR (önceden sabit kural): {sec_['STRATEJI']} / {sec_['TUR']} | OOT hariç ort. test {sec_['ORT_TEST_WMAPE']:.2f} "
      f"(en iyi {en_iyi:.2f}) | OOT {sec_['OOT_WMAPE']:.2f}")
a_ = IA_C_OZET[IA_C_OZET["STRATEJI"] == "AILE_60_3_MM"]
print("Bilgi (karar dışı) AILE_60_3_MM: " + " | ".join(
    f"{r.TUR}: ort. test {r.ORT_TEST_WMAPE:.2f}, OOT {r.OOT_WMAPE:.2f}" for r in a_.itertuples()))
print(f"18C tamam | {_time.time() - t0_:.0f} sn")
