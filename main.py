# %% [0] Kütüphaneler ve veri
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

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 250)

DATASET_ADI = "BDS_L12M90_KAYNAK"          # <-- Dataiku'daki kaynak dataset adı
if "df_raw" not in globals():
    import dataiku
    df_raw = dataiku.Dataset(DATASET_ADI).get_dataframe()
print("Ham veri:", df_raw.shape)

# Sunum renk paleti (tüm grafiklerde ortak)
RENK = {"GERCEK": "#111827", "EGITIM": "#94a3b8", "VALIDASYON": "#2563eb", "OOT": "#ea580c",
        "TAHMIN": "#16a34a", "KISMI": "#6b7280", "CHALLENGER": "#7c3aed", "DUMMY": "#cbd5e1"}
ARKA = {"EGITIM": "#f1f5f9", "VALIDASYON": "#eff6ff", "OOT": "#fff7ed", "TAHMIN": "#f0fdf4"}
yuzde = lambda v, d=2: "-" if pd.isna(v) else f"%{v * 100:.{d}f}".replace(".", ",")
sayi = lambda v, d=2: "-" if pd.isna(v) else f"{v:.{d}f}".replace(".", ",")
PCT_AX = FuncFormatter(lambda x, _: f"%{x:.2f}".replace(".", ","))


# %% [1] Konfigürasyon
CONFIG = {
    # Hedef
    "HEDEF_TUR": "TUTARSAL",                  # "TUTARSAL" | "ADETSEL"
    "HEDEF_SON": "90_12",

    # Bölme (zaman bazlı)
    "GELISTIRME_ARALIK": (202101, 202409),    # eğitim + walk-forward CV + model seçimi
    "OOT_ARALIK":        (202410, 202509),    # sadece final ölçüm

    # Olgunlaşma: M kohortunun 12. ay verisi M+12 ayının ilk haftasında gelir
    "BILGI_AYI": "2026-10",
    "HEDEF_OLGUNLASMA_AY": 12,
    "RAPOR_GECIKME_AY": 0,

    # Walk-forward CV
    "N_SPLITS": 5, "TEST_SIZE": 4, "MIN_TRAIN": 12, "IC_CV_SPLITS": 3,

    # Aday havuzu ve arama
    "MAX_K": 3,
    "AYNI_AYDAN_TEK_ONCUL": True,
    "MODEL_AILELERI": ["Linear", "Ridge", "Lasso", "ElasticNet"],
    "TREND_DENE": True,                       # test edilir ve raporlanır
    "MEVSIM_DENE": True,
    "TREND_SECILEBILIR": False,               # trend gelecekte doğrusal ekstrapole eder -> seçilemez
    "TREND_ORIJIN": "2021-01",

    # Ortalama limit (başvuru başına onaylanan limit)
    "LIMIT_KOLON": "TOPLAM_ONAYLANAN_LIMIT",
    "LIMIT_PAYDA_KOLON": "BASVURU_ADEDI",
    "TUFE_KOLON": None,                       # varsa reel limit
    "SAMPLE_WEIGHT_KOLON": None,

    # Sıralama (geçerli modeller içinde yüzdelik sıra, küçük = iyi)
    "AGIRLIK": {"CV_WMAPE": 0.30, "EN_KOTU_WMAPE": 0.12, "FOLD_WMAPE_STD": 0.12,
                "OVERFIT_ORANI": 0.15, "DELTA_MAE": 0.12, "MUTLAK_FOLD_SAPMA": 0.19},
    # Geçerlilik eşikleri (naif ortalama modeline göre)
    "GECERLI": {"CV_WMAPE_DUMMY_ORAN": 0.50, "EN_KOTU_DUMMY_ORAN": 0.75,
                "OVERFIT_ORANI_MAX": 2.0, "SAPMA_DUMMY_ORAN": 0.35},
    "YON_TOLERANSI": "oto",

    # Optuna: sağlamlık kontrolü (doğrusal olmayan aileler raporlanır, seçilemez)
    "OPTUNA_CALISTIR": True, "OPTUNA_TRIALS": 60, "OPTUNA_SET_SAYISI": 20,
    "FORMUL_AILELERI": ["Linear", "Ridge", "Lasso", "ElasticNet", "Ridge*", "Lasso*",
                        "ElasticNet*", "Huber"],

    # Model seçimi
    "SECIM_MODU": "SABIT",                    # "SABIT" (onaylı champion) | "KURAL" (sadelik kuralı)
    "CHAMPION_SABIT": {"girdiler": ["30_2", "KKB_SKOR_O0120"], "aile": "Linear", "mevsim": False},
    "SADELIK_TOLERANSI": 0.03,                # en iyi CV_WMAPE'in %3 yakınındakiler kısa listeye
    "HAC_P_ESIK": 0.10,                       # tüm katsayılar anlamlı olmalı (geliştirme verisinde)

    # Seviye düzeltmesi (yönetim overlay'i)
    "OVERLAY_PENCERE": 12,                    # son 12 olgun kohortun walk-forward artık ortalaması
    "OVERLAY_MIN_AYNI_ISARET": 10,            # en az 10/12 aynı işaret ise uygulanır
}

ORTAK_ALANLAR = ["KKB_SKOR_O0120", "MM_SKOR", "KALIBRE_PD_AGIRLIKLI", "ORTALAMA_LIMIT"]
ONCU_KODLARI = ["20_2", "30_2", "30_3", "60_3", "30_4", "60_4", "30_5", "60_5", "30_6", "60_6"]
ACIKLAMA = {"KKB_SKOR_O0120": "KKB skoru (başvuru anı, ağırlıklı ort.)",
            "MM_SKOR": "Müşteri modeli skoru (başvuru anı)",
            "KALIBRE_PD_AGIRLIKLI": "Kalibre PD (limit ağırlıklı)",
            "ORTALAMA_LIMIT": "Başvuru başına onaylanan limit"}
MEVSIM_KOL, TREND_KOL = ["SIN_AY", "COS_AY"], "TREND_IDX"

oncul_kolon = lambda kod, tur=None: f"KK_TAKIP_{kod}_{tur or CONFIG['HEDEF_TUR']}"
oncul_ay = lambda kod: int(kod.split("_")[1])
kisa_ad = lambda k: k.replace("KK_TAKIP_", "").replace(f"_{CONFIG['HEDEF_TUR']}", "")
kolon_coz = lambda ad: oncul_kolon(ad) if ad in ONCU_KODLARI else ad
taban = lambda kolonlar: [c for c in kolonlar if c not in MEVSIM_KOL + [TREND_KOL]]
print("Ağırlık toplamı:", round(sum(CONFIG["AGIRLIK"].values()), 4))


# %% [2] Veri hazırlama, olgunlaşma ve bölme
def veri_hazirla(df_raw, config):
    tur = config["HEDEF_TUR"]
    hedef = oncul_kolon(config["HEDEF_SON"], tur)
    onculler = [oncul_kolon(k, tur) for k in ONCU_KODLARI]
    k90 = sorted([c for c in df_raw.columns if c.startswith("KK_TAKIP_90_") and c.endswith(f"_{tur}")
                  and c.split("_")[3].isdigit() and int(c.split("_")[3]) < 12],
                 key=lambda c: int(c.split("_")[3]))
    ham_ortak = [c for c in ORTAK_ALANLAR if c != "ORTALAMA_LIMIT"]
    ek = [config["LIMIT_KOLON"], config["LIMIT_PAYDA_KOLON"]] + \
         [config[k] for k in ["SAMPLE_WEIGHT_KOLON", "TUFE_KOLON"] if config[k]]
    gerekli = list(dict.fromkeys(["DONEM_1", hedef, *onculler, *ham_ortak, *ek, *k90]))
    eksik = sorted(set(gerekli) - set(df_raw.columns))
    if eksik:
        raise ValueError(f"Eksik kolon: {eksik}")
    if any("_90_" in k for k in onculler + ORTAK_ALANLAR):
        raise ValueError("90+ alanı girdi olamaz.")

    d = df_raw[gerekli].copy()
    d["DONEM_1"] = pd.to_numeric(d["DONEM_1"], errors="raise").astype(int)
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
    d["HEDEF_ERISIM"] = d["AY"] + config["HEDEF_OLGUNLASMA_AY"] + gec
    gecerli_hedef = d[hedef].notna() & np.isfinite(d[hedef]) & d[hedef].between(0, 1)
    d["HEDEF_KULLANILABILIR"] = (d["HEDEF_ERISIM"] < bilgi) & gecerli_hedef
    d["GOZLENEN_AY"] = d["AY"].apply(lambda p: (bilgi - p).n) - 1 - gec

    kismi = {}
    for k, kol in zip(ONCU_KODLARI, onculler):
        tam_mi = (d["AY"] + oncul_ay(k) + gec) < bilgi
        if (~tam_mi & d[kol].notna()).any():
            kismi[kisa_ad(kol)] = int((~tam_mi & d[kol].notna()).sum())
        d.loc[~tam_mi, kol] = np.nan
    if kismi:
        print("Kısmi öncül değeri NaN yapıldı:", kismi)
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
    return gel, oot


d, HEDEF, ONCULLER, K90 = veri_hazirla(df_raw, CONFIG)
gel_idx, oot_idx = bolme_uygula(d, CONFIG)
HK = kisa_ad(HEDEF)
d["BOLGE"] = np.select([d.index.isin(gel_idx), d.index.isin(oot_idx), ~d["HEDEF_KULLANILABILIR"]],
                       ["GELISTIRME", "OOT", "TAHMIN"], "DISARIDA")
VERI_OZET = (d.groupby("BOLGE", sort=False)
              .agg(N=("AY", "size"), BAS=("AY", "min"), SON=("AY", "max"), HEDEF_ORT=(HEDEF, "mean"))
              .reindex(["GELISTIRME", "OOT", "TAHMIN"]))
print(f"Hedef: {HEDEF} | Bilgi ayı: {CONFIG['BILGI_AYI']} | 90+ olgunluk kolonları: {[kisa_ad(k) for k in K90]}")
display(VERI_OZET.style.format({"HEDEF_ORT": yuzde}, na_rep="-"))


# %% [3] Aday havuzu
HAVUZ = ONCULLER + ORTAK_ALANLAR
KOLON_AYI = {oncul_kolon(k): oncul_ay(k) for k in ONCU_KODLARI}
kolon_ayi = lambda c: KOLON_AYI.get(c, 0)
bilgi_ayi_hesapla = lambda kolonlar: max([kolon_ayi(c) for c in kolonlar], default=0)

def kombinasyon_uygun(alt):
    aylar = [kolon_ayi(c) for c in alt if kolon_ayi(c) > 0]
    return (not CONFIG["AYNI_AYDAN_TEK_ONCUL"]) or len(aylar) == len(set(aylar))

KOMBS = [a for k in range(1, CONFIG["MAX_K"] + 1) for a in combinations(HAVUZ, k) if kombinasyon_uygun(a)]
HAVUZ_TABLO = pd.DataFrame({
    "DEGISKEN": [kisa_ad(c) for c in HAVUZ],
    "TUR": ["Öncül (gecikme)" if c in ONCULLER else "Başvuru anı" for c in HAVUZ],
    "GOZLEM_AYI": [kolon_ayi(c) for c in HAVUZ],
    "ACIKLAMA": [f"{kisa_ad(c).split('_')[0]}+ gün gecikme, ilk {kolon_ayi(c)} ay (tutar ağırlıklı)"
                 if c in ONCULLER else ACIKLAMA.get(c, "") for c in HAVUZ]})
print(f"Havuz: {len(HAVUZ)} değişken | uygun kombinasyon (K<={CONFIG['MAX_K']}): {len(KOMBS)}")
print("Bilgi ayı dağılımı:", pd.Series([bilgi_ayi_hesapla(a) for a in KOMBS]).value_counts().sort_index().to_dict())
display(HAVUZ_TABLO)


# %% [4] Olgunlaşma kontrollü walk-forward CV + metrik motoru
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import (LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet,
                                  ElasticNetCV, HuberRegressor)
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor

MIN_TRAIN = CONFIG["MIN_TRAIN"]
ALPHALAR = np.logspace(-3, 3, 13)
cv_df = d.loc[gel_idx].sort_values("AY").reset_index(drop=True)
HEDEF_ORT = float(cv_df[HEDEF].mean())
TOL = (0.5 * float(np.std(np.diff(cv_df[HEDEF]))) if CONFIG["YON_TOLERANSI"] == "oto"
       else float(CONFIG["YON_TOLERANSI"]))


def model_fabrika(aile, p=None):
    p = p or {}
    ic = TimeSeriesSplit(n_splits=CONFIG["IC_CV_SPLITS"])
    return {
        "Dummy":       lambda: DummyRegressor(strategy="mean"),
        "Linear":      lambda: LinearRegression(),
        "Ridge":       lambda: RidgeCV(alphas=ALPHALAR, cv=ic),
        "Lasso":       lambda: LassoCV(alphas=ALPHALAR, cv=ic, max_iter=100000),
        "ElasticNet":  lambda: ElasticNetCV(alphas=ALPHALAR, l1_ratio=[0.2, 0.5, 0.8], cv=ic, max_iter=100000),
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


def agirlik_al(df):
    kol = CONFIG["SAMPLE_WEIGHT_KOLON"]
    return None if kol is None else df[kol].to_numpy(dtype=float)


def pipe_fit(estimator, X, y, w=None):
    p = make_pipeline(StandardScaler(), clone(estimator))
    return p.fit(X, y) if w is None else p.fit(X, y, **{f"{p.steps[-1][0]}__sample_weight": w})


def fold_olustur(bilgi_ayi):
    """Walk-forward; eğitimden, bloğun ilk kohortunun tahmin anında hedefi belli olmayanlar çıkarılır."""
    aylar = pd.PeriodIndex(cv_df["AY"])
    foldlar, ozet = [], []
    tscv = TimeSeriesSplit(n_splits=CONFIG["N_SPLITS"], test_size=CONFIG["TEST_SIZE"])
    for i, (tr, val) in enumerate(tscv.split(np.arange(len(cv_df))), 1):
        ilk_tahmin = aylar[val[0]] + bilgi_ayi
        tr = tr[(aylar[tr] + CONFIG["HEDEF_OLGUNLASMA_AY"] + CONFIG["RAPOR_GECIKME_AY"]) < ilk_tahmin]
        if len(tr) < MIN_TRAIN:
            raise ValueError(f"Fold {i} (bilgi ayı {bilgi_ayi}): eğitim {len(tr)} < {MIN_TRAIN}")
        foldlar.append((tr, val))
        ozet.append({"FOLD": i, "EGITIM_N": len(tr), "EGITIM": f"{aylar[tr[0]]} – {aylar[tr[-1]]}",
                     "DISLANAN (olgunlaşmamış)": int(val[0] - tr[-1] - 1),
                     "VALIDASYON": f"{aylar[val[0]]} – {aylar[val[-1]]}", "TAHMIN_ANI": str(ilk_tahmin)})
    return foldlar, pd.DataFrame(ozet)


FOLD_CACHE = {}
def fold_al(b):
    if b not in FOLD_CACHE:
        FOLD_CACHE[b] = fold_olustur(b)[0]
    return FOLD_CACHE[b]


def wmape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.abs(p - y).sum() / max(np.abs(y).sum(), 1e-12) * 100)


def cv_degerlendir(kolonlar, estimator, foldlar):
    y = cv_df[HEDEF].to_numpy(dtype=float)
    aylar = pd.PeriodIndex(cv_df["AY"])
    X = cv_df[list(kolonlar)]
    hata, gercek, f_wmape, f_sapma, f_oran, tr_mse = [], [], [], [], [], []
    d_g, d_t, aralik_disi = [], [], 0
    for tr, val in foldlar:
        m = pipe_fit(estimator, X.iloc[tr], y[tr], agirlik_al(cv_df.iloc[tr]))
        p, p_tr = m.predict(X.iloc[val]), m.predict(X.iloc[tr])
        tr_mse.append(np.mean((p_tr - y[tr]) ** 2))
        f_wmape.append(wmape(y[val], p))
        f_sapma.append(float(np.mean(p - y[val])))
        f_oran.append(f_wmape[-1] / max(wmape(y[tr], p_tr), 1e-12))
        aralik_disi += int(((p < 0) | (p > 2 * y[tr].max())).sum())
        bit = np.diff(aylar[val].asi8) == 1
        d_g.extend(np.diff(y[val])[bit]); d_t.extend(np.diff(p)[bit])
        hata.extend(p - y[val]); gercek.extend(y[val])
    hata, gercek, d_g, d_t = map(np.asarray, (hata, gercek, d_g, d_t))
    anlamli = np.abs(d_g) > TOL
    dogru = np.sign(np.where(np.abs(d_t) <= TOL, 0, d_t))[anlamli] == np.sign(d_g[anlamli])
    buy = np.abs(d_g[anlamli])
    return {
        "CV_WMAPE": float(np.abs(hata).sum() / gercek.sum() * 100),
        "CV_MAE": float(np.mean(np.abs(hata))), "CV_RMSE": float(np.sqrt(np.mean(hata ** 2))),
        "CV_SAPMA": float(np.mean(hata)),
        "MUTLAK_FOLD_SAPMA": float(np.mean(np.abs(f_sapma))),
        "EN_KOTU_WMAPE": float(np.max(f_wmape)), "FOLD_WMAPE_STD": float(np.std(f_wmape)),
        "OVERFIT_ORANI": float(np.median(f_oran)),
        "DELTA_MAE": float(np.mean(np.abs(d_t - d_g))) if len(d_g) else np.nan,
        "YON_AGIRLIKLI": float((buy * dogru).sum() / buy.sum()) if anlamli.any() else np.nan,
        "ARALIK_DISI": aralik_disi,
    }


for b in [2, 3, 6]:
    print(f"=== Walk-forward fold yapısı | bilgi ayı = {b} ===")
    display(fold_olustur(b)[1])

DUMMY = cv_degerlendir(HAVUZ[:1], model_fabrika("Dummy"), fold_al(0))
print(f"Naif referans (geçmiş ortalama): CV_WMAPE {DUMMY['CV_WMAPE']:.2f} | "
      f"EN_KOTU {DUMMY['EN_KOTU_WMAPE']:.2f} | MUTLAK_FOLD_SAPMA {DUMMY['MUTLAK_FOLD_SAPMA']:.4f}")


# %% [5] Kombinasyonlu grid araması + sıralama
def varyant_ekleri():
    v = [[]]
    if CONFIG["TREND_DENE"]:
        v.append([TREND_KOL])
    if CONFIG["MEVSIM_DENE"]:
        v.append(MEVSIM_KOL)
    if CONFIG["TREND_DENE"] and CONFIG["MEVSIM_DENE"]:
        v.append([TREND_KOL] + MEVSIM_KOL)
    return v


def kayit_ekle(s, kolonlar, aile, params):
    trend = int(TREND_KOL in kolonlar)
    s.update({"GIRDILER": ", ".join(kisa_ad(c) for c in taban(kolonlar)), "MODEL": aile, "PARAMS": params,
              "BILGI_AY": bilgi_ayi_hesapla(taban(kolonlar)), "K": len(taban(kolonlar)),
              "TREND": trend, "MEVSIM": int("SIN_AY" in kolonlar), "KOLONLAR": tuple(kolonlar),
              "SECILEBILIR": bool((not trend or CONFIG["TREND_SECILEBILIR"])
                                  and aile in CONFIG["FORMUL_AILELERI"])})
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


t0, kayit, hatalar = time.time(), [], []
for alt in KOMBS:
    foldlar = fold_al(bilgi_ayi_hesapla(alt))
    for ek in varyant_ekleri():
        for aile in CONFIG["MODEL_AILELERI"]:
            try:
                s = cv_degerlendir(list(alt) + ek, model_fabrika(aile), foldlar)
                kayit.append(kayit_ekle(s, list(alt) + ek, aile, None))
            except Exception as e:
                hatalar.append({"GIRDILER": ", ".join(map(kisa_ad, alt)), "MODEL": aile, "HATA": repr(e)[:120]})
grid_ham = pd.DataFrame(kayit)
grid = rank_ekle(grid_ham)
print(f"{len(grid)} varyant | geçerli {int(grid['GECERLI'].sum())} | hata {len(hatalar)} | {time.time() - t0:.0f} sn")
if hatalar:
    display(pd.DataFrame(hatalar).head())

GOSTER = ["GIRDILER", "MODEL", "BILGI_AY", "K", "TREND", "MEVSIM", "CV_WMAPE", "EN_KOTU_WMAPE",
          "FOLD_WMAPE_STD", "OVERFIT_ORANI", "DELTA_MAE", "CV_SAPMA", "MUTLAK_FOLD_SAPMA",
          "YON_AGIRLIKLI", "GECERLI", "SECILEBILIR", "RANK_SKORU"]
FMT = {**{c: "{:.2f}" for c in ["CV_WMAPE", "EN_KOTU_WMAPE", "FOLD_WMAPE_STD"]},
       **{c: "{:.4f}" for c in ["OVERFIT_ORANI", "DELTA_MAE", "CV_SAPMA", "MUTLAK_FOLD_SAPMA",
                                "YON_AGIRLIKLI", "RANK_SKORU"]}}
print("=== Genel ilk 15 ===")
display(grid[GOSTER].head(15).style.format(FMT, decimal=","))


def grup_ozet(df, by):
    out = df.groupby(by).agg(VARYANT=("CV_WMAPE", "size"), EN_IYI_CV_WMAPE=("CV_WMAPE", "min"),
                             MEDYAN_CV_WMAPE=("CV_WMAPE", "median"))
    out["GECERLI"] = df[df["GECERLI"]].groupby(by).size().reindex(out.index).fillna(0).astype(int)
    return out[["VARYANT", "GECERLI", "EN_IYI_CV_WMAPE", "MEDYAN_CV_WMAPE"]]


grid["VARYANT"] = np.select([(grid["TREND"] == 1) & (grid["MEVSIM"] == 1), grid["TREND"] == 1,
                             grid["MEVSIM"] == 1], ["Trend+Mevsim", "Trend", "Mevsim"], "Yok")
OZET_VARYANT = grup_ozet(grid, "VARYANT").reindex(["Yok", "Mevsim", "Trend", "Trend+Mevsim"]).dropna(how="all")
OZET_BILGI_AY = grup_ozet(grid, "BILGI_AY")
OZET_AILE = grup_ozet(grid, "MODEL")
OZET_K = grup_ozet(grid, "K")
for ad, t in [("Trend / mevsim", OZET_VARYANT), ("Bilgi ayı (en geç öncül)", OZET_BILGI_AY),
              ("Model ailesi", OZET_AILE), ("Değişken sayısı", OZET_K)]:
    print(f"=== {ad} ===")
    display(t.style.format({"EN_IYI_CV_WMAPE": "{:.2f}", "MEDYAN_CV_WMAPE": "{:.2f}"}, decimal=",", na_rep="-"))


# %% [6] Optuna: geniş model aileleri (sağlamlık kontrolü)
if CONFIG["OPTUNA_CALISTIR"]:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    setler = (grid[grid["GECERLI"] & grid["SECILEBILIR"]].drop_duplicates("KOLONLAR")["KOLONLAR"]
              .head(CONFIG["OPTUNA_SET_SAYISI"]).tolist())
    opt_kayit = []

    def amac(trial):
        kolonlar = list(setler[trial.suggest_int("set", 0, len(setler) - 1)])
        aile = trial.suggest_categorical("aile", ["Ridge*", "Lasso*", "ElasticNet*", "Huber",
                                                  "SVR", "ExtraTrees", "GBoost"])
        p = {}
        if aile in ("Ridge*", "Lasso*", "ElasticNet*", "Huber"):
            p["alpha"] = trial.suggest_float(f"{aile}_alpha", 1e-4, 1e2, log=True)
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
        s = cv_degerlendir(kolonlar, model_fabrika(aile, p), fold_al(bilgi_ayi_hesapla(taban(kolonlar))))
        opt_kayit.append(kayit_ekle(s, kolonlar, aile, p))
        return s["CV_WMAPE"], (s["DELTA_MAE"] if np.isfinite(s["DELTA_MAE"]) else 1.0)

    study = optuna.create_study(directions=["minimize", "minimize"], sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(amac, n_trials=CONFIG["OPTUNA_TRIALS"])
    grid = rank_ekle(pd.concat([grid_ham, pd.DataFrame(opt_kayit)], ignore_index=True))
    grid["VARYANT"] = np.select([(grid["TREND"] == 1) & (grid["MEVSIM"] == 1), grid["TREND"] == 1,
                                 grid["MEVSIM"] == 1], ["Trend+Mevsim", "Trend", "Mevsim"], "Yok")
    OZET_AILE = grup_ozet(grid, "MODEL")
    print(f"Optuna sonrası: {len(grid)} varyant | geçerli {int(grid['GECERLI'].sum())}")
    print("=== Model ailesine göre en iyi (Optuna dahil) ===")
    display(OZET_AILE.sort_values("EN_IYI_CV_WMAPE").style.format(
        {"EN_IYI_CV_WMAPE": "{:.2f}", "MEDYAN_CV_WMAPE": "{:.2f}"}, decimal=",", na_rep="-"))
else:
    print("Optuna atlandı.")


# %% [7] Model seçimi: sadelik + anlamlılık kuralı, Champion / Challenger
AILE_SIRA = {"Linear": 0, "Ridge": 1, "Ridge*": 1, "Lasso": 2, "Lasso*": 2,
             "ElasticNet": 3, "ElasticNet*": 3, "Huber": 4}


def hac_p(kolonlar, df):
    X = sm.add_constant(df[list(kolonlar)].astype(float).to_numpy(), has_constant="add")
    y = df[HEDEF].astype(float).to_numpy()
    lag = int(np.ceil(4 * (len(y) / 100) ** (2 / 9)))
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag}).pvalues[1:]


aday = grid[grid["GECERLI"] & grid["SECILEBILIR"]].copy()
if aday.empty:
    raise ValueError("Seçilebilir geçerli model yok: GECERLI eşiklerini kontrol et.")
kisa = aday[aday["CV_WMAPE"] <= aday["CV_WMAPE"].min() * (1 + CONFIG["SADELIK_TOLERANSI"])].copy()
kisa = kisa.drop_duplicates(["KOLONLAR", "MODEL"])
kisa["PARAMETRE"] = kisa["KOLONLAR"].apply(len)
kisa["HAC_MAKS_P"] = kisa["KOLONLAR"].apply(lambda k: float(np.max(hac_p(k, cv_df))))
kisa["ANLAMLI"] = kisa["HAC_MAKS_P"] <= CONFIG["HAC_P_ESIK"]
kisa["AILE_SIRA"] = kisa["MODEL"].map(AILE_SIRA)
kisa = kisa.sort_values(["ANLAMLI", "PARAMETRE", "BILGI_AY", "AILE_SIRA", "RANK_SKORU"],
                        ascending=[False, True, True, True, True]).reset_index(drop=True)
print(f"=== Sadelik kısa listesi (CV_WMAPE <= en iyi x {1 + CONFIG['SADELIK_TOLERANSI']:.2f}) ===")
display(kisa[["GIRDILER", "MODEL", "BILGI_AY", "MEVSIM", "PARAMETRE", "CV_WMAPE", "EN_KOTU_WMAPE",
              "HAC_MAKS_P", "ANLAMLI", "RANK_SKORU"]].style.format(
    {"CV_WMAPE": "{:.2f}", "EN_KOTU_WMAPE": "{:.2f}", "HAC_MAKS_P": "{:.3f}", "RANK_SKORU": "{:.4f}"}, decimal=","))


def spec_yap(satir, rol):
    return {"rol": rol, "kolonlar": list(satir["KOLONLAR"]), "aile": satir["MODEL"], "params": satir["PARAMS"],
            "bilgi_ay": int(satir["BILGI_AY"]), "girdiler": satir["GIRDILER"],
            "mevsim": int(satir["MEVSIM"]),
            "ad": f"{rol}: {satir['GIRDILER']}{' + mevsim' if satir['MEVSIM'] else ''} | {satir['MODEL']} | "
                  f"{int(satir['BILGI_AY'])}. ay"}


KURAL_SATIR = kisa.iloc[0]
if CONFIG["SECIM_MODU"] == "SABIT":
    cs = CONFIG["CHAMPION_SABIT"]
    hedef_set = {kolon_coz(g) for g in cs["girdiler"]} | (set(MEVSIM_KOL) if cs["mevsim"] else set())
    es = grid[(grid["KOLONLAR"].apply(set) == hedef_set) & (grid["MODEL"] == cs["aile"])]
    if es.empty:
        raise ValueError(f"Onaylı champion grid'de bulunamadı: {cs}")
    CHAMP_SATIR = es.iloc[0]
    if not CHAMP_SATIR["GECERLI"]:
        print("UYARI: onaylı champion bu çalıştırmada GECERLI değil, sonuçları kontrol et.")
else:
    CHAMP_SATIR = KURAL_SATIR
if set(KURAL_SATIR["KOLONLAR"]) != set(CHAMP_SATIR["KOLONLAR"]) or KURAL_SATIR["MODEL"] != CHAMP_SATIR["MODEL"]:
    print(f"BİLGİ: sadelik kuralının önerisi farklı -> {KURAL_SATIR['GIRDILER']} | {KURAL_SATIR['MODEL']}")

chal_aday = aday[aday["KOLONLAR"].apply(lambda k: set(taban(k))) != set(taban(CHAMP_SATIR["KOLONLAR"]))]
CHAL_SATIR = chal_aday.iloc[0]          # en yüksek RANK_SKORU (performans odaklı)
CHAMP = spec_yap(CHAMP_SATIR, "CHAMPION")
CHAL = spec_yap(CHAL_SATIR, "CHALLENGER")
SECIMLER = {"CHAMPION": CHAMP, "CHALLENGER": CHAL}
print("\n" + CHAMP["ad"] + "   <- sadelik + anlamlılık")
print(CHAL["ad"] + "   <- en yüksek bileşik skor")
SECIM_TABLO = pd.DataFrame([CHAMP_SATIR, CHAL_SATIR], index=["CHAMPION", "CHALLENGER"])[GOSTER[:-3]]
display(SECIM_TABLO.style.format(FMT, decimal=","))


# %% [8] OOT walk-forward, performans tablosu, overlay
olgun_df = d[d["HEDEF_KULLANILABILIR"]].copy()
BILGI = pd.Period(CONFIG["BILGI_AYI"], "M")


def wf_tahmin(spec):
    """Her olgun kohort, kendi tahmin anında hedefi belli olan kohortlarla eğitilen modelle tahmin edilir."""
    kol, kayit = spec["kolonlar"], []
    for i in olgun_df.index:
        c = olgun_df.loc[i]
        tahmin_ay = c["AY"] + spec["bilgi_ay"]
        eg = olgun_df[olgun_df["HEDEF_ERISIM"] < tahmin_ay]
        if len(eg) < MIN_TRAIN:
            continue
        m = pipe_fit(model_fabrika(spec["aile"], spec["params"]), eg[kol], eg[HEDEF], agirlik_al(eg))
        kayit.append({"DONEM_1": int(c["DONEM_1"]), "AY": c["AY"], "BOLGE": c["BOLGE"],
                      "TAHMIN_AY": tahmin_ay, "HEDEF_ERISIM": c["HEDEF_ERISIM"], "GERCEK": float(c[HEDEF]),
                      "TAHMIN": float(m.predict(olgun_df.loc[[i], kol])[0]),
                      "DUMMY": float(eg[HEDEF].mean()), "EGITIM_N": len(eg)})
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
            "EGIM": float(np.polyfit(p, g, 1)[0]) if np.std(p) > 0 else np.nan,
            "YON": f"{int((np.sign(dg) == np.sign(dp)).sum())}/{len(dg)}",
            "POZITIF_SAPMA": f"{int((p > g).sum())}/{len(g)}"}


VAL_BAS = cv_df.loc[fold_al(CHAMP["bilgi_ay"])[0][1][0], "AY"]      # ilk validasyon kohortu
WF, OVERLAY, PERF = {}, {}, []
for rol, spec in SECIMLER.items():
    w = wf_tahmin(spec)
    WF[rol] = w
    for bolge, maske in [("VALIDASYON", (w["BOLGE"] == "GELISTIRME") & (w["AY"] >= VAL_BAS)),
                         ("OOT", w["BOLGE"] == "OOT")]:
        s = w[maske]
        mm, dm = metrik(s["GERCEK"], s["TAHMIN"]), metrik(s["GERCEK"], s["DUMMY"])
        PERF.append({"MODEL": rol, "BOLGE": bolge, **mm, "DUMMY_WMAPE": dm["WMAPE"],
                     "BECERI": 1 - mm["WMAPE"] / dm["WMAPE"]})
    son = w[w["HEDEF_ERISIM"] < BILGI].sort_values("HEDEF_ERISIM").tail(CONFIG["OVERLAY_PENCERE"])
    ayni = int(max((son["ARTIK"] > 0).sum(), (son["ARTIK"] < 0).sum()))
    OVERLAY[rol] = {"deger": float(son["ARTIK"].mean()), "ayni_isaret": ayni, "n": len(son),
                    "donem": f"{son['DONEM_1'].min()}-{son['DONEM_1'].max()}",
                    "uygula": ayni >= CONFIG["OVERLAY_MIN_AYNI_ISARET"]}

PERF = pd.DataFrame(PERF)
print(f"Validasyon: {VAL_BAS} – {cv_df['AY'].max()} | OOT: {d.loc[oot_idx, 'AY'].min()} – {d.loc[oot_idx, 'AY'].max()}")
display(PERF.style.format({"WMAPE": "{:.2f}", "MAE_yp": "{:.3f}", "RMSE_yp": "{:.3f}", "SAPMA_yp": "{:+.3f}",
                           "R2": "{:.3f}", "KORELASYON": "{:.3f}", "EGIM": "{:.2f}", "DUMMY_WMAPE": "{:.2f}",
                           "BECERI": "{:.3f}"}, decimal=","))
for rol, o in OVERLAY.items():
    print(f"{rol}: overlay {o['deger'] * 100:+.3f} yp | son {o['n']} olgun kohortun {o['ayni_isaret']}'i aynı "
          f"işaret ({o['donem']}) -> {'UYGULANIR' if o['uygula'] else 'uygulanmaz'}")


# %% [9] Üretim: final model, formül, tahminler, chain-ladder, kayıt
URETIM, FORMUL, KATSAYI = {}, {}, []
for rol, spec in SECIMLER.items():
    kol = spec["kolonlar"]
    m = pipe_fit(model_fabrika(spec["aile"], spec["params"]), olgun_df[kol], olgun_df[HEDEF], agirlik_al(olgun_df))
    URETIM[rol] = m
    reg, sc = m[-1], m[0]
    if hasattr(reg, "coef_"):
        coef = np.ravel(reg.coef_) / sc.scale_
        b0 = float(np.ravel([reg.intercept_])[0] - np.dot(coef, sc.mean_))
        FORMUL[rol] = f"{HK} = {b0:.6f} " + " ".join(f"{c:+.6g}·{kisa_ad(k)}" for k, c in zip(kol, coef))
        for k, c in zip(kol, coef):
            KATSAYI.append({"MODEL": rol, "DEGISKEN": kisa_ad(k), "KATSAYI_HAM": c,
                            "1_STD_ETKI_yp": c * olgun_df[k].std() * 100})
        KATSAYI.append({"MODEL": rol, "DEGISKEN": "Sabit", "KATSAYI_HAM": b0, "1_STD_ETKI_yp": np.nan})
    print(f"{spec['ad']} | N={len(olgun_df)}\n  {FORMUL.get(rol, 'formül yok')}")
KATSAYI = pd.DataFrame(KATSAYI)
display(KATSAYI.style.format({"KATSAYI_HAM": "{:.6g}", "1_STD_ETKI_yp": "{:+.3f}"}, decimal=",", na_rep="-"))

gelecek = d[d["BOLGE"] == "TAHMIN"].sort_values("AY")
TAHMIN = gelecek[["DONEM_1", "AY", "GOZLENEN_AY"]].reset_index(drop=True)
TAHMIN["KISMI_GERCEK"] = gelecek[HEDEF].to_numpy()
for rol, spec in SECIMLER.items():
    X = gelecek[spec["kolonlar"]]
    ok = X.notna().all(axis=1).to_numpy()
    ham = np.full(len(X), np.nan)
    ham[ok] = URETIM[rol].predict(X[ok])
    ov = OVERLAY[rol]["deger"] if OVERLAY[rol]["uygula"] else 0.0
    TAHMIN[f"{rol}_HAM"] = np.clip(ham, 0, 1)
    TAHMIN[f"{rol}_DUZ"] = np.clip(ham - ov, 0, 1)

# Chain-ladder: kısmi gerçek / güncel olgunlaşma oranı (modelden bağımsız referans)
if K90:
    son12 = olgun_df.sort_values("AY").tail(12)
    OLGUNLASMA = {int(k.split("_")[3]): float((son12[k] / son12[HEDEF]).mean()) for k in K90}
    OLGUNLASMA[12] = 1.0
    xs = sorted(OLGUNLASMA)
    TAHMIN["OLGUNLASMA_ORANI"] = np.interp(TAHMIN["GOZLENEN_AY"], xs, [OLGUNLASMA[x] for x in xs])
    TAHMIN["CHAIN_LADDER"] = TAHMIN["KISMI_GERCEK"] / TAHMIN["OLGUNLASMA_ORANI"]
    print("Olgunlaşma oranı (90_m / 90_12, son 12 olgun kohort):", {k: round(v, 3) for k, v in OLGUNLASMA.items()})
else:
    OLGUNLASMA = {}
TAHMIN["ARALIK_ALT"] = TAHMIN[["CHAMPION_HAM", "CHAMPION_DUZ"]].min(axis=1)
TAHMIN["ARALIK_UST"] = TAHMIN[["CHAMPION_HAM", "CHAMPION_DUZ"]].max(axis=1)

pct_kol = [c for c in TAHMIN.columns if c not in ("DONEM_1", "AY", "GOZLENEN_AY", "OLGUNLASMA_ORANI")]
print(f"=== {HK} TAHMİNLERİ (bilgi ayı {CONFIG['BILGI_AYI']}) ===")
display(TAHMIN.drop(columns="AY").style.format({**{c: "{:.2%}" for c in pct_kol}, "DONEM_1": "{:.0f}",
                                                "OLGUNLASMA_ORANI": "{:.3f}"}, decimal=",", na_rep="-"))

KAYIT = TAHMIN.drop(columns="AY").copy()
KAYIT.insert(0, "KAYIT_TARIHI", pd.Timestamp.today().normalize())
KAYIT.insert(1, "BILGI_AYI", CONFIG["BILGI_AYI"])
KAYIT.insert(2, "CHAMPION", CHAMP["ad"])
# import dataiku; dataiku.Dataset("L12M90_TAHMIN_KAYIT").write_with_schema(KAYIT)   # izleme için


# %% [10] İstatistiksel testler (champion) + girdi kayması + dönüşüm oranı
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor

kol = CHAMP["kolonlar"]
Xc = sm.add_constant(olgun_df[kol].astype(float).to_numpy(), has_constant="add")
yv = olgun_df[HEDEF].astype(float).to_numpy()
ols = sm.OLS(yv, Xc).fit()
lag = int(np.ceil(4 * (len(yv) / 100) ** (2 / 9)))
hac = sm.OLS(yv, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
resid = URETIM["CHAMPION"].predict(olgun_df[kol]) - yv
rho = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
jb = jarque_bera(resid)
TESTLER = pd.DataFrame([
    {"TEST": "Katsayı anlamlılığı (HAC)", "SONUC": ", ".join(f"{kisa_ad(k)} p={p:.3f}" for k, p in zip(kol, hac.pvalues[1:])),
     "YORUM": "p<0,05 anlamlı"},
    {"TEST": "R² / Düzeltilmiş R² (tüm olgun veri)", "SONUC": f"{ols.rsquared:.3f} / {ols.rsquared_adj:.3f}", "YORUM": "in-sample"},
    {"TEST": "Durbin-Watson", "SONUC": f"{durbin_watson(resid):.3f}", "YORUM": "2 = otokorelasyon yok"},
    {"TEST": "Efektif gözlem", "SONUC": f"{len(yv) * (1 - rho) / (1 + rho):.1f} / {len(yv)}", "YORUM": f"lag-1 rho {rho:.2f}"},
    {"TEST": "Ljung-Box p (lag 1/3/6)", "SONUC": " / ".join(f"{p:.3f}" for p in acorr_ljungbox(resid, lags=[1, 3, 6])["lb_pvalue"]),
     "YORUM": "<0,05 otokorelasyon"},
    {"TEST": "Breusch-Pagan p", "SONUC": f"{het_breuschpagan(resid, Xc)[1]:.3f}", "YORUM": ">0,05 sabit varyans"},
    {"TEST": "Jarque-Bera p", "SONUC": f"{jb[1]:.3f} (çarp. {jb[2]:.2f}, bas. {jb[3]:.2f})", "YORUM": ">0,05 normal"},
    {"TEST": "Ramsey RESET p", "SONUC": f"{linear_reset(ols, power=2, use_f=True).pvalue:.3f}", "YORUM": ">0,05 doğrusal form yeterli"},
    {"TEST": "VIF", "SONUC": ", ".join(f"{kisa_ad(k)}={variance_inflation_factor(Xc, i + 1):.2f}" for i, k in enumerate(kol))
     if len(kol) > 1 else "-", "YORUM": "<5 iyi, >10 sorunlu"},
])
print(f"=== {CHAMP['ad']} | istatistiksel testler ===")
display(TESTLER)

drift_kol = list(dict.fromkeys(taban(CHAMP["kolonlar"]) + taban(CHAL["kolonlar"]) + [HEDEF]))
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


# %% [11] Sunum görselleri
def ts(p):
    return p.to_timestamp() if hasattr(p, "to_timestamp") else pd.Period(p, "M").to_timestamp()


def bolge_boya(ax, alanlar, etiket=True):
    for ad, (p0, p1) in alanlar.items():
        x0, x1 = ts(p0) - pd.Timedelta(days=15), ts(p1) + pd.Timedelta(days=15)
        ax.axvspan(x0, x1, color=ARKA[ad], zorder=0)
        if etiket:
            ax.text(x0 + (x1 - x0) / 2, 1.01, {"EGITIM": "EĞİTİM", "VALIDASYON": "VALİDASYON (CV)",
                                               "OOT": "OOT", "TAHMIN": "TAHMİN"}[ad],
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=RENK[ad] if ad != "EGITIM" else "#475569")


def ana_grafik(rol):
    """Gerçekleşen + eğitim uyumu + validasyon + OOT + tahmin, altında performans tablosu."""
    spec, w = SECIMLER[rol], WF[rol]
    kol = spec["kolonlar"]
    egit = olgun_df[olgun_df["AY"] < VAL_BAS]
    m_eg = pipe_fit(model_fabrika(spec["aile"], spec["params"]), egit[kol], egit[HEDEF])
    seg = {
        "EGITIM": pd.DataFrame({"AY": egit["AY"], "GERCEK": egit[HEDEF], "TAHMIN": m_eg.predict(egit[kol])}),
        "VALIDASYON": w[(w["BOLGE"] == "GELISTIRME") & (w["AY"] >= VAL_BAS)],
        "OOT": w[w["BOLGE"] == "OOT"],
    }
    alanlar = {k: (v["AY"].min(), v["AY"].max()) for k, v in seg.items()}
    alanlar["TAHMIN"] = (TAHMIN["AY"].min(), TAHMIN["AY"].max())

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.25], hspace=0.12)
    ax = fig.add_subplot(gs[0])
    bolge_boya(ax, alanlar)
    hist = olgun_df.sort_values("AY")
    ax.plot(hist["AY"].dt.to_timestamp(), hist[HEDEF] * 100, "o-", color=RENK["GERCEK"], lw=2.2, ms=4,
            label="Gerçekleşen", zorder=5)
    etiket = {"EGITIM": "Eğitim uyumu (in-sample)", "VALIDASYON": "Validasyon tahmini (walk-forward)",
              "OOT": "OOT tahmini (walk-forward)"}
    for k, s in seg.items():
        ax.plot(s["AY"].dt.to_timestamp(), s["TAHMIN"] * 100, "o-" if k != "EGITIM" else "-",
                color=RENK[k], lw=2, ms=4, label=etiket[k], zorder=4)
    tx = TAHMIN["AY"].dt.to_timestamp()
    ax.fill_between(tx, TAHMIN[f"{rol}_HAM"] * 100, TAHMIN[f"{rol}_DUZ"] * 100, color=RENK["TAHMIN"],
                    alpha=0.15, zorder=2)
    ax.plot(tx, TAHMIN[f"{rol}_DUZ"] * 100, "D-", color=RENK["TAHMIN"], lw=2.4, ms=6,
            label="Tahmin (seviye düzeltmeli)", zorder=6)
    ax.plot(tx, TAHMIN[f"{rol}_HAM"] * 100, "D--", color=RENK["TAHMIN"], lw=1.2, ms=4, alpha=0.6,
            label="Tahmin (ham model)", zorder=6)
    ax.plot(tx, TAHMIN["KISMI_GERCEK"] * 100, "x:", color=RENK["KISMI"], lw=1.4, ms=6,
            label="Kısmi gerçekleşme (alt sınır)", zorder=5)
    for x, v in zip(tx, TAHMIN[f"{rol}_DUZ"]):
        ax.annotate(yuzde(v), (x, v * 100), textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=8, color=RENK["TAHMIN"], fontweight="bold")
    ax.yaxis.set_major_formatter(PCT_AX)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.25)
    ax.set_ylabel(f"{HK} (12 ayda 90+ gecikme oranı)")
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
    ov = OVERLAY[rol]
    satirlar.append(["Tahmin (olgunlaşmamış)", f"{alanlar['TAHMIN'][0]} – {alanlar['TAHMIN'][1]}", len(TAHMIN),
                     "-", "-", f"overlay {ov['deger'] * 100:+.3f}".replace(".", ","), "-", "-", "-", "-"])
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
    axt.text(0, -0.02, "* Beceri = 1 − WMAPE / naif ortalama modelin WMAPE'i. yp = yüzde puan. "
                       "Walk-forward: her kohort, kendi tahmin anında hedefi belli olan kohortlarla eğitilen modelle tahmin edilir.",
             transform=axt.transAxes, fontsize=8, color="#6b7280")
    plt.show()


def zaman_cizelgesi():
    """Veri bölmesi + walk-forward fold yapısı."""
    def span(p0, p1):
        a, b = mdates.date2num(ts(p0)), mdates.date2num(ts(p1 + 1))
        return (a, b - a)
    aylar = pd.PeriodIndex(cv_df["AY"])
    satir = [("Kohortlar", [(span(olgun_df["AY"].min(), VAL_BAS - 1), ARKA["EGITIM"], "Eğitim"),
                            (span(VAL_BAS, aylar.max()), RENK["VALIDASYON"], "Validasyon"),
                            (span(d.loc[oot_idx, "AY"].min(), d.loc[oot_idx, "AY"].max()), RENK["OOT"], "OOT"),
                            (span(TAHMIN["AY"].min(), TAHMIN["AY"].max()), RENK["TAHMIN"], "Tahmin")])]
    for i, (tr, val) in enumerate(fold_al(CHAMP["bilgi_ay"]), 1):
        parca = [(span(aylar[tr[0]], aylar[tr[-1]]), "#bfdbfe", None)]
        if val[0] - tr[-1] > 1:
            parca.append((span(aylar[tr[-1]] + 1, aylar[val[0]] - 1), "#e5e7eb", "hatch"))
        parca.append((span(aylar[val[0]], aylar[val[-1]]), RENK["VALIDASYON"], None))
        satir.append((f"CV Fold {i}", parca))
    satir.append(("Üretim", [(span(olgun_df["AY"].min(), olgun_df["AY"].max()), "#bbf7d0", None),
                             (span(TAHMIN["AY"].min(), TAHMIN["AY"].max()), RENK["TAHMIN"], None)]))
    fig, ax = plt.subplots(figsize=(16, 4.8))
    for y, (ad, parcalar) in enumerate(reversed(satir)):
        for (x, gen), renk, ek in parcalar:
            ax.broken_barh([(x, gen)], (y - 0.35, 0.7), facecolors=renk, edgecolor="white",
                           hatch="///" if ek == "hatch" else None)
            if isinstance(ek, str) and ek != "hatch":
                ax.text(x + gen / 2, y, ek, ha="center", va="center", fontsize=9, fontweight="bold",
                        color="white" if renk != ARKA["EGITIM"] else "#334155")
    ax.set_yticks(range(len(satir))); ax.set_yticklabels([s[0] for s in reversed(satir)])
    ax.xaxis_date(); ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")); ax.grid(axis="x", alpha=0.3)
    ax.legend(handles=[Patch(color="#bfdbfe", label="Fold eğitimi"),
                       Patch(facecolor="#e5e7eb", hatch="///", label="Dışlanan (hedef henüz olgunlaşmamış)"),
                       Patch(color=RENK["VALIDASYON"], label="Validasyon"), Patch(color=RENK["OOT"], label="OOT"),
                       Patch(color="#bbf7d0", label="Final eğitim"), Patch(color=RENK["TAHMIN"], label="Tahmin")],
              loc="upper left", fontsize=8, ncol=6, bbox_to_anchor=(0, -0.12))
    ax.set_title(f"Veri bölmesi ve walk-forward yapı (champion bilgi ayı = {CHAMP['bilgi_ay']})",
                 fontweight="bold", loc="left")
    plt.tight_layout(); plt.show()


def model_karsilastirma():
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.3))
    p = PERF.set_index(["MODEL", "BOLGE"])
    x = np.arange(2)
    for i, (rol, renk) in enumerate([("CHAMPION", RENK["TAHMIN"]), ("CHALLENGER", RENK["CHALLENGER"])]):
        v = [p.loc[(rol, b), "WMAPE"] for b in ["VALIDASYON", "OOT"]]
        bars = axs[0].bar(x + (i - 1) * 0.27, v, 0.27, color=renk, label=rol.title())
        axs[0].bar_label(bars, labels=[sayi(t, 1) for t in v], fontsize=8)
    dv = [p.loc[("CHAMPION", b), "DUMMY_WMAPE"] for b in ["VALIDASYON", "OOT"]]
    bars = axs[0].bar(x + 0.27, dv, 0.27, color=RENK["DUMMY"], label="Naif ortalama")
    axs[0].bar_label(bars, labels=[sayi(t, 1) for t in dv], fontsize=8)
    axs[0].set_xticks(x, ["Validasyon", "OOT"]); axs[0].set_title("WMAPE (düşük = iyi)", fontweight="bold")
    axs[0].legend(fontsize=8)
    for j, (kol, baslik) in enumerate([("VARYANT", "Trend / mevsim etkisi"), ("BILGI_AY", "Öncül olgunluğu (ay)")], 1):
        t = OZET_VARYANT if kol == "VARYANT" else OZET_BILGI_AY
        t = t.dropna(subset=["EN_IYI_CV_WMAPE"])
        renk = [RENK["TAHMIN"] if (str(k) == "Yok" or k == CHAMP["bilgi_ay"]) else "#cbd5e1" for k in t.index]
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
    ref = DONUSUM.loc[DONUSUM.index < str(VAL_BAS.year + 1)]
    for c, renk in zip(oran_onc, [RENK["TAHMIN"], RENK["OOT"], RENK["CHALLENGER"], RENK["VALIDASYON"]]):
        kol = f"HEDEF/{kisa_ad(c)}"
        endeks = DONUSUM[kol] / ref[kol].mean() * 100
        ax.plot(DONUSUM.index, endeks, "o-", lw=2.2, color=renk, label=f"{HK} / {kisa_ad(c)}")
        for xx, v in zip(DONUSUM.index, endeks):
            ax.annotate(f"{v:.0f}", (xx, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=renk)
    ax.axhline(100, color="#9ca3af", ls="--", lw=1)
    ax.set_title(f"Erken gecikmenin batığa dönüşüm oranı (endeks, {DONUSUM.index[0][:4]}–{VAL_BAS.year} ort. = 100)",
                 fontweight="bold", loc="left")
    ax.grid(alpha=0.25); ax.legend(fontsize=9)
    plt.tight_layout(); plt.show()


zaman_cizelgesi()
ana_grafik("CHAMPION")
ana_grafik("CHALLENGER")
model_karsilastirma()
donusum_grafigi()


# %% [12] Sunum özeti (tek ekran)
p = PERF.set_index(["MODEL", "BOLGE"])
print("=" * 100)
print(f"L12M90 ERKEN UYARI MODELİ | hedef {HEDEF} | bilgi ayı {CONFIG['BILGI_AYI']}")
print("=" * 100)
print(f"Veri          : {d['AY'].min()} – {d['AY'].max()} ({len(d)} kohort) | olgun {len(olgun_df)} | tahmin {len(TAHMIN)}")
for b, r_ in VERI_OZET.iterrows():
    print(f"  {b:<11}: {r_['BAS']} – {r_['SON']} | N={r_['N']} | hedef ort. {yuzde(r_['HEDEF_ORT'])}")
print(f"Validasyon    : {VAL_BAS} – {cv_df['AY'].max()} ({CONFIG['N_SPLITS']} fold x {CONFIG['TEST_SIZE']} kohort, walk-forward)")
print(f"Aday havuzu   : {len(HAVUZ)} değişken ({len(ONCULLER)} öncül + {len(ORTAK_ALANLAR)} başvuru) | "
      f"{len(KOMBS)} kombinasyon | {len(grid)} varyant | geçerli {int(grid['GECERLI'].sum())}")
print(f"Model aileleri: {', '.join(sorted(grid['MODEL'].unique()))}")
print(f"Trend/mevsim  : " + " | ".join(f"{k}: {sayi(v)}" for k, v in OZET_VARYANT["EN_IYI_CV_WMAPE"].items()))
print(f"Bilgi ayı     : " + " | ".join(f"{k}. ay: {sayi(v)}" for k, v in OZET_BILGI_AY["EN_IYI_CV_WMAPE"].dropna().items()))
print(f"Naif referans : CV WMAPE {sayi(DUMMY['CV_WMAPE'])}")
for rol, spec in SECIMLER.items():
    print("-" * 100)
    print(spec["ad"])
    print(f"  Formül      : {FORMUL.get(rol, '-')}")
    for b in ["VALIDASYON", "OOT"]:
        q = p.loc[(rol, b)]
        print(f"  {b:<11}: WMAPE {sayi(q['WMAPE'])} | sapma {q['SAPMA_yp']:+.3f} yp | korelasyon {sayi(q['KORELASYON'], 3)} "
              f"| yön {q['YON']} | beceri {sayi(q['BECERI'])} (naif {sayi(q['DUMMY_WMAPE'])})")
    o = OVERLAY[rol]
    print(f"  Overlay     : {o['deger'] * 100:+.3f} yp | {o['ayni_isaret']}/{o['n']} aynı işaret | "
          f"{'uygulanıyor' if o['uygula'] else 'uygulanmıyor'}")
print("-" * 100)
print("Tahminler (champion, ham -> düzeltilmiş | kısmi | chain-ladder):")
for _, t in TAHMIN.iterrows():
    print(f"  {t['DONEM_1']} ({int(t['GOZLENEN_AY'])} ay): {yuzde(t['CHAMPION_HAM'])} -> {yuzde(t['CHAMPION_DUZ'])} | "
          f"kısmi {yuzde(t['KISMI_GERCEK'])} | CL {yuzde(t.get('CHAIN_LADDER', np.nan))}")
print("-" * 100)
print("Testler:", " | ".join(f"{a}: {b}" for a, b in TESTLER[["TEST", "SONUC"]].values[1:5]))
print("Dönüşüm  :", " | ".join(f"{i}: " + ", ".join(f"{k.split('/')[1]}={v:.2f}" for k, v in r_.items() if k.startswith("HEDEF/"))
                               for i, r_ in DONUSUM.iterrows()))
print("=" * 100)
# ADETSEL için: CONFIG["HEDEF_TUR"] = "ADETSEL" yapıp Bölüm 2'den itibaren tekrar çalıştır.
