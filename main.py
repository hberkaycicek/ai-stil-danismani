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
                kayit_.append(kayit_ekle(cv_degerlendir(kolonlar, model_fabrika(aile), foldlar), kolonlar, aile, None))
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

GOSTER = ["GIRDILER", "MODEL", "BILGI_AY", "TOPLAM_K", "TREND", "MEVSIM", "ORTALAMA", "CV_WMAPE",
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
