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
