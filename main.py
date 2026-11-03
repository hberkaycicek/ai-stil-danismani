# DÜZELTME: her bilgi ayı için ayrı fold seti. Bloğun ilk kohortu "AY + bilgi_ayı" anında tahmin
# edilir; eğitimde sadece hedefi o anda BELLİ olan kohortlar (AY + 12 < tahmin anı) kalır.
# 2. ay modelinde 10, 3. ayda 9, 6. ayda 6 kohort eğitimden çıkar.
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
        ilk_tahmin = AY_CV[val[0]] + bilgi_ayi
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


def cv_degerlendir(kolonlar, estimator, foldlar=None):
    kolonlar = list(kolonlar)
    if foldlar is None:
        foldlar = fold_al(bilgi_ayi_hesapla(kolonlar))
    X, y = cv_df[kolonlar].to_numpy(dtype=float), Y_CV
    dolu = ~np.isnan(X).any(axis=1)
    hata, gercek, f_wmape, f_sapma, f_oran = [], [], [], [], []
    d_g, d_t, aralik_disi, n_tr = [], [], 0, []
    for tr, val in foldlar:
        tr = tr[dolu[tr]]                                       # ilk kohortlarda boş olan ortalamalar
        if len(tr) < MIN_TRAIN:
            raise ValueError(f"eğitim yetersiz ({len(tr)} < {MIN_TRAIN})")
        if not dolu[val].all():
            raise ValueError("validasyonda boş değer")
        w = None if CONFIG["SAMPLE_WEIGHT_KOLON"] is None else cv_df.iloc[tr][CONFIG["SAMPLE_WEIGHT_KOLON"]].to_numpy()
        m = pipe_fit(estimator, X[tr], y[tr], w)
        p, p_tr = m.predict(X[val]), m.predict(X[tr])
        n_tr.append(len(tr))
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
        "ARALIK_DISI": aralik_disi, "FOLD1_EGITIM_N": n_tr[0],
    }


DUMMY = cv_degerlendir([HAM_HAVUZ[0]], DummyRegressor(strategy="mean"), fold_al(0))
print(f"Naif referans (geçmiş ortalama): CV_WMAPE {DUMMY['CV_WMAPE']:.2f} | "
      f"EN_KOTU {DUMMY['EN_KOTU_WMAPE']:.2f} | MUTLAK_FOLD_SAPMA {DUMMY['MUTLAK_FOLD_SAPMA']:.4f}")
