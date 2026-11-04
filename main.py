# Bölüm 1–14'e DOKUNMAZ: model, seçim ve OOT aynı kalır. Yeni kohortlar ayrı bir tabloya (d_tum) eklenir,
# öncül/ortalama kuralları (Bölüm 3–4) aynen uygulanır, final modeller tüm olgun veriyle kurulup tahmin yapılır.
# Kaynaklar (2026-11-04): kohort raporu (öncüller, ADET, LIMIT), Excel (KKB, MM, PD), DONEM pivotu (90_4/5/6).
# Oranlar YÜZDE yazılır, kod /100 yapar. Raporda kısmi olan öncüller (bilgi ayında tamamlanmamış) NaN yapılır.
YENI_DONEMLER = [202510, 202511, 202512, 202601, 202602, 202603, 202604, 202605, 202606, 202607, 202608]
YENI_BASVURU = {   # DONEM: (ADET, LIMIT, KKB, MM_SKOR %, KALIBRE_PD %)
    202510: (147970, 7200471490, 1370.24, 10.31, 1.83),
    202511: (140430, 7196514711, 1373.49, 10.28, 1.79),
    202512: (160275, 9033999394, 1371.84, 10.28, 1.75),
    202601: (156526, 9542190375, 1382.34, 9.97, 1.66),
    202602: (150351, 9251086456, 1399.88, 9.44, 1.61),
    202603: (174853, 10965123843, 1415.94, 9.05, 1.57),
    202604: (164257, 9933086144, 1407.02, 9.29, 1.62),
    202605: (137114, 7746839735, 1380.53, 9.94, 1.70),
    202606: (158690, 9144006550, 1367.22, 10.27, 1.82),
    202607: (156361, 9701897450, 1351.88, 10.92, 2.22),
    202608: (137122, 8437839300, 1350.78, 11.03, 2.13),
}
# Öncüller (%), YENI_DONEMLER sırasıyla (rapordaki değerler; kısmi olanları Bölüm 3 eler)
YENI_ONCUL = {
    "20_2": [1.11, 1.07, 1.08, 1.09, 0.80, 0.92, 1.03, 1.01, 1.21, 1.66, 1.83],   # TUTARSAL_2M20
    "30_3": [1.33, 1.29, 1.23, 1.31, 0.97, 1.08, 1.20, 1.30, 1.48, 2.11, 1.47],   # TUTARSAL_3M30
    "60_3": [0.28, 0.24, 0.27, 0.30, 0.23, 0.19, 0.24, 0.25, 0.28, 0.37, 0.08],   # TUTARSAL_3M60
    "30_4": [2.15, 1.92, 1.94, 2.08, 1.63, 1.80, 2.03, 2.10, 2.36, 2.74, 1.47],   # TUTARSAL_4M30
    "60_4": [0.68, 0.66, 0.67, 0.77, 0.54, 0.56, 0.63, 0.63, 0.76, 0.67, 0.08],   # TUTARSAL_4M60
    "30_5": [2.84, 2.58, 2.73, 2.76, 2.32, 2.58, 2.78, 2.98, 2.79, 2.74, 1.47],   # TUTARSAL_5M30
    "60_5": [0.98, 0.90, 0.98, 1.10, 0.77, 0.79, 0.93, 0.97, 0.90, 0.67, 0.08],   # TUTARSAL_5M60
    "30_6": [3.57, 3.28, 3.63, 3.56, 3.01, 3.26, 3.54, 3.47, 2.79, 2.74, 1.47],   # TUTARSAL_6M30
    "30_2": None,   # DOLDUR: TUTARSAL_2M30 (champion bunu kullanıyor) – raporda görünmedi
    "60_6": None,   # TUTARSAL_6M60 – raporda görünmedi
}
# 90+ olgunlaşma kolonları (chain-ladder için, GİRDİ DEĞİL): 202401'den itibaren aylık, 202510+ dahil
K90_202401 = {
    "90_4": [0.18, 0.18, 0.18, 0.11, 0.23, 0.25, 0.25, 0.25, 0.18, 0.29, 0.23, 0.14, 0.23, 0.13, 0.18, 0.20, 0.12,
             0.13, 0.20, 0.21, 0.21, 0.22, 0.15, 0.20, 0.20, 0.19, 0.15, 0.19, 0.20, 0.19, 0.03, 0.00],
    "90_5": [0.44, 0.43, 0.43, 0.35, 0.56, 0.60, 0.58, 0.63, 0.53, 0.64, 0.53, 0.40, 0.47, 0.39, 0.41, 0.45, 0.36,
             0.36, 0.45, 0.50, 0.52, 0.46, 0.41, 0.50, 0.55, 0.41, 0.41, 0.46],
    "90_6": [0.63, 0.60, 0.62, 0.55, 0.78, 0.81, 0.82, 0.83, 0.70, 0.84, 0.69, 0.57, 0.60, 0.58, 0.59, 0.63, 0.54,
             0.55, 0.65, 0.68, 0.66, 0.61, 0.55, 0.70, 0.72, 0.53, 0.52],
}


# --- 1) Ham veriye ekle (df_raw değişmez; birleşik kopya)
yeni = pd.DataFrame({"DONEM_1": YENI_DONEMLER})
b_ = pd.DataFrame.from_dict(YENI_BASVURU, orient="index",
                            columns=["BASVURU_ADEDI", "TOPLAM_ONAYLANAN_LIMIT", KKB_KOL, "MM_SKOR",
                                     "KALIBRE_PD_AGIRLIKLI"])
yeni = yeni.join(b_, on="DONEM_1")
yeni[["MM_SKOR", "KALIBRE_PD_AGIRLIKLI"]] /= 100
for kod, v in YENI_ONCUL.items():
    v = v or []
    yeni[oncul_kolon(kod)] = [x / 100 for x in v] + [np.nan] * (len(yeni) - len(v))
df_tum_raw = pd.concat([df_raw, yeni], ignore_index=True)
ay_ = pd.PeriodIndex(pd.to_datetime(df_tum_raw["DONEM_1"].astype(str), format="%Y%m"), freq="M")
for kod, v in K90_202401.items():
    aylar_ = pd.period_range("2024-01", periods=len(v), freq="M")
    df_tum_raw[oncul_kolon(kod)] = pd.Series(ay_).map(dict(zip(aylar_, np.array(v) / 100))).to_numpy()

# --- 2) Bölüm 3 ve 4'ün aynı fonksiyonlarıyla hazırla
d_tum, _, _, K90_TUM = veri_hazirla(df_tum_raw, CONFIG)
for kol in K90_TUM:                                   # bilgi ayında tamamlanmamış 90_m -> NaN
    m_ = int(kol.split("_")[3])
    d_tum.loc[(d_tum["AY"] + m_ + CONFIG["RAPOR_GECIKME_AY"]) >= BILGI, kol] = np.nan
d_tum = d_tum.join(ortalama_uret(d_tum, BILGI))

# Kontrol: olgun kohortların girdileri Bölüm 1–14'teki d ile birebir aynı olmalı (yeni veri geçmişi değiştirmez)
_kol = [c for c in HAVUZ if c in d.columns]
_fark = np.nanmax(np.abs(d_tum.loc[d.index, _kol].to_numpy(float) - d[_kol].to_numpy(float)))
print(f"Olgun kohort girdileri d ile aynı mı? maks fark = {_fark:.1e}", "-> OK" if _fark < 1e-12 else "-> KONTROL ET")
print(f"Yeni kohortlar: {len(yeni)} ({YENI_DONEMLER[0]} – {YENI_DONEMLER[-1]}) | ortalama limit 202509 "
      f"{d_tum['ORTALAMA_LIMIT'].iloc[len(d) - 1]:,.0f} -> 202510 {d_tum['ORTALAMA_LIMIT'].iloc[len(d)]:,.0f}")

# --- 3) Final modeller (tüm olgun veri) ve tahmin
URETIM, FORMUL = {}, {}
gelecek = d_tum[~d_tum["HEDEF_KULLANILABILIR"]].sort_values("AY")
TAHMIN = gelecek[["DONEM_1", "AY", "GOZLENEN_AY"]].reset_index(drop=True)
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
    if len(X) and not ok.any():
        eksik_ = [kisa_ad(c) for c in kol_ if X[c].isna().all()]
        print(f"UYARI {rol}: tahmin üretilemedi, eksik girdi -> {eksik_} (YENI_ONCUL'u doldur)")

# --- 4) Chain-ladder (modelden bağımsız): bilinen en geç 90_m / olgunlaşma oranı (son 12 olgun kohort)
olgun_tum = d_tum[d_tum["HEDEF_KULLANILABILIR"]]
OLGUNLASMA, OLG_STD = {}, {}
for k in K90_TUM:
    m_ = int(k.split("_")[3])
    r_ = (olgun_tum[k] / olgun_tum[HEDEF]).dropna().tail(12)
    if len(r_) >= 6:
        OLGUNLASMA[m_], OLG_STD[m_] = float(r_.mean()), float(r_.std())
cl, kay = np.full(len(gelecek), np.nan), [None] * len(gelecek)
for j, (_, r) in enumerate(gelecek.iterrows()):
    for m_ in sorted(OLGUNLASMA, reverse=True):
        v = r.get(oncul_kolon(f"90_{m_}"))
        if pd.notna(v):
            cl[j], kay[j] = v / OLGUNLASMA[m_], f"90_{m_}"
            break
TAHMIN["CHAIN_LADDER"], TAHMIN["CL_KAYNAK"] = cl, kay
print("Olgunlaşma oranı (90_m / 90_12, son 12 olgun kohort):",
      {f"90_{m_}": f"{OLGUNLASMA[m_]:.3f} ± {OLG_STD[m_]:.3f}" for m_ in sorted(OLGUNLASMA)})
display(TAHMIN.drop(columns="AY").style.format(
    {**{c: "{:.2%}" for c in TAHMIN.columns if c.endswith(("_HAM", "_DUZ")) or c == "CHAIN_LADDER"},
     "DONEM_1": "{:.0f}"}, decimal=",", na_rep="-"))
