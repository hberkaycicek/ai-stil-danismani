# Bölüm 1–14'e DOKUNMAZ: model, seçim ve OOT aynı kalır. Yeni kohortlar ayrı bir tabloya (d_tum) eklenir,
# öncül/ortalama kuralları (Bölüm 3–4) aynen uygulanır, final modeller tüm olgun veriyle kurulup tahmin yapılır.
# Oranlar YÜZDE yazılır, kod /100 yapar. Bilgi ayında tamamlanmamış öncüller otomatik NaN yapılır.
# Kaynak (2026-11-05): BDS Vintage raporu, ADETSEL_xMyy kolonları. Başvuru metrikleri tutarsal ile aynı.
# 202609 / 202610 alınmadı: hiçbir öncül tamamlanmadı.
YENI_DONEMLER = [202510, 202511, 202512, 202601, 202602, 202603, 202604, 202605, 202606, 202607, 202608]
YENI_BASVURU = {   # DONEM: (ADET, LIMIT, KKB, MM_SKOR %, KALIBRE_PD %) – tutarsal ile aynı
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
# ADETSEL öncüller (%), YENI_DONEMLER sırasıyla (rapordaki değerler; kısmi olanları Bölüm 3 kuralı eler)
YENI_ONCUL = {
    "20_2": [2.81, 2.91, 2.75, 2.52, 2.36, 2.38, 2.68, 2.65, 2.90, 3.40, 3.55],   # ADETSEL_2M20
    "30_3": [3.39, 3.33, 2.86, 3.03, 2.85, 2.97, 3.24, 3.29, 3.45, 4.10, 2.75],   # ADETSEL_3M30
    "60_3": [0.50, 0.49, 0.47, 0.47, 0.47, 0.43, 0.45, 0.45, 0.49, 0.62, 0.16],   # ADETSEL_3M60
    "30_4": [5.26, 4.65, 4.51, 4.91, 4.66, 4.88, 5.22, 5.38, 5.55, 5.23, 2.75],   # ADETSEL_4M30
    "60_4": [1.37, 1.26, 1.25, 1.31, 1.21, 1.25, 1.28, 1.30, 1.35, 1.12, 0.16],   # ADETSEL_4M60
    "30_5": [6.47, 6.07, 6.17, 6.86, 6.46, 6.62, 6.96, 7.37, 6.47, 5.23, 2.75],   # ADETSEL_5M30
    "60_5": [1.92, 1.77, 1.89, 2.01, 1.82, 1.83, 1.91, 1.99, 1.64, 1.12, 0.16],   # ADETSEL_5M60
    "30_6": [7.77, 7.46, 8.03, 8.60, 7.99, 8.13, 8.53, 8.28, 6.47, 5.23, 2.75],   # ADETSEL_6M30
    "60_6": [2.41, 2.39, 2.60, 2.75, 2.47, 2.41, 2.52, 2.30, 1.64, 1.12, 0.16],   # ADETSEL_6M60
    "30_2": None,   # DOLDUR: ADETSEL_2M30 – raporda görünmedi
}
# ADETSEL 90+ olgunlaşma kolonları (chain-ladder, GİRDİ DEĞİL), YENI_DONEMLER sırasıyla.
# Oran hesabı için olgun kohortlarda da (df_raw_adetsel.csv) KK_TAKIP_90_4/5/6_ADETSEL kolonları olmalı.
K90_YENI = {
    "90_4": [0.30, 0.27, 0.28, 0.30, 0.31, 0.26, 0.27, 0.27, 0.30, 0.07, 0.00],   # ADETSEL_4M90
    "90_5": [0.75, 0.72, 0.76, 0.83, 0.73, 0.74, 0.76, 0.80, 0.52, 0.07, 0.00],   # ADETSEL_5M90
    "90_6": [1.04, 1.00, 1.12, 1.17, 1.03, 1.05, 1.09, 0.96, 0.52, 0.07, 0.00],   # ADETSEL_6M90
}

# --- 1) Ham veriye ekle (df_raw değişmez; birleşik kopya)
yeni = pd.DataFrame({"DONEM_1": pd.Series(YENI_DONEMLER, dtype="int64")})
b_ = pd.DataFrame.from_dict(YENI_BASVURU, orient="index",
                            columns=["BASVURU_ADEDI", "TOPLAM_ONAYLANAN_LIMIT", KKB_KOL, "MM_SKOR",
                                     "KALIBRE_PD_AGIRLIKLI"])
yeni = yeni.join(b_, on="DONEM_1")
yeni[["MM_SKOR", "KALIBRE_PD_AGIRLIKLI"]] /= 100
for kod, v in YENI_ONCUL.items():
    v = v or []
    yeni[oncul_kolon(kod)] = [x / 100 for x in v] + [np.nan] * (len(yeni) - len(v))
df_tum_raw = pd.concat([df_raw, yeni], ignore_index=True)
df_tum_raw["DONEM_1"] = df_tum_raw["DONEM_1"].astype(int)
for kod, v in K90_YENI.items():                       # yeni kohortların 90_m değerleri (olgunlar df_raw'dan)
    kol = oncul_kolon(kod)
    if kol not in df_tum_raw:
        df_tum_raw[kol] = np.nan
    df_tum_raw.loc[df_tum_raw["DONEM_1"].isin(YENI_DONEMLER[:len(v)]), kol] = np.array(v) / 100

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
if len(yeni):
    print(f"Yeni kohortlar: {len(yeni)} ({YENI_DONEMLER[0]} – {YENI_DONEMLER[-1]}) | ortalama limit son olgun "
          f"{d_tum['ORTALAMA_LIMIT'].iloc[len(d) - 1]:,.0f} -> ilk yeni {d_tum['ORTALAMA_LIMIT'].iloc[len(d)]:,.0f}")
else:
    print("Yeni kohort girilmedi (YENI_DONEMLER boş) -> tahmin üretilmez; Bölüm 15–17 tahminsiz çalışır.")

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
if OLGUNLASMA:
    print("Olgunlaşma oranı (90_m / 90_12, son 12 olgun kohort):",
          {f"90_{m_}": f"{OLGUNLASMA[m_]:.3f} ± {OLG_STD[m_]:.3f}" for m_ in sorted(OLGUNLASMA)})
else:
    print("Chain-ladder hesaplanmadı: olgun kohortlarda 90_4/5/6 ADETSEL kolonu yok (df_raw_adetsel.csv'ye ekle).")
display(TAHMIN.drop(columns="AY").style.format(
    {**{c: "{:.2%}" for c in TAHMIN.columns if c.endswith(("_HAM", "_DUZ")) or c == "CHAIN_LADDER"},
     "DONEM_1": "{:.0f}"}, decimal=",", na_rep="-"))
