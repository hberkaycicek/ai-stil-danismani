# Gelecek Tahmini: olgunlaşmamış kohortlar df_tahmin'den (manuel giriş yok)
# Bölüm 1–14'e DOKUNMAZ: model, seçim ve OOT aynı kalır. df_tahmin (202510+) df_raw ile birleştirilir,
# USD limit aynı formülle üretilir (TOPLAM_ONAYLANAN_LIMIT / EFEKTIF_SATIS), öncül/ortalama kuralları
# (Bölüm 3–4) aynen uygulanır: bilgi ayında tamamlanmamış öncül ve 90_m -> NaN, ortalamalar USD ağırlıklı.
LIMIT_USD_KOL = "TOPLAM_ONAYLANAN_LIMIT_USD"

# --- 1) df_tahmin hazırla: dönem, USD limit, kur kontrolü
yeni = df_tahmin.copy()
yeni["DONEM_1"] = pd.to_numeric(yeni["DONEM_1"], errors="raise").astype(int)
yeni = yeni[~yeni["DONEM_1"].isin(df_raw["DONEM_1"])]                  # df_raw ile çakışan dönem alınmaz
yeni = yeni.sort_values("DONEM_1").reset_index(drop=True)
if yeni.empty:
    raise ValueError("df_tahmin'de df_raw sonrası kohort yok.")

limit_tl_ = pd.to_numeric(yeni["TOPLAM_ONAYLANAN_LIMIT"], errors="raise")
kur_ = pd.to_numeric(yeni["EFEKTIF_SATIS"], errors="raise")
if not np.isfinite(limit_tl_).all() or (limit_tl_ < 0).any():
    raise ValueError("df_tahmin: TL limitleri sonlu ve negatif olmayan değerler olmalı.")
if not np.isfinite(kur_).all() or (kur_ <= 0).any():
    raise ValueError(f"df_tahmin: EFEKTIF_SATIS eksik/geçersiz -> {yeni.loc[~(np.isfinite(kur_) & (kur_ > 0)), 'DONEM_1'].tolist()}")
kohort_ayi_ = pd.to_datetime(yeni["DONEM_1"].astype(str), format="%Y%m").dt.to_period("M")
kur_ayi_ = pd.to_datetime(yeni["KUR_TARIHI"], errors="raise").dt.to_period("M")
if (kur_ayi_.isna() | kur_ayi_.ne(kohort_ayi_)).any():
    raise ValueError(f"df_tahmin: kur ayı kohort ayıyla eşleşmiyor -> "
                     f"{yeni.loc[kur_ayi_.isna() | kur_ayi_.ne(kohort_ayi_), 'DONEM_1'].tolist()}")
yeni["TOPLAM_ONAYLANAN_LIMIT"] = limit_tl_
yeni[LIMIT_USD_KOL] = limit_tl_ / kur_

df_tum_raw = pd.concat([df_raw, yeni[[c for c in yeni.columns if c in df_raw.columns or c.startswith("KK_TAKIP_90_")]]],
                       ignore_index=True).sort_values("DONEM_1").reset_index(drop=True)
YENI_DONEMLER = yeni["DONEM_1"].tolist()

# --- 2) Bölüm 3 ve 4'ün aynı fonksiyonlarıyla hazırla
d_tum, _, _, K90_TUM = veri_hazirla(df_tum_raw, CONFIG)
for kol in K90_TUM:                                   # bilgi ayında tamamlanmamış 90_m -> NaN (chain-ladder, GİRDİ DEĞİL)
    m_ = int(kol.split("_")[3])
    d_tum.loc[(d_tum["AY"] + m_ + CONFIG["RAPOR_GECIKME_AY"]) >= BILGI, kol] = np.nan
d_tum = d_tum.join(ortalama_uret(d_tum, BILGI))

# Kontrol: olgun kohortların girdileri Bölüm 1–14'teki d ile birebir aynı olmalı (yeni veri geçmişi değiştirmez)
_kol = [c for c in HAVUZ if c in d.columns]
_fark = np.nanmax(np.abs(d_tum.loc[d.index, _kol].to_numpy(float) - d[_kol].to_numpy(float)))
print(f"Olgun kohort girdileri d ile aynı mı? maks fark = {_fark:.1e}", "-> OK" if _fark < 1e-12 else "-> KONTROL ET")
print(f"Yeni kohortlar (df_tahmin): {len(yeni)} ({YENI_DONEMLER[0]} – {YENI_DONEMLER[-1]}) | ağırlık: "
      f"{CONFIG['ORTALAMA_AGIRLIK_KOLON']} | ortalama limit 202509 {d_tum['ORTALAMA_LIMIT'].iloc[len(d) - 1]:,.0f} "
      f"-> {YENI_DONEMLER[0]} {d_tum['ORTALAMA_LIMIT'].iloc[len(d)]:,.0f}")
display(yeni[["DONEM_1", "KUR_TARIHI", "EFEKTIF_SATIS", "TOPLAM_ONAYLANAN_LIMIT", LIMIT_USD_KOL]]
        .style.format({"EFEKTIF_SATIS": "{:.4f}", "TOPLAM_ONAYLANAN_LIMIT": "{:,.0f}", LIMIT_USD_KOL: "{:,.0f}"})
        .hide(axis="index"))

# --- 3) Final modeller (tüm olgun veri) ve tahmin
URETIM, FORMUL = {}, {}
gelecek = d_tum[~d_tum["HEDEF_KULLANILABILIR"]].sort_values("AY")
TAHMIN = gelecek[["DONEM_1", "AY", "GOZLENEN_AY"]].reset_index(drop=True)
GIRDI_DURUM = gelecek[["DONEM_1", "GOZLENEN_AY"]].reset_index(drop=True)
for c in dict.fromkeys(c for s_ in MODELLER.values() for c in taban(s_["kolonlar"])):
    GIRDI_DURUM[kisa_ad(c)] = np.where(gelecek[c].notna().to_numpy(), "✓", "–")
print("Girdi durumu (✓ = bilgi ayında hazır, – = henüz tamamlanmadı):")
display(GIRDI_DURUM.style.hide(axis="index"))

# Öncül köprüsü: bilgi ayında henüz tamamlanmamış öncül (ör. 202607'nin 60_3'ü), aynı kohortun bilinen en geç
# öncülünden oranla tahmin edilir. Oran = son 12 kohortta (iki öncülün de bugün bilindiği) ortalama hedef/kaynak.
# Bu tahminler "köprü" olarak işaretlenir; gerçek öncül gelince normal tahmin yerine geçer.
KOPRU_PENCERE = 12
KOPRU_ORAN = {}


def kopru_orani(hedef_kol, kaynak_kol):
    if (hedef_kol, kaynak_kol) not in KOPRU_ORAN:
        r_ = d_tum.dropna(subset=[hedef_kol, kaynak_kol]).sort_values("AY").tail(KOPRU_PENCERE)
        oran_ = r_[hedef_kol] / r_[kaynak_kol].where(r_[kaynak_kol] > 0)
        KOPRU_ORAN[(hedef_kol, kaynak_kol)] = (float(oran_.mean()), float(oran_.std() / oran_.mean()), len(oran_))
    return KOPRU_ORAN[(hedef_kol, kaynak_kol)]


def oncul_kopru(satir, hedef_kol):
    """Eksik öncül -> (tahmini değer, açıklama). Kaynak: aynı kohortun bilinen en geç öncülü, oranı en istikrarlı olan."""
    if hedef_kol not in ONCULLER:
        return np.nan, None
    adaylar = [c for c in ONCULLER if KOLON_AYI[c] < KOLON_AYI[hedef_kol] and pd.notna(satir[c])]
    if not adaylar:
        return np.nan, None
    en_gec = max(KOLON_AYI[c] for c in adaylar)
    kaynak = min([c for c in adaylar if KOLON_AYI[c] == en_gec], key=lambda c: kopru_orani(hedef_kol, c)[1])
    oran_, cv_, n_ = kopru_orani(hedef_kol, kaynak)
    return satir[kaynak] * oran_, f"{kisa_ad(hedef_kol)}≈{kisa_ad(kaynak)}×{oran_:.3f}"


for rol, spec in MODELLER.items():
    kol_ = spec["kolonlar"]
    uret_df = olgun_df.dropna(subset=kol_)
    m = pipe_fit(model_fabrika(spec["aile"], spec["params"]), uret_df[kol_], uret_df[HEDEF], agirlik_al(uret_df))
    URETIM[rol] = m
    if hasattr(m[-1], "coef_"):
        coef = np.ravel(m[-1].coef_) / m[0].scale_
        b0 = float(np.ravel([m[-1].intercept_])[0] - np.dot(coef, m[0].mean_))
        FORMUL[rol] = f"{HK} = {b0:.6f} " + " ".join(f"{c:+.6g}·{kisa_ad(k)}" for k, c in zip(kol_, coef))
    X = gelecek[kol_].copy()
    ok = X.notna().all(axis=1).to_numpy()
    kopru = [None] * len(X)
    for j in np.where(~ok)[0]:                                          # eksik öncül -> köprü
        satir, acik = gelecek.iloc[j], []
        for c in [c for c in kol_ if pd.isna(X.iloc[j][c])]:
            v_, a_ = oncul_kopru(satir, c)
            if pd.notna(v_):
                X.iloc[j, X.columns.get_loc(c)] = v_
                acik.append(a_)
        if X.iloc[j].notna().all():
            kopru[j] = "köprü: " + ", ".join(acik)
    tam = X.notna().all(axis=1).to_numpy()
    ham = np.full(len(X), np.nan)
    if tam.any():
        ham[tam] = m.predict(X[tam])
    ov = OVERLAY[rol]["deger"] if OVERLAY[rol]["uygula"] else 0.0
    TAHMIN[f"{rol}_HAM"] = np.clip(ham, 0, 1)
    TAHMIN[f"{rol}_DUZ"] = np.clip(ham - ov, 0, 1)
    TAHMIN[f"{rol}_GIRDI"] = np.where(ok, "tam", np.where(tam, kopru, "eksik"))
    print(f"{spec['ad']} | {int(ok.sum())} tam + {int(tam.sum() - ok.sum())} köprü / {len(X)} kohort tahmin edildi")
    for j in np.where(tam & ~ok)[0]:
        print(f"    {TAHMIN.loc[j, 'DONEM_1']}: {kopru[j]}")
    if len(X) and not tam.any():
        print(f"UYARI {rol}: tahmin üretilemedi, eksik girdi -> {[kisa_ad(c) for c in kol_ if X[c].isna().all()]}")

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
      {f"90_{m_}": f"{OLGUNLASMA[m_]:.3f} ± {OLG_STD[m_]:.3f}" for m_ in sorted(OLGUNLASMA)} or "90_m kolonu yok")
print("GIRDI: tam = bütün girdiler bilgi ayında hazır | köprü = eksik öncül oranla tahmin edildi (daha belirsiz)")
display(TAHMIN.drop(columns="AY").style.format(
    {**{c: "{:.2%}" for c in TAHMIN.columns if c.endswith(("_HAM", "_DUZ")) or c == "CHAIN_LADDER"},
     "DONEM_1": "{:.0f}"}, decimal=",", na_rep="-"))
