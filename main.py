# Kural: kohort c'nin her girdisi, c'nin tahmin anında sistemde olan veriyle üretilir.
#   - X_Y öncülü, kohort j için ancak j + Y < T ise bilinir (T = verinin geldiği ay)
#   - Ortalama: sadece ÖNCEKİ kohortlar (shift(1)), limit ağırlıklı, tam pencere. Lead YASAK.
HAM_HAVUZ = ONCULLER + ORTAK_ALANLAR
KOLON_AYI = {c: 0 for c in ORTAK_ALANLAR + TAKVIM_KOL}
KOLON_AYI.update({oncul_kolon(k): oncul_ay(k) for k in ONCU_KODLARI})


def ortalama_uret(df, T):
    """T ayının başında bilinen veriyle geçmiş kohort ortalamaları. Dönüş: DataFrame."""
    x = df.sort_values("AY").copy()
    for k in ONCU_KODLARI:                                                   # T'de tamamlanmamış öncül -> NaN
        x.loc[x["AY"] + oncul_ay(k) >= T, oncul_kolon(k)] = np.nan
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
            yeniden = ortalama_uret(gecmis, c_ay + Y + 1)                    # Y. ay verisinin geldiği ay
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
