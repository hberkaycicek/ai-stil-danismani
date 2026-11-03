p = PERF.set_index(["MODEL", "BOLGE"])
print("=" * 100)
print(f"L12M90 ERKEN UYARI MODELİ v7 | hedef {HEDEF} | bilgi ayı {BILGI}")
print("=" * 100)
print(f"Veri          : {d['AY'].min()} – {d['AY'].max()} ({len(d)} kohort) | olgun {len(olgun_df)} | tahmin {len(TAHMIN)}")
for b, r_ in VERI_OZET.dropna(subset=["N"]).iterrows():
    print(f"  {b:<11}: {r_['BAS']} – {r_['SON']} | N={int(r_['N'])} | hedef ort. {yuzde(r_['HEDEF_ORT'])}")
print(f"Validasyon    : {VAL_BAS} – {AY_CV.max()} ({CONFIG['N_SPLITS']} fold x {CONFIG['TEST_SIZE']} kohort, "
      f"olgunlaşma boşluklu walk-forward)")
print(f"Aday havuzu   : {len(HAM_HAVUZ)} ham + {len(ORTALAMA_KOL)} ortalama | {len(ADAY_TABLO):,} aday | "
      f"{len(grid):,} değerlendirme | geçerli {int(grid['GECERLI'].sum()):,}")
print("Takvim        : " + " | ".join(f"{k}: {sayi(v)}" for k, v in OZET_VARYANT["EN_IYI_CV_WMAPE"].items()))
print("Bilgi ayı     : " + " | ".join(f"{k}. ay: {sayi(v)}" for k, v in OZET_BILGI_AY["EN_IYI_CV_WMAPE"].dropna().items()))
print("Ortalama      : " + " | ".join(f"{k}: {sayi(v)}" for k, v in OZET_ORTALAMA["EN_IYI_CV_WMAPE"].items()))
print(f"Naif referans : CV WMAPE {sayi(DUMMY['CV_WMAPE'])}")
for rol, spec in MODELLER.items():
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
if len(TAHMIN):
    print("-" * 100)
    print("Tahminler (champion ham -> overlay sonrası | kısmi | chain-ladder):")
    for _, t in TAHMIN.iterrows():
        print(f"  {t['DONEM_1']} ({int(t['GOZLENEN_AY'])} ay): {yuzde(t['CHAMPION_HAM'])} -> {yuzde(t['CHAMPION_DUZ'])} | "
              f"kısmi {yuzde(t['KISMI_GERCEK'])} | CL {yuzde(t.get('CHAIN_LADDER', np.nan))}")
print("-" * 100)
print("Karar         :", KARAR_GEREKCE)
print("Testler:", " | ".join(f"{a}: {b}" for a, b in TESTLER[["TEST", "SONUC"]].values[:5]))
print("=" * 100)
