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

if len(TAHMIN):
    pct_kol = [c for c in TAHMIN.columns if c.endswith(("_HAM", "_DUZ")) or c == "CHAIN_LADDER"]
    print(f"=== {HK} TAHMİNLERİ (bilgi ayı {BILGI}) | boş = girdi henüz yok/tamamlanmadı ===")
    display(TAHMIN.drop(columns="AY").style.format({**{c: "{:.2%}" for c in pct_kol}, "DONEM_1": "{:.0f}"},
                                                   decimal=",", na_rep="-"))
else:
    print("Olgunlaşmamış kohort yok -> tahmin üretilmedi (Bölüm 3 bilgisine bak).")

KAYIT = TAHMIN.drop(columns="AY").copy()
KAYIT.insert(0, "KAYIT_TARIHI", pd.Timestamp.today().normalize())
KAYIT.insert(1, "BILGI_AYI", CONFIG["BILGI_AYI"])
KAYIT.insert(2, "CHAMPION", CHAMP["ad"])
# KAYIT.to_csv("L12M90_tahmin_kayit.csv", index=False)   # izleme için
