# DÜZELTME: her olgun kohort (validasyon ve OOT), KENDİ tahmin anında (AY + bilgi ayı) hedefi belli olan
# kohortlarla yeniden eğitilen modelle tahmin edilir. Tek fit yok.
olgun_df = d[d["HEDEF_KULLANILABILIR"]].copy()


def wf_tahmin(spec):
    kol, kayit = spec["kolonlar"], []
    for i in olgun_df.index:
        c = olgun_df.loc[i]
        tahmin_ay = c["AY"] + spec["bilgi_ay"]
        eg = olgun_df[olgun_df["HEDEF_ERISIM"] < tahmin_ay].dropna(subset=kol)
        if len(eg) < MIN_TRAIN or c[kol].isna().any():
            continue
        m = pipe_fit(model_fabrika(spec["aile"], spec["params"]), eg[kol], eg[HEDEF], agirlik_al(eg))
        kayit.append({"DONEM_1": int(c["DONEM_1"]), "AY": c["AY"], "BOLGE": c["BOLGE"], "TAHMIN_AY": tahmin_ay,
                      "HEDEF_ERISIM": c["HEDEF_ERISIM"], "GERCEK": float(c[HEDEF]),
                      "TAHMIN": float(m.predict(olgun_df.loc[[i], kol])[0]),
                      "DUMMY": float(eg[HEDEF].mean()), "EGITIM_N": len(eg), "EGITIM_SON": str(eg["AY"].max())})
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
            "YON": f"{int((np.sign(dg) == np.sign(dp)).sum())}/{len(dg)}",
            "POZITIF_SAPMA": f"{int((p > g).sum())}/{len(g)}"}


VAL_BAS = AY_CV[fold_al(0)[0][1][0]]
WF, OVERLAY, PERF, SIZINTI = {}, {}, [], []
for rol, spec in MODELLER.items():
    w = wf_tahmin(spec)
    WF[rol] = w
    for bolge, maske in [("VALIDASYON", (w["BOLGE"] == "GELISTIRME") & (w["AY"] >= VAL_BAS)),
                         ("OOT", w["BOLGE"] == "OOT")]:
        s = w[maske]
        mm, dm = metrik(s["GERCEK"], s["TAHMIN"]), metrik(s["GERCEK"], s["DUMMY"])
        PERF.append({"MODEL": rol, "BOLGE": bolge, **mm, "DUMMY_WMAPE": dm["WMAPE"],
                     "BECERI": 1 - mm["WMAPE"] / dm["WMAPE"]})
    o1 = w[w["BOLGE"] == "OOT"].iloc[0]
    SIZINTI.append({"MODEL": rol, "BILGI_AY": spec["bilgi_ay"], "ILK_OOT_KOHORT": str(o1["AY"]),
                    "TAHMIN_ANI": str(o1["TAHMIN_AY"]), "EGITIM_SON_KOHORT": o1["EGITIM_SON"],
                    "EGITIM_N": o1["EGITIM_N"]})
    son = w[w["HEDEF_ERISIM"] < BILGI].sort_values("HEDEF_ERISIM").tail(CONFIG["OVERLAY_PENCERE"])
    ayni = int(max((son["ARTIK"] > 0).sum(), (son["ARTIK"] < 0).sum()))
    OVERLAY[rol] = {"deger": float(son["ARTIK"].mean()), "ayni_isaret": ayni, "n": len(son),
                    "donem": f"{son['DONEM_1'].min()}-{son['DONEM_1'].max()}",
                    "uygula": ayni >= CONFIG["OVERLAY_MIN_AYNI_ISARET"]}

PERF = pd.DataFrame(PERF)
print("Sızıntı kontrolü (eğitimin son kohortu + 12 < tahmin anı olmalı):")
display(pd.DataFrame(SIZINTI))
print(f"Validasyon: {VAL_BAS} – {AY_CV.max()} | OOT: {d.loc[oot_idx, 'AY'].min()} – {d.loc[oot_idx, 'AY'].max()}")
display(PERF.style.format({"WMAPE": "{:.2f}", "MAE_yp": "{:.3f}", "RMSE_yp": "{:.3f}", "SAPMA_yp": "{:+.3f}",
                           "R2": "{:.3f}", "KORELASYON": "{:.3f}", "DUMMY_WMAPE": "{:.2f}", "BECERI": "{:.3f}"},
                          decimal=","))
for rol, o in OVERLAY.items():
    print(f"{rol}: overlay {o['deger'] * 100:+.3f} yp | son {o['n']} olgun kohortun {o['ayni_isaret']}'i aynı "
          f"işaret ({o['donem']}) -> {'UYGULANIR' if o['uygula'] else 'uygulanmaz'}")
