from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor

kol = CHAMP["kolonlar"]
t_df = olgun_df.dropna(subset=kol)
Xc = sm.add_constant(t_df[kol].astype(float).to_numpy(), has_constant="add")
yv = t_df[HEDEF].astype(float).to_numpy()
ols = sm.OLS(yv, Xc).fit()
m_ch = pipe_fit(model_fabrika(CHAMP["aile"], CHAMP["params"]), t_df[kol], t_df[HEDEF])   # 14'ten bağımsız
resid = m_ch.predict(t_df[kol]) - yv
rho = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
jb = jarque_bera(resid)
p_gel, p_tum = hac_p(kol, cv_df), hac_p(kol, olgun_df)
TESTLER = pd.DataFrame([
    {"TEST": "HAC p (geliştirme, seçimde kullanılan)", "SONUC": ", ".join(f"{kisa_ad(k)}={p:.3f}" for k, p in zip(kol, p_gel)),
     "YORUM": f"<= {CONFIG['HAC_P_ESIK']} anlamlı"},
    {"TEST": "HAC p (tüm olgun veri, OOT dahil)", "SONUC": ", ".join(f"{kisa_ad(k)}={p:.3f}" for k, p in zip(kol, p_tum)),
     "YORUM": "anlamlılık OOT eklenince korunuyor mu?"},
    {"TEST": "R² / Düzeltilmiş R² (tüm olgun veri)", "SONUC": f"{ols.rsquared:.3f} / {ols.rsquared_adj:.3f}", "YORUM": "in-sample"},
    {"TEST": "Durbin-Watson", "SONUC": f"{durbin_watson(resid):.3f}", "YORUM": "2 = otokorelasyon yok"},
    {"TEST": "Efektif gözlem", "SONUC": f"{len(yv) * (1 - rho) / (1 + rho):.1f} / {len(yv)}", "YORUM": f"lag-1 rho {rho:.2f}"},
    {"TEST": "Ljung-Box p (lag 1/3/6)", "SONUC": " / ".join(f"{p:.3f}" for p in acorr_ljungbox(resid, lags=[1, 3, 6])["lb_pvalue"]),
     "YORUM": "<0,05 otokorelasyon"},
    {"TEST": "Breusch-Pagan p", "SONUC": f"{het_breuschpagan(resid, Xc)[1]:.3f}", "YORUM": ">0,05 sabit varyans"},
    {"TEST": "Jarque-Bera p", "SONUC": f"{jb[1]:.3f} (çarp. {jb[2]:.2f}, bas. {jb[3]:.2f})", "YORUM": ">0,05 normal"},
    {"TEST": "Ramsey RESET p", "SONUC": f"{linear_reset(ols, power=2, use_f=True).pvalue:.3f}", "YORUM": ">0,05 doğrusal form yeterli"},
    {"TEST": "VIF", "SONUC": ", ".join(f"{kisa_ad(k)}={variance_inflation_factor(Xc, i + 1):.2f}" for i, k in enumerate(kol))
     if len(kol) > 1 else "-", "YORUM": "<5 iyi, >10 sorunlu"},
])
print(f"=== {CHAMP['ad']} | istatistiksel testler ===")
display(TESTLER)

drift_kol = list(dict.fromkeys([c for s_ in MODELLER.values() for c in taban(s_["kolonlar"])] + [HEDEF]))
DRIFT = pd.DataFrame([{"DEGISKEN": kisa_ad(c), "GEL_ORT": d.loc[gel_idx, c].mean(), "OOT_ORT": d.loc[oot_idx, c].mean(),
                       "DEGISIM_%": (d.loc[oot_idx, c].mean() / d.loc[gel_idx, c].mean() - 1) * 100,
                       "DRIFT_Z": (d.loc[oot_idx, c].mean() - d.loc[gel_idx, c].mean()) / d.loc[gel_idx, c].std()}
                      for c in drift_kol])
print("=== Girdi / hedef kayması (geliştirme -> OOT) ===")
display(DRIFT.style.format({"GEL_ORT": "{:.4f}", "OOT_ORT": "{:.4f}", "DEGISIM_%": "{:+.1f}", "DRIFT_Z": "{:+.2f}"}, decimal=","))

oran_onc = [c for c in drift_kol if c in ONCULLER]
r = olgun_df.assign(DONEM=olgun_df["AY"].dt.year.astype(str) + "-H" + ((olgun_df["AY"].dt.month > 6) + 1).astype(str))
DONUSUM = r.groupby("DONEM").agg(N=(HEDEF, "size"), HEDEF_ORT=(HEDEF, "mean"), **{f"ORT_{kisa_ad(c)}": (c, "mean") for c in oran_onc})
for c in oran_onc:
    DONUSUM[f"HEDEF/{kisa_ad(c)}"] = DONUSUM["HEDEF_ORT"] / DONUSUM[f"ORT_{kisa_ad(c)}"]
print("=== Erken gecikmenin batığa dönüşüm oranı (yarıyıl) ===")
display(DONUSUM.style.format({c: ("{:.2%}" if c.startswith(("HEDEF_ORT", "ORT_")) else "{:.2f}")
                              for c in DONUSUM.columns if c != "N"}, decimal=","))
