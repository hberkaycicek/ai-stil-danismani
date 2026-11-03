# Seçim kuralı CV'ye göre champion seçer; OOT seçimde kullanılmaz. OOT, CV kazancının gerçek hayata
# taşınıp taşınmadığının testidir. Taşınmıyorsa karar burada, gerekçesiyle, açıkça verilir.
KARAR = {"CHAMPION": "REFERANS_V4", "CHALLENGER": "CHAMPION"}   # yeni rol: Bölüm 12'deki rol
KARAR_GEREKCE = "Geçmiş ortalamalı model CV'de iyi, OOT'de naif modelden kötü -> v4 champion kalır"

if "WF_ILK" not in globals():                                  # hücre tekrar çalışırsa bozulmasın
    WF_ILK, OVERLAY_ILK, PERF_ILK, MODELLER_ILK = dict(WF), dict(OVERLAY), PERF.copy(), dict(MODELLER)
WF = {yeni: WF_ILK[eski] for yeni, eski in KARAR.items()}
OVERLAY = {yeni: OVERLAY_ILK[eski] for yeni, eski in KARAR.items()}
PERF = PERF_ILK[PERF_ILK["MODEL"].isin(KARAR.values())].copy()
PERF["MODEL"] = PERF["MODEL"].map({eski: yeni for yeni, eski in KARAR.items()})
MODELLER = {}
for yeni, eski in KARAR.items():
    s_ = dict(MODELLER_ILK[eski])
    s_["rol"] = yeni
    s_["ad"] = yeni + s_["ad"][s_["ad"].index(":"):]
    MODELLER[yeni] = s_
SECIMLER, CHAMP, CHAL = MODELLER, MODELLER["CHAMPION"], MODELLER["CHALLENGER"]
print("KARAR:", KARAR_GEREKCE)
for s_ in MODELLER.values():
    print(" ", s_["ad"])
display(PERF.style.format({"WMAPE": "{:.2f}", "MAE_yp": "{:.3f}", "RMSE_yp": "{:.3f}", "SAPMA_yp": "{:+.3f}",
                           "R2": "{:.3f}", "KORELASYON": "{:.3f}", "DUMMY_WMAPE": "{:.2f}", "BECERI": "{:.3f}"},
                          decimal=","))
