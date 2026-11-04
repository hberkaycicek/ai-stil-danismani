def spec_yap(satir, rol):
    return {"rol": rol, "kolonlar": list(satir["KOLONLAR"]), "aile": satir["MODEL"], "params": satir["PARAMS"],
            "bilgi_ay": int(satir["BILGI_AY"]), "girdiler": satir["GIRDILER"], "mevsim": int(satir["MEVSIM"]),
            "ad": f"{rol}: {satir['GIRDILER']}{' + mevsim' if satir['MEVSIM'] else ''} | {satir['MODEL']} | "
                  f"{int(satir['BILGI_AY'])}. ay"}


CHAMP_SATIR = kisa.iloc[0]
ham_set = lambda k: {c for c in k if c in HAM_HAVUZ}             # ortalama/takvim hariç ham girdiler
farkli = lambda df: df[df["KOLONLAR"].apply(ham_set) != ham_set(CHAMP_SATIR["KOLONLAR"])] \
    if CONFIG["CHALLENGER_FARKLI_GIRDI"] else df
# Challenger: <= CHALLENGER_MAX_AY, champion'dan farklı ham girdi. Yedek yollar da anlamlılık kuralına uyar.
ch = farkli(kisa_c)
CHAL_KAYNAK = "1-SE + anlamlılık"
if ch.empty:
    h_ = erken.nsmallest(CONFIG["HAC_ILK_N"], "CV_WMAPE").drop_duplicates(["KOLONLAR", "MODEL"]).copy()
    h_["HAC_MAKS_P"] = h_["KOLONLAR"].apply(lambda k: float(np.max(hac_p(k, cv_df))))
    ch = farkli(h_[h_["HAC_MAKS_P"] <= CONFIG["HAC_P_ESIK"]]).sort_values("CV_WMAPE")
    CHAL_KAYNAK = "anlamlı erken modeller (1-SE dışı)"
if ch.empty:
    ch = kisa_c
    CHAL_KAYNAK = "1-SE listesi (champion ile aynı ham girdiler)"
print(f"Challenger kaynağı: {CHAL_KAYNAK}")
CHAL_SATIR = ch.iloc[0]

CHAMP, CHAL = spec_yap(CHAMP_SATIR, "CHAMPION"), spec_yap(CHAL_SATIR, "CHALLENGER")
SECIMLER = {"CHAMPION": CHAMP, "CHALLENGER": CHAL}
MODELLER = dict(SECIMLER)
for s_ in MODELLER.values():
    print(s_["ad"])
SECIM_TABLO = pd.DataFrame([CHAMP_SATIR, CHAL_SATIR], index=list(MODELLER))
SECIM_TABLO["HAC_MAKS_P"] = SECIM_TABLO["KOLONLAR"].apply(lambda k: float(np.max(hac_p(k, cv_df))))
display(SECIM_TABLO[GOSTER[:-3] + ["HAC_MAKS_P"]].style.format({**FMT, "HAC_MAKS_P": "{:.3f}"}, decimal=","))
