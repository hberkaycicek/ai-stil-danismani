def spec_yap(satir, rol):
    return {"rol": rol, "kolonlar": list(satir["KOLONLAR"]), "aile": satir["MODEL"], "params": satir["PARAMS"],
            "bilgi_ay": int(satir["BILGI_AY"]), "girdiler": satir["GIRDILER"], "mevsim": int(satir["MEVSIM"]),
            "ad": f"{rol}: {satir['GIRDILER']}{' + mevsim' if satir['MEVSIM'] else ''} | {satir['MODEL']} | "
                  f"{int(satir['BILGI_AY'])}. ay"}


CHAMP_SATIR = kisa.iloc[0]
farkli = lambda df: df[df["KOLONLAR"].apply(lambda k: set(taban(k))) != set(taban(CHAMP_SATIR["KOLONLAR"]))] \
    if CONFIG["CHALLENGER_FARKLI_GIRDI"] else df
ch = farkli(kisa_c)
if ch.empty:
    print("UYARI: challenger 1-SE listesinde farklı girdili model yok -> geçerli erken modellerden en iyi CV.")
    ch = farkli(erken).sort_values("CV_WMAPE")
CHAL_SATIR = ch.iloc[0]

ref_set = {oncul_kolon("30_2"), KKB_KOL}
ref = grid[(grid["KOLONLAR"].apply(set) == ref_set) & (grid["MODEL"] == CONFIG["REFERANS"]["aile"])]
if ref.empty:
    s = kayit_ekle(cv_degerlendir(sorted(ref_set), model_fabrika("Linear")), sorted(ref_set), "Linear", None)
    ref = rank_ekle(pd.concat([grid_ham, pd.DataFrame([s])], ignore_index=True))
    ref = ref[(ref["KOLONLAR"].apply(set) == ref_set) & (ref["MODEL"] == "Linear")]
REF_SATIR = ref.iloc[0]

CHAMP, CHAL, REF = spec_yap(CHAMP_SATIR, "CHAMPION"), spec_yap(CHAL_SATIR, "CHALLENGER"), spec_yap(REF_SATIR, "REFERANS_V4")
SECIMLER = {"CHAMPION": CHAMP, "CHALLENGER": CHAL}
MODELLER = {**SECIMLER, "REFERANS_V4": REF}
for s_ in MODELLER.values():
    print(s_["ad"])
SECIM_TABLO = pd.DataFrame([CHAMP_SATIR, CHAL_SATIR, REF_SATIR], index=list(MODELLER))
SECIM_TABLO["HAC_MAKS_P"] = SECIM_TABLO["KOLONLAR"].apply(lambda k: float(np.max(hac_p(k, cv_df))))
display(SECIM_TABLO[GOSTER[:-3] + ["HAC_MAKS_P"]].style.format({**FMT, "HAC_MAKS_P": "{:.3f}"}, decimal=","))
