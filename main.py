def veri_hazirla(df_raw, config):
    tur = config["HEDEF_TUR"]
    hedef = oncul_kolon(config["HEDEF_SON"], tur)
    onculler = [oncul_kolon(k, tur) for k in ONCU_KODLARI]
    k90 = sorted([c for c in df_raw.columns if c.startswith("KK_TAKIP_90_") and c.endswith(f"_{tur}")
                  and c.split("_")[3].isdigit() and int(c.split("_")[3]) < 12],
                 key=lambda c: int(c.split("_")[3]))
    ham_ortak = [c for c in ORTAK_ALANLAR if c != "ORTALAMA_LIMIT"]
    ek = [config[k] for k in ["LIMIT_KOLON", "LIMIT_PAYDA_KOLON", "ORTALAMA_AGIRLIK_KOLON",
                              "SAMPLE_WEIGHT_KOLON", "TUFE_KOLON"] if config[k]]
    gerekli = list(dict.fromkeys(["DONEM_1", hedef, *onculler, *ham_ortak, *ek, *k90]))
    eksik = sorted(set(gerekli) - set(df_raw.columns))
    if eksik:
        raise ValueError(f"Eksik kolon: {eksik}")
    if any("_90_" in k for k in onculler + ORTAK_ALANLAR):
        raise ValueError("90+ alanı girdi olamaz.")

    d = df_raw[gerekli].copy()
    d["AY"] = pd.to_datetime(d["DONEM_1"].astype(str), format="%Y%m").dt.to_period("M")
    d = d.sort_values("AY").reset_index(drop=True)
    if d["AY"].duplicated().any():
        raise ValueError("Tekrarlanan kohort var.")
    tam = pd.period_range(d["AY"].min(), d["AY"].max(), freq="M")
    if len(tam) != len(d):
        raise ValueError(f"Eksik ay: {sorted(set(tam.astype(str)) - set(d['AY'].astype(str)))}")
    sayisal = [c for c in gerekli if c != "DONEM_1"]
    d[sayisal] = d[sayisal].apply(pd.to_numeric, errors="raise")

    payda = d[config["LIMIT_PAYDA_KOLON"]].where(d[config["LIMIT_PAYDA_KOLON"]] > 0)
    d["ORTALAMA_LIMIT"] = d[config["LIMIT_KOLON"]] / payda
    if config["TUFE_KOLON"]:
        d["ORTALAMA_LIMIT"] = d["ORTALAMA_LIMIT"] / d[config["TUFE_KOLON"]] * 100

    orijin = pd.Period(config["TREND_ORIJIN"], freq="M")
    d[TREND_KOL] = d["AY"].apply(lambda p: (p - orijin).n).astype(float)
    d["SIN_AY"] = np.sin(2 * np.pi * d["AY"].dt.month / 12)
    d["COS_AY"] = np.cos(2 * np.pi * d["AY"].dt.month / 12)

    bilgi = pd.Period(config["BILGI_AYI"], freq="M")
    gec = config["RAPOR_GECIKME_AY"]
    d["HEDEF_ERISIM"] = d["AY"] + config["HEDEF_OLGUNLASMA_AY"] + gec       # hedefin belli olduğu ay
    gecerli_hedef = d[hedef].notna() & np.isfinite(d[hedef]) & d[hedef].between(0, 1)
    d["HEDEF_KULLANILABILIR"] = (d["HEDEF_ERISIM"] < bilgi) & gecerli_hedef
    d["GOZLENEN_AY"] = d["AY"].apply(lambda p: (bilgi - p).n) - 1 - gec

    kismi = {}
    for k, kol in zip(ONCU_KODLARI, onculler):                               # bugün tamamlanmamış öncül -> NaN
        tam_mi = (d["AY"] + oncul_ay(k) + gec) < bilgi
        if (~tam_mi & d[kol].notna()).any():
            kismi[kisa_ad(kol)] = int((~tam_mi & d[kol].notna()).sum())
        d.loc[~tam_mi, kol] = np.nan
    if kismi:
        print("Kısmi (tamamlanmamış) öncül değeri NaN yapıldı:", kismi)
    if ((d[onculler] < 0) | (d[onculler] > 1)).sum().sum():
        raise ValueError("0-1 dışında oran girdisi var.")
    na = d.loc[d["HEDEF_KULLANILABILIR"], onculler + ORTAK_ALANLAR].isna().sum()
    if na.any():
        print("UYARI - olgun kohortlarda boş girdi:\n", na[na > 0].to_string())
    return d, hedef, onculler, k90


def bolme_uygula(d, config):
    kullan = d[d["HEDEF_KULLANILABILIR"]]
    gel = kullan.index[kullan["DONEM_1"].between(*config["GELISTIRME_ARALIK"])].to_numpy()
    oot = kullan.index[kullan["DONEM_1"].between(*config["OOT_ARALIK"])].to_numpy()
    disarida = kullan.index.difference(np.concatenate([gel, oot]))
    if len(disarida):
        print("UYARI - bölgeye atanmamış olgun kohort:", list(kullan.loc[disarida, "DONEM_1"]))
    if len(gel) == 0 or len(oot) == 0 or len(set(gel) & set(oot)):
        raise ValueError("Geliştirme/OOT bölmesi hatalı.")
    if d.loc[gel, "AY"].max() >= d.loc[oot, "AY"].min():
        raise ValueError("OOT geliştirmeden sonra başlamalı.")
    return gel, oot


d, HEDEF, ONCULLER, K90 = veri_hazirla(df_raw, CONFIG)
gel_idx, oot_idx = bolme_uygula(d, CONFIG)
HK = kisa_ad(HEDEF)
BILGI = pd.Period(CONFIG["BILGI_AYI"], "M")
d["BOLGE"] = np.select([d.index.isin(gel_idx), d.index.isin(oot_idx), ~d["HEDEF_KULLANILABILIR"]],
                       ["GELISTIRME", "OOT", "TAHMIN"], "DISARIDA")
VERI_OZET = (d.groupby("BOLGE", sort=False)
              .agg(N=("AY", "size"), BAS=("AY", "min"), SON=("AY", "max"), HEDEF_ORT=(HEDEF, "mean"))
              .reindex(["GELISTIRME", "OOT", "TAHMIN"]))
print(f"Hedef: {HEDEF} | Bilgi ayı: {BILGI} | 90+ olgunluk kolonları: {[kisa_ad(k) for k in K90] or 'yok'}")
display(VERI_OZET.style.format({"HEDEF_ORT": yuzde}, na_rep="-"))
if not (d["BOLGE"] == "TAHMIN").any():
    print("Bilgi: veride olgunlaşmamış kohort yok -> Bölüm 15-16'da gelecek tahmini üretilmez. "
          "Gelecek tahmini için 202510 sonrası kohortları veriye ekle (hedef boş/kısmi kalabilir).")
