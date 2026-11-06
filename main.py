# -*- coding: utf-8 -*-
import dataiku
import pandas as pd

ID     = "BASVURU_NUMARASI"
TAKIPS = ["KK_TAKIP_60_4", "KK_TAKIP_90_12"]

def temiz_id(s):
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

# Gecikme tablosu kolon kontrolü
gds = dataiku.Dataset("LDS_MPS_TRD_HARIC_PERFORMANS")
mevcut = [c["name"] for c in gds.read_schema()]
eksik = [c for c in [ID] + TAKIPS if c not in mevcut]
if eksik:
    raise KeyError(f"Gecikmede eksik kolon {eksik} | mevcut: {mevcut}")

b = dataiku.Dataset("LDS_MPS_TRD_HARIC_BIRLESIK").get_dataframe()
g = gds.get_dataframe(columns=[ID] + TAKIPS)
b[ID], g[ID] = temiz_id(b[ID]), temiz_id(g[ID])
b["SONUC_DONEM"] = b["SONUC_DONEM"].astype(str).str[:6]
for c in TAKIPS:
    g[c] = pd.to_numeric(g[c], errors="coerce")
print("BİRLEŞİK:", b.shape, "| GECİKME:", g.shape)

# ID format teşhisi
fmt = lambda s: s.str[0].where(~s.str[0].str.isdigit(), "rakam")
print("BİRLEŞİK ID başlangıcı:", fmt(b[ID]).value_counts().to_dict())
print("GECİKME  ID başlangıcı:", fmt(g[ID]).value_counts().to_dict())

dup = g[ID].duplicated().sum()
if dup:
    print(f"UYARI: gecikmede {dup:,} tekrar eden ID → max alındı")
    g = g.groupby(ID, as_index=False).max()

m = b.merge(g, on=ID, how="left")          # orijinal: all_df.merge(gec_df, on='BASVURU_NUMARASI', how='left')
assert len(m) == len(b), "Join satır sayısını değiştirdi!"
del b, g

o = m[m["ISLEM_SONUCU"] == "O"]
kontrol = o.groupby("SONUC_DONEM").agg(onay=(ID, "size"),
                                       takip60_dolu=("KK_TAKIP_60_4", lambda s: s.notna().mean()),
                                       takip90_dolu=("KK_TAKIP_90_12", lambda s: s.notna().mean()),
                                       bad60=("KK_TAKIP_60_4", "mean"))
display(kontrol.round(4))
del o

dataiku.Dataset("LDS_MPS_TRD_HARIC_MERGED").write_with_schema(m)
print("MERGED:", m.shape)
