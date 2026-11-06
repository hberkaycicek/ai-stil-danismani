# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
try:
    from IPython.display import display
except ImportError:
    display = print

GIRDI, CIKTI = "LIMIT_ARTIS_MPS", "LIMIT_ARTIS_MPS_MERGED"   # DLA'sız: "LIMIT_ARTIS_DLASIZ", "LIMIT_ARTIS_DLASIZ_MERGED"
KEY    = ["KDS_BOUT_MBB_NO", "ISLEM_DONEM"]
TAKIPS = ["KK_TAKIP_60_4", "KK_TAKIP_90_12"]

def temiz(d):
    d["KDS_BOUT_MBB_NO"] = d["KDS_BOUT_MBB_NO"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    d["ISLEM_DONEM"]     = d["ISLEM_DONEM"].astype(str).str.replace(r"\D", "", regex=True).str[:6]
    return d

b = temiz(dataiku.Dataset(GIRDI).get_dataframe())
g = temiz(dataiku.Dataset("LIMIT_ARTIS_PERFORMANS").get_dataframe(columns=KEY + TAKIPS))
for c in TAKIPS:
    g[c] = pd.to_numeric(g[c], errors="coerce")
print(GIRDI, b.shape, b["ISLEM_DONEM"].min(), "→", b["ISLEM_DONEM"].max(), "| PERFORMANS:", g.shape)
print("Tekrar eden müşteri-ay:", b.duplicated(KEY).sum())        # SQL'deki TEKIL sayesinde 0 olmalı

dup = g.duplicated(KEY).sum()
if dup:
    print(f"UYARI: performansta {dup:,} tekrar → max alındı")
    g = g.groupby(KEY, as_index=False).max()

# Orijinal: dla.merge(gec_df, on=['KDS_BOUT_MBB_NO','ISLEM_DONEM'], how='left')
m = b.merge(g, on=KEY, how="left")
assert len(m) == len(b), "Join satır sayısını değiştirdi!"
del b, g

o = m[m["ONAY_RET"] == "ONAY"]
display(o.groupby("ISLEM_DONEM").agg(onay=("KDS_BOUT_MBB_NO", "size"),
                                     takip_dolu=("KK_TAKIP_60_4", lambda s: s.notna().mean()),
                                     bad60=("KK_TAKIP_60_4", "mean")).round(4))
del o

dataiku.Dataset(CIKTI).write_with_schema(m)
print(CIKTI, "yazıldı:", m.shape)
