ds = dataiku.Dataset("LDS_MPS_TRD_HARIC")
mevcut = [c["name"] for c in ds.read_schema()]
eksik = [c for c in KOLONLAR if c not in mevcut]
if eksik:
    raise KeyError(f"LDS'de eksik kolon {eksik} | mevcut: {mevcut}")

lds = standartla(ds.get_dataframe(columns=KOLONLAR), "LDS")
print("LDS:", lds.shape, lds[DONEM].min(), "→", lds[DONEM].max())
