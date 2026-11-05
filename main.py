import io

paths = FEATHER_ARSIV.list_paths_in_partition()
print(paths)                                   # ör. ['/basv_data01_10_204_vs.feather']

feather_path = [p for p in paths if p.endswith(".feather")][0]
with FEATHER_ARSIV.get_download_stream(feather_path) as s:
    tar = pd.read_feather(io.BytesIO(s.read()))

print(tar.shape)





print(tar.dtypes)
display(tar.head())
print("DONEM aralığı:", tar["DONEM"].min(), "→", tar["DONEM"].max())
print(tar["DONEM"].value_counts().sort_index().head(15))




raw_cols = ["BASVURU_NO", "BASVURU_GIRIS_TARIHI", "DONEM", "STATU", "KKB_SKOR", "MM_SKOR",
            "OGRENCI_MODEL_SKOR", "SEGMENTATION_03200", "CALISMA_SEKLI", "KANAL"]

eksik = [c for c in raw_cols if c not in tar.columns]
print("Eksik kolonlar:", eksik)                # boş liste olmalı

tar = tar.reindex(columns=raw_cols).copy()
tar["BASVURU_NO"] = tar["BASVURU_NO"].astype(str).str.strip()
tar["DONEM"] = tar["DONEM"].astype(str).str[:6]
tar["BASVURU_GIRIS_TARIHI"] = pd.to_datetime(tar["BASVURU_GIRIS_TARIHI"])
for c in ["KKB_SKOR", "MM_SKOR", "OGRENCI_MODEL_SKOR"]:
    tar[c] = pd.to_numeric(tar[c], errors="coerce")

MODEL_PERFORMANS_TARIHSEL = dataiku.Dataset("MODEL_PERFORMANS_TARIHSEL")
MODEL_PERFORMANS_TARIHSEL.write_with_schema(tar)
print("Yazıldı:", tar.shape)
