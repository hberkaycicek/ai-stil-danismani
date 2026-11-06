# -*- coding: utf-8 -*-
import dataiku, io
import pandas as pd

KOLONLAR = ["BASVURU_NUMARASI", "SONUC_DONEM", "ISLEM_SONUCU",
            "MM_SKOR", "KKB_SKOR", "DAVRANIS_SKOR", "KANAL"]
DOSYA = "lds_old_2301_2505.parquet"

df = dataiku.Dataset("LDS_MPS_TRD_HARIC_TARIHSEL").get_dataframe(columns=KOLONLAR)
df["SONUC_DONEM"] = df["SONUC_DONEM"].astype(str).str.replace(r"\D", "", regex=True).str[:6]
print("Satır:", df.shape, "| boş dönem:", (df["SONUC_DONEM"].str.len() != 6).sum())
print(df["SONUC_DONEM"].value_counts().sort_index().to_string())

buf = io.BytesIO()
df.to_parquet(buf, index=False)
folder = dataiku.Folder("kq7RFR4o")                 # FEATHER_ARSIV
folder.upload_data(DOSYA, buf.getvalue())

# Doğrulama: geri oku, bozuk mu / satır tutuyor mu
with folder.get_download_stream(DOSYA) as s:
    raw = s.read()
assert raw[:4] == b"PAR1" and raw[-4:] == b"PAR1", "Dosya bozuk yazıldı!"
assert len(pd.read_parquet(io.BytesIO(raw), columns=["SONUC_DONEM"])) == len(df), "Satır sayısı tutmuyor!"
print("Yazıldı ve doğrulandı:", DOSYA, round(len(raw) / 1e6, 1), "MB")
