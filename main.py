# -*- coding: utf-8 -*-
import dataiku, io
import pandas as pd

ID, DONEM = "BASVURU_NUMARASI", "SONUC_DONEM"
KOLONLAR  = [ID, DONEM, "ISLEM_SONUCU", "MM_SKOR", "KKB_SKOR", "DAVRANIS_SKOR", "KANAL"]
FOLDER_ID, DOSYA = "kq7RFR4o", "lds_old_2301_2505.parquet"
TAR_BAS, TAR_BIT = "202311", "202505"

def standartla(d, kaynak):
    d = d[KOLONLAR].copy()
    d[ID]    = d[ID].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    d[DONEM] = d[DONEM].astype(str).str.replace(r"\D", "", regex=True).str[:6]
    for c in ["MM_SKOR", "KKB_SKOR", "DAVRANIS_SKOR"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ["ISLEM_SONUCU", "KANAL"]:
        d[c] = d[c].astype(object).map(lambda x: x.replace("\xa0", "").strip() if isinstance(x, str) else x)
    d["KAYNAK"] = kaynak
    return d

def ozet(d, ad):
    print(f"{ad}: {d.shape} | {d[DONEM].min()} → {d[DONEM].max()} | "
          f"boş dönem: {(d[DONEM].str.len() != 6).sum():,} | tekrar eden ID: {d[ID].duplicated().sum():,}")

# 1) Güncel LDS
lds = standartla(dataiku.Dataset("LDS_MPS_TRD_HARIC").get_dataframe(columns=KOLONLAR), "LDS")
ozet(lds, "LDS")

# 2) Tarihsel parquet
with dataiku.Folder(FOLDER_ID).get_download_stream(DOSYA) as s:
    raw = s.read()
assert raw[:4] == b"PAR1" and raw[-4:] == b"PAR1", "Arşiv dosyası bozuk!"
tar = standartla(pd.read_parquet(io.BytesIO(raw), columns=KOLONLAR), "TARIHSEL")
del raw
tar = tar[tar[DONEM].between(TAR_BAS, TAR_BIT)]
ozet(tar, "TARİHSEL")

# 3) Alt alta ekle (LDS üstte → çakışmada güncel kayıt kalır)
birlesik = pd.concat([lds, tar], ignore_index=True)
del lds, tar
birlesik = birlesik[birlesik[DONEM].str.len() == 6]
tekrar = birlesik[ID].duplicated(keep="first").sum()
print(f"Tekrar eden başvuru: {tekrar:,} → tekilleştirildi")
birlesik = birlesik.drop_duplicates(ID, keep="first")

display(birlesik.groupby(DONEM).agg(basvuru=(ID, "size"),
                                    onay=("ISLEM_SONUCU", lambda s: (s == "O").sum())))
dataiku.Dataset("LDS_MPS_TRD_HARIC_BIRLESIK").write_with_schema(birlesik)
print("Yazıldı:", birlesik.shape)
