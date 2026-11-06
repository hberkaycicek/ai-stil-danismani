# -*- coding: utf-8 -*-
import dataiku, os
import pandas as pd, numpy as np
import pyarrow as pa, pyarrow.feather as pf

ID, DONEM    = "BASVURU_NUMARASI", "SONUC_DONEM"
KOLONLAR     = [ID, DONEM, "ISLEM_SONUCU", "MM_SKOR", "KKB_SKOR", "DAVRANIS_SKOR", "KANAL"]
FOLDER_ID    = "kq7RFR4o"          # FEATHER_ARSIV
FEATHER_ONEK = "lds_old"

def temiz_id(s):
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

def donem_str(s):                  # her formatı 'YYYYMM' string'e çevirir
    if pd.api.types.is_datetime64_any_dtype(s):
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
        return s.dt.strftime("%Y%m")
    return s.astype(str).str.replace(r"\D", "", regex=True).str[:6]

def standartla(d, kaynak):
    d = d[KOLONLAR].copy()
    d[ID]    = temiz_id(d[ID])
    d[DONEM] = donem_str(d[DONEM])
    for c in ["MM_SKOR", "KKB_SKOR", "DAVRANIS_SKOR"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ["ISLEM_SONUCU", "KANAL"]:
        d[c] = (d[c].astype(object).where(d[c].notna(), None)
                   .map(lambda x: x.replace("\xa0", "").strip() if isinstance(x, str) else x))
    d["KAYNAK"] = kaynak
    return d
