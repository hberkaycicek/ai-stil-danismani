# DÜZELTME: 202509 kesmesi kaldırıldı. Olgunlaşmamış kohortlar (hedefi henüz belli olmayan) veride
# kalır; model kurulmaz, sadece üretimde tahmin edilir. Hangi kohortun hedefi belli, Bölüm 3 karar verir.
import time
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import statsmodels.api as sm
from IPython.display import display
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, enet_path)
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 250)

DATASET_ADI = "BDS_TAHMIN_ALL"
if "kaynak" not in globals():
    import dataiku
    kaynak = dataiku.Dataset(DATASET_ADI).get_dataframe()
if kaynak.empty or kaynak.columns.duplicated().any():
    raise ValueError("Kaynak boş veya kolon adları tekrarlanıyor.")

donem = pd.to_numeric(kaynak["DONEM_1"], errors="coerce")
if donem.isna().any() or (donem % 1 != 0).any():
    raise ValueError("DONEM_1 geçerli YYYYMM tam sayıları olmalı.")
df_raw = kaynak.loc[donem >= 202101].copy()
df_raw["DONEM_1"] = donem.loc[df_raw.index].astype(int)

RENK = {"GERCEK": "#111827", "EGITIM": "#94a3b8", "VALIDASYON": "#2563eb", "OOT": "#ea580c",
        "TAHMIN": "#16a34a", "KISMI": "#6b7280", "CHALLENGER": "#7c3aed", "REFERANS": "#0891b2",
        "DUMMY": "#cbd5e1"}
ARKA = {"EGITIM": "#f1f5f9", "VALIDASYON": "#eff6ff", "OOT": "#fff7ed", "TAHMIN": "#f0fdf4"}
yuzde = lambda v, d=2: "-" if pd.isna(v) else f"%{v * 100:.{d}f}".replace(".", ",")
sayi = lambda v, d=2: "-" if pd.isna(v) else f"{v:.{d}f}".replace(".", ",")
PCT_AX = FuncFormatter(lambda x, _: f"%{x:.2f}".replace(".", ","))
print(f"Kaynak: {DATASET_ADI} | Satır: {len(df_raw)} | Kolon: {df_raw.shape[1]} | "
      f"Dönem: {df_raw['DONEM_1'].min()} – {df_raw['DONEM_1'].max()}")
