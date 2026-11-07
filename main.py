# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from sklearn.metrics import roc_auc_score
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
try:
    from IPython.display import display
except ImportError:
    display = print
t0 = datetime.now()

# =========================================================
# 0) PARAMETRELER
# =========================================================
GIRDI, CIKTI = "YENI_KART_SATIS_MPS", "YENI_KART_SATIS_MPS_MAIL"
SUBJECT = "(Otomatik Mail): Ön Onay Yeni Kart Satış Model Performans Raporu"

TEST_MODU = True                         # True: mail sadece SENDER'a gider
SMTP_HOST = "smtp.xxxx.com"
SENDER    = "berkay_cicek@xxx.com"
TO        = ["xxxx@xxx.com"]
CC        = ["xxxx@xxxxx.com"]

PSI_ESIK, GINI_ESIK = 0.2, 0.2
INF_KORUMA = True                        # False: boş bin'de orijinaldeki gibi inf
CHUNK      = 10_000_000
BAS_INT    = 202301
REF_BAS, REF_BIT = 202301, 202303        # PSI referansı (2023 ilk 3 ay)

SKORLAR = ["MM_SKOR", "KKB_SKOR"]
BASLIK  = {"MM_SKOR": "Müşteri Model Skor BIN Dağılımı", "KKB_SKOR": "KKB Skor BIN Dağılımı"}
TAKIP1, MIN1, AD1 = "KK_TAKIP_60_4",  5,  "Gini 4 Ay 60+"
TAKIP2, MIN2, AD2 = "KK_TAKIP_90_12", 13, "Gini 12 Ay 90+"
LABELS = [str(i) for i in range(1, 11)]

_vars    = dataiku.get_custom_variables()
END_DATE = pd.Period(_vars.get("RAPOR_AYI") or pd.Timestamp.now(), freq="M")
END_INT  = END_DATE.year * 100 + END_DATE.month
SON_AY   = (END_DATE - 1).year * 100 + (END_DATE - 1).month
END_IDX  = END_DATE.year * 12 + END_DATE.month
BAS_IDX  = 2023 * 12 + 1
NAY      = END_IDX - BAS_IDX
DONEMLER = np.array([((BAS_IDX + i - 1) // 12) * 100 + (BAS_IDX + i - 1) % 12 + 1 for i in range(NAY)])
fmt = lambda d: f"{int(d)//100}-{int(d)%100:02d}"
print(f"Yeni Kart | rapor ayı {END_DATE} | son ay {fmt(SON_AY)}")

def ay_idx(d):
    return (d // 100) * 12 + d % 100

def model_maske(df, s):                            # KKB için 0 = kayıt yok → dışla
    m = df[s].notna()
    return (m & (df[s] != 0)) if s != "MM_SKOR" else m

def oku(kolonlar):
    for ch in dataiku.Dataset(GIRDI).iter_dataframes(chunksize=CHUNK, columns=kolonlar):
        ch = ch.apply(pd.to_numeric, errors="coerce")
        ch = ch[(ch["ISLEM_DONEM"] >= BAS_INT) & (ch["ISLEM_DONEM"] < END_INT)]
        ch = ch.astype({c: "float32" for c in kolonlar if c != "ISLEM_DONEM"})
        ch["ISLEM_DONEM"] = ch["ISLEM_DONEM"].astype("int32")
        yield ch

# =========================================================
# 1) 1. GEÇİŞ: referans + Gini verisi
# =========================================================
ref_p, gini_p, n, son_ay_var = [], [], 0, False
for ch in oku(["ISLEM_DONEM", "ONAY"] + SKORLAR + [TAKIP1, TAKIP2]):
    n += len(ch)
    son_ay_var |= bool((ch["ISLEM_DONEM"] == SON_AY).any())
    ref_p.append(ch.loc[ch["ISLEM_DONEM"].between(REF_BAS, REF_BIT), SKORLAR])
    yas = END_IDX - ay_idx(ch["ISLEM_DONEM"])
    gini_p.append(ch.loc[(ch["ONAY"] == 1) & (yas >= MIN1), ["ISLEM_DONEM"] + SKORLAR + [TAKIP1, TAKIP2]])
    print(f"1. geçiş: {n:,} satır | {datetime.now() - t0}")

if not son_ay_var:
    raise ValueError(f"{fmt(SON_AY)} dönemine ait kayıt yok!")
ref_df  = pd.concat(ref_p,  ignore_index=True); del ref_p
gini_df = pd.concat(gini_p, ignore_index=True); del gini_p
print("Referans:", ref_df.shape, "| Gini (olgun onaylı):", gini_df.shape)

EDGES, BASE = {}, {}
for s in SKORLAR:
    r = ref_df.loc[model_maske(ref_df, s), s]
    if r.empty:
        raise ValueError(f"{s}: referans dönemde skor yok!")
    q = r.quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9]).tolist()
    e = np.array(sorted(set([-999999] + q + [999999])), dtype="float64")
    k = len(e)
    b = np.searchsorted(e, r.values.astype("float64"), side="left")      # pd.cut ile aynı: (a, b]
    cnt = np.bincount(b, minlength=k + 1)[1:k]
    EDGES[s], BASE[s] = e, cnt / cnt.sum()
    if k < 11:
        print(f"UYARI {s}: tekrar eden quantile → {k - 1} bin")
del ref_df

# =========================================================
# 2) 2. GEÇİŞ: aylık bin sayaçları (PSI + BIN tabloları)
# =========================================================
SAYAC = {s: np.zeros((NAY, len(EDGES[s]) - 1)) for s in SKORLAR}
n = 0
for ch in oku(["ISLEM_DONEM"] + SKORLAR):
    n += len(ch)
    mi = ay_idx(ch["ISLEM_DONEM"].values) - BAS_IDX
    for s in SKORLAR:
        m = model_maske(ch, s).values
        e = EDGES[s]; k = len(e)
        b = np.searchsorted(e, ch[s].values[m].astype("float64"), side="left")
        ok = (b >= 1) & (b <= k - 1)
        key = mi[m][ok] * (k - 1) + (b[ok] - 1)
        SAYAC[s] += np.bincount(key, minlength=NAY * (k - 1)).reshape(NAY, k - 1)
    print(f"2. geçiş: {n:,} satır | {datetime.now() - t0}")

# =========================================================
# 3) METRİKLER
# =========================================================
def gini(y, s):
    return np.nan if y.nunique() < 2 else 2 * roc_auc_score(y, s) - 1

def gini_tablo(df, s):
    b = df[model_maske(df, s)]
    yas = END_IDX - ay_idx(b["ISLEM_DONEM"])
    out = {}
    for takip, min_yas, ad in [(TAKIP1, MIN1, AD1), (TAKIP2, MIN2, AD2)]:
        d = b[(yas >= min_yas) & b[takip].notna()]
        out[ad] = d.groupby("ISLEM_DONEM").apply(lambda x: gini(x[takip].astype(int), x[s]))
    return pd.DataFrame(out)

parcalar, bt = [], {}
for s in SKORLAR:
    c = SAYAC[s]; tot = c.sum(axis=1); var = tot > 0
    k = len(EDGES[s])
    ct = pd.DataFrame(c[var] / tot[var, None], index=DONEMLER[var], columns=[str(i) for i in range(1, k)])
    comp, base = ct.values, BASE[s]
    with np.errstate(divide="ignore", invalid="ignore"):
        terim = (comp - base) * np.log(comp / base)
    if INF_KORUMA:
        terim = np.where((comp > 0) & (base > 0), terim, 0)
    psi = pd.Series(terim.sum(axis=1), index=ct.index, name="PSI")

    f = gini_tablo(gini_df, s).join(psi, how="outer")
    f.index.name = "ISLEM_DONEM"
    parcalar.append(f.reset_index().assign(model=s))

    t = (ct * 100).round(2).sort_index(ascending=False)
    t.index = [fmt(i) for i in t.index]
    bt[s] = t[[x for x in LABELS if x in t.columns]].fillna("*")
    print(s, "tamam", datetime.now() - t0)
del gini_df

final = pd.concat(parcalar, ignore_index=True)
final["DONEM"] = final["ISLEM_DONEM"].map(fmt)

# =========================================================
# 4) TABLO + UYARI
# =========================================================
p = final.pivot_table(index="DONEM", columns="model", values=[AD1, AD2, "PSI"], aggfunc="first").abs()
p = p[[x for x in [AD1, AD2, "PSI"] if x in p.columns.get_level_values(0)]].sort_index(ascending=False)

def renkli_html(t):
    def hucre(v, metrik):
        if pd.isna(v):
            return "*"
        kirmizi = (metrik == "PSI" and v >= PSI_ESIK) or (metrik.startswith("Gini") and v <= GINI_ESIK)
        return f'<span style="color:red">{v:.3f}</span>' if kirmizi else f"{v:.3f}"
    out = t.astype(object).copy()
    for col in t.columns:
        out[col] = [hucre(v, col[0]) for v in t[col]]
    return out.to_html(escape=False)

uy = final[(final["ISLEM_DONEM"] == SON_AY) & (final["PSI"].abs() >= PSI_ESIK)]
warning_html = "<br>".join(
    f"<span style='color:red;font-weight:bold;'>!Dikkat:</span> {r.DONEM} döneminde {r.model} PSI değeri beklenen aralığın üstünde."
    for r in uy.itertuples())

display(p.round(3))
for s in SKORLAR:
    display(bt[s])
print("Uyarı sayısı:", len(uy))

# =========================================================
# 5) LOG DATASET
# =========================================================
log = final.rename(columns={"model": "MODEL", AD1: "GINI_4AY_60", AD2: "GINI_12AY_90"})
log = log.assign(POPULASYON="YENI_KART", RAPOR_AYI=str(END_DATE), RUN_TS=pd.Timestamp.now())
log = log.reindex(columns=["RAPOR_AYI", "POPULASYON", "DONEM", "MODEL", "GINI_4AY_60",
                           "GINI_12AY_90", "PSI", "RUN_TS"]).replace([np.inf, -np.inf], np.nan)
dataiku.Dataset(CIKTI).write_with_schema(log)
print("Log yazıldı:", log.shape)

# =========================================================
# 6) MAİL
# =========================================================
CSS = """<style>table{border:1px solid black;border-collapse:collapse;table-layout:auto;}
th,td{border:1px solid black;padding:5px;text-align:center;}</style>"""
ACIKLAMA = """<div style="font-size:14px;color:black;font-weight:normal;">
<p>-BIN Dağılımında her bin belirli bir skor aralığını temsil eder, en düşükten en yükseğe doğru sıralanır.
<strong>Bin 1</strong> en düşük skorların bulunduğu aralığı, <strong>Bin 10</strong> en yüksek skorların bulunduğu aralığı içerir.</p>
<p>-<strong>PSI</strong> değeri <strong>&lt;0.1</strong> ise iyi, <strong>0.1- 0.2</strong> arasında ise kabul edilebilir,
<strong>0.2</strong>'nin üzeri değerler modelin performansının kötüleştiğini gösterir.</p>
<p>-<strong>Gini</strong> değerinin <strong>0.4 ile 0.6</strong> arasında olması genellikle iyi, <strong>0.6</strong> üzeri olması çok iyi,
<strong>0.4</strong> altı değerler ise kötü model performansına işaret eder.</p>
<p>-PSI hesaplamasında <strong>2023</strong> yılının ilk <strong>3</strong> ayı baz alınmıştır.</p></div>"""

bin_html = "".join(f"<h3>{BASLIK[s]}</h3>{bt[s].to_html()}" for s in SKORLAR)
body = (f"<html><head>{CSS}</head><body><div>{warning_html}</div>{renkli_html(p)}<br>"
        f"{bin_html}{ACIKLAMA}</body></html>")

to, cc = ([SENDER], []) if TEST_MODU else (TO, CC)
msg = MIMEMultipart()
msg["From"], msg["To"], msg["Subject"] = SENDER, ",".join(to), SUBJECT
if cc:
    msg["Cc"] = ",".join(cc)
msg.attach(MIMEText(body, "html", "utf-8"))
with smtplib.SMTP(SMTP_HOST) as s:
    s.sendmail(SENDER, to + cc, msg.as_string())
print("Gönderildi:", SUBJECT, "->", to + cc, "| toplam süre:", datetime.now() - t0)
