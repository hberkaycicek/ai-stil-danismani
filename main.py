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
    from IPython.display import display as goster
except ImportError:
    goster = print
t0 = datetime.now()

# =========================================================
# 0) PARAMETRELER
# =========================================================
TEST_MODU = True                         # True: mailler sadece SENDER'a gider. Canlıda False
SMTP_HOST = "smtp.akbank.com"
SENDER    = "berkay_cicek@akbank.com"
TO        = ["xxxx@akbank.com"]
CC        = ["xxxx@akbank.com"]

PSI_ESIK, GINI_ESIK = 0.2, 0.2
UYARI_SADECE_SON_AY = False              # False: PSI>=0.2 olan tüm aylar uyarı üretir (orijinal)
MIN_BAD = 20                             # kırılımda kötü sayısı bunun altındaysa Gini '*' (0 = kapalı)

BASLANGIC = pd.Period("2023-11", freq="M")
REF       = pd.period_range("2023-11", "2024-01", freq="M")      # PSI referansı (3 model)

DONEM, STATU, KANAL = "SONUC_DONEM", "ISLEM_SONUCU", "KANAL"
SKORLAR = ["MM_SKOR", "KKB_SKOR", "DAVRANIS_SKOR"]
TAKIP1, MIN1, AD1 = "KK_TAKIP_60_4",  5,  "Gini 4 Ay 60+"
TAKIP2, MIN2, AD2 = "KK_TAKIP_90_12", 13, "Gini 12 Ay 90+"
GINI_TANIM = [(TAKIP1, MIN1, AD1), (TAKIP2, MIN2, AD2)]

_vars    = dataiku.get_custom_variables()
END_DATE = pd.Period(_vars.get("RAPOR_AYI") or pd.Timestamp.now(), freq="M")   # RAPOR_AYI='YYYY-MM' (opsiyonel)
SON_AY   = END_DATE - 1
LABELS   = [str(i) for i in range(1, 11)]
print(f"Rapor ayı: {END_DATE} | son değerlendirilen ay: {SON_AY}")

# =========================================================
# 1) VERİYİ OKU
# =========================================================
KOLONLAR = [DONEM, STATU, KANAL] + SKORLAR + [TAKIP1, TAKIP2]
ds = dataiku.Dataset("LDS_MPS_TRD_HARIC_MERGED")
mevcut = [c["name"] for c in ds.read_schema()]
eksik = [c for c in KOLONLAR if c not in mevcut]
if eksik:
    raise KeyError(f"MERGED'de eksik kolon {eksik} | mevcut: {mevcut}")

m_df = ds.get_dataframe(columns=KOLONLAR)
for c in SKORLAR + [TAKIP1, TAKIP2]:
    m_df[c] = pd.to_numeric(m_df[c], errors="coerce").astype("float32")
m_df[DONEM] = pd.to_datetime(m_df[DONEM].astype(str).str[:6], format="%Y%m", errors="coerce").dt.to_period("M")
for c in [STATU, KANAL]:
    m_df[c] = m_df[c].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan}).astype("category")

m_df = m_df[(m_df[DONEM] >= BASLANGIC) & (m_df[DONEM] < END_DATE)]
print("VERİ:", m_df.shape, "|", m_df[DONEM].min(), "→", m_df[DONEM].max(), "|", datetime.now() - t0)
if (m_df[DONEM] == SON_AY).sum() == 0:
    raise ValueError(f"{SON_AY} dönemine ait kayıt yok! LDS tablosu güncel mi?")

# =========================================================
# 2) MODEL ALT KÜMELERİ + REFERANSLAR
# =========================================================
kkb_df = m_df[m_df["KKB_SKOR"] != 0]
dv_df  = m_df[m_df["DAVRANIS_SKOR"] != 0]
MODELLER = [
    ("MM_SKOR",       m_df,   m_df[m_df[DONEM].isin(REF)]),
    ("KKB_SKOR",      kkb_df, kkb_df[kkb_df[DONEM].isin(REF)]),
    ("DAVRANIS_SKOR", dv_df,  dv_df[dv_df[DONEM].isin(REF)]),
]
for score, df, ref in MODELLER:
    print(f"{score:15s} popülasyon: {len(df):>11,} | referans dolu skor: {ref[score].notna().sum():>9,}")

# =========================================================
# 3) METRİK FONKSİYONLARI
# =========================================================
def gini(y, s):
    if y.nunique() < 2:
        return np.nan
    return 2 * roc_auc_score(y, s) - 1

END_IDX = END_DATE.year * 12 + END_DATE.month
def ay_yasi(p):                                          # dönemden rapor ayına geçen ay
    return END_IDX - (p.dt.year * 12 + p.dt.month)

def psi_by_month(df, ref, score):
    """PSI: TÜM başvurular (onay/red ayrımı yok)."""
    ref_s = ref[score].dropna()
    if ref_s.empty:
        raise ValueError(f"{score}: referans dönemde skor yok!")
    q = ref_s.quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9]).tolist()
    edges  = sorted(set([-999999] + q + [999999]))      # çakışan quantile koruması
    labels = [str(i) for i in range(1, len(edges))]
    if len(labels) < 10:
        print(f"UYARI {score}: tekrar eden quantile → {len(labels)} bin")
    base = pd.cut(ref_s, bins=edges, labels=labels).value_counts(normalize=True)
    base.index = base.index.astype(str)

    p = df.loc[df[score].notna(), [DONEM, score]]
    p = p.assign(BIN=pd.cut(p[score], bins=edges, labels=labels).astype(str))
    comp = p.groupby(DONEM)["BIN"].value_counts(normalize=True).rename("proportion_comp").reset_index()
    comp["proportion"] = comp["BIN"].map(base)
    v = comp[(comp["proportion_comp"] > 0) & (comp["proportion"] > 0)]   # log(0) → inf koruması
    v = v.assign(PSI=(v["proportion_comp"] - v["proportion"]) * np.log(v["proportion_comp"] / v["proportion"]))
    return v.groupby(DONEM)["PSI"].sum(), comp[[DONEM, "BIN", "proportion_comp"]]

def gini_tablo(df, score, group_col=None, min_bad=0):
    """Gini: SADECE onaylı (O) + performansı olgunlaşmış başvurular."""
    keys = [DONEM] + ([group_col] if group_col else [])
    b = df.loc[df[score].notna() & (df[STATU] == "O"), keys + [score, TAKIP1, TAKIP2]]
    if group_col:
        b = b[b[group_col].notna() & (b[group_col].astype(str) != "NA")]
    yas = ay_yasi(b[DONEM])

    parcalar = []
    for takip, min_yas, ad in GINI_TANIM:
        d = b[(yas >= min_yas) & b[takip].notna()]
        if d.empty:
            continue
        g = d.groupby(keys, observed=True).apply(
            lambda x: pd.Series({ad: gini(x[takip].astype(int), x[score]),
                                 f"{ad}_BAD": x[takip].sum()}))
        if min_bad:
            g.loc[g[f"{ad}_BAD"] < min_bad, ad] = np.nan
        parcalar.append(g[[ad]])
    res = pd.concat(parcalar, axis=1).reset_index() if parcalar else pd.DataFrame(columns=keys)
    res["model"] = score
    return res

# =========================================================
# 4) HESAPLAMA
# =========================================================
psi_list, bin_dict, gini_list, kanal_list = [], {}, [], []
for score, df, ref in MODELLER:
    psi, bins = psi_by_month(df, ref, score)
    psi_list.append(psi.rename("PSI").reset_index().assign(model=score))
    bin_dict[score] = bins.rename(columns={DONEM: "DONEM"})
    gini_list.append(gini_tablo(df, score))
    kanal_list.append(gini_tablo(df, score, KANAL, min_bad=MIN_BAD))
    print(score, "tamam", datetime.now() - t0)
del m_df, kkb_df, dv_df, MODELLER

final1 = (pd.concat(gini_list, ignore_index=True)
            .merge(pd.concat(psi_list), on=[DONEM, "model"], how="outer")
            .rename(columns={DONEM: "DONEM"}))
kanal_df = pd.concat(kanal_list, ignore_index=True).rename(columns={DONEM: "DONEM"})
kanal_df[KANAL] = kanal_df[KANAL].astype(str)

# =========================================================
# 5) RAPOR TABLOLARI
# =========================================================
METRIK = [AD1, AD2, "PSI"]

def pivotla(df, cols, metrics):
    metrics = [m for m in metrics if m in df.columns]
    p = df.pivot_table(index="DONEM", columns=cols, values=metrics, aggfunc="first", dropna=False).abs()
    p = p[[m for m in metrics if m in p.columns.get_level_values(0)]]
    return p.sort_index(ascending=False)

def renkli_html(p):
    def hucre(v, metrik):
        if pd.isna(v):
            return "*"
        kirmizi = (metrik == "PSI" and v >= PSI_ESIK) or (metrik.startswith("Gini") and v <= GINI_ESIK)
        return f'<span style="color:red">{v:.3f}</span>' if kirmizi else f"{v:.3f}"
    out = p.astype(object).copy()
    for col in p.columns:
        out[col] = [hucre(v, col[0]) for v in p[col]]
    out.index = out.index.astype(str)
    return out.to_html(escape=False)

def bin_tablo(bins):
    t = bins.pivot_table(index="DONEM", columns="BIN", values="proportion_comp", aggfunc="first")
    t = (t * 100).round(2).sort_index(ascending=False)
    t = t[[c for c in LABELS if c in t.columns]]
    t.index = t.index.astype(str)
    return t.fillna("*")

psi_gini_tbl = pivotla(final1, ["model"], METRIK)
kanal_tbl    = pivotla(kanal_df.rename(columns={KANAL: "-"}), ["-", "model"], [AD1, AD2])
bin_mm_, bin_kkb_, bin_dv_ = (bin_tablo(bin_dict[s]) for s in SKORLAR)

uy = final1[final1["PSI"].abs() >= PSI_ESIK]
if UYARI_SADECE_SON_AY:
    uy = uy[uy["DONEM"] == SON_AY]
uy = uy.sort_values(["DONEM", "model"], ascending=[False, True])
warning_html = "<br>".join(
    f"<span style='color:red;font-weight:bold;'>!Dikkat:</span> {r.DONEM} döneminde {r.model} PSI değeri beklenen aralığın üstünde."
    for r in uy.itertuples())

# Notebook kontrolü
goster(psi_gini_tbl.round(3))
goster(kanal_tbl.round(3))
goster(bin_mm_); goster(bin_kkb_); goster(bin_dv_)
print("Uyarı sayısı:", len(uy))

# =========================================================
# 6) ÇIKTI DATASET (aylık log)
# =========================================================
long = pd.concat([
    final1.assign(KIRILIM_TIPI="GENEL", KIRILIM="-"),
    kanal_df.rename(columns={KANAL: "KIRILIM"}).assign(KIRILIM_TIPI="KANAL"),
], ignore_index=True).rename(columns={"model": "MODEL", AD1: "GINI_4AY_60", AD2: "GINI_12AY_90"})
long["DONEM"]     = long["DONEM"].astype(str)
long["KIRILIM"]   = long["KIRILIM"].astype(str)
long["RAPOR_AYI"] = str(END_DATE)
long["RUN_TS"]    = pd.Timestamp.now()
long = long.reindex(columns=["RAPOR_AYI", "DONEM", "KIRILIM_TIPI", "KIRILIM", "MODEL",
                             "GINI_4AY_60", "GINI_12AY_90", "PSI", "RUN_TS"])
dataiku.Dataset("LDS_MPS_TRD_HARIC_MAIL").write_with_schema(long)
print("Çıktı yazıldı:", long.shape)

# =========================================================
# 7) MAİLLER
# =========================================================
CSS_GENEL = """<style>table{border:1px solid gray;border-collapse:collapse;table-layout:auto;}
th,td{border:1px solid gray;padding:5px;text-align:center;}</style>"""
CSS_KANAL = """<style>table{border:1px solid gray;border-collapse:collapse;table-layout:auto;}
th{border:1px solid gray;padding:2px;text-align:center;word-wrap:break-word;font-size:10px;}
td{border:1px solid gray;padding:2px;text-align:center;word-wrap:break-word;font-size:12px;}</style>"""

ACIKLAMA_GINI = """<p>-<strong>Gini</strong> değerinin <strong>0.4 ile 0.6</strong> arasında olması genellikle iyi,
<strong>0.6</strong> üzeri olması çok iyi, <strong>0.4</strong> altı değerler ise kötü model performansına işaret eder.</p>"""
ACIKLAMA_KANAL = ACIKLAMA_GINI + (f"<p>-Kötü müşteri sayısı <strong>{MIN_BAD}</strong>'nin altında olan kırılımlarda "
                                  f"Gini hesaplanmamış, <strong>*</strong> ile gösterilmiştir.</p>" if MIN_BAD else "")
ACIKLAMA_GENEL = f"""<p>-BIN Dağılımında her bin belirli bir skor aralığını temsil eder, en düşükten en yükseğe doğru sıralanır.
<strong>Bin 1</strong> en düşük skorların bulunduğu aralığı, <strong>Bin 10</strong> en yüksek skorların bulunduğu aralığı içerir.</p>
<p>-<strong>PSI</strong> değeri <strong>&lt;0.1</strong> ise iyi, <strong>0.1-0.2</strong> arasında ise kabul edilebilir,
<strong>0.2</strong>'nin üzeri değerler modelin performansının kötüleştiğini gösterir.</p>{ACIKLAMA_GINI}
<p>-PSI hesaplamasında <strong>{REF[0]} – {REF[-1]}</strong> dönemi baz alınmıştır.</p>"""
ALT = '<div style="font-size:14px;color:black;font-weight:normal;">{}</div>'

def html_govde(css, icerik):
    return f"<html><head>{css}</head><body>{icerik}</body></html>"

def mail_gonder(subject, body):
    to, cc = ([SENDER], []) if TEST_MODU else (TO, CC)
    msg = MIMEMultipart()
    msg["From"], msg["To"], msg["Subject"] = SENDER, ",".join(to), subject
    if cc:
        msg["Cc"] = ",".join(cc)
    msg.attach(MIMEText(body, "html", "utf-8"))
    with smtplib.SMTP(SMTP_HOST) as s:
        s.sendmail(SENDER, to + cc, msg.as_string())
    print("Gönderildi:", subject, "->", to + cc)

mail_gonder(
    "(Otomatik Mail): LDS Model Performans Raporu (TRD HARİÇ)",
    html_govde(CSS_GENEL,
        f"<div>{warning_html}</div>{renkli_html(psi_gini_tbl)}<br>"
        f"<h3>Müşteri Model Skor BIN Dağılımı</h3>{bin_mm_.to_html()}"
        f"<h3>KKB Skor BIN Dağılımı</h3>{bin_kkb_.to_html()}"
        f"<h3>Davranış Skor BIN Dağılımı</h3>{bin_dv_.to_html()}"
        + ALT.format(ACIKLAMA_GENEL)))

mail_gonder(
    "(Otomatik Mail): LDS Kanal Bazlı Model Performans Raporu (TRD HARİÇ)",
    html_govde(CSS_KANAL, renkli_html(kanal_tbl) + ALT.format(ACIKLAMA_KANAL)))

print("Toplam süre:", datetime.now() - t0)
