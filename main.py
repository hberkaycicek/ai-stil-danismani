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
    from IPython.display import display as goster      # notebook'ta tablo gösterir
except ImportError:
    goster = print                                       # recipe run'da print'e düşer

t0 = datetime.now()

# =========================================================
# 0) PARAMETRELER

GINI_ESIK = 0.2

BASLANGIC  = pd.Period("2024-10", freq="M")                    # verinin başladığı ay
REF_MM_KKB = pd.period_range("2024-10", "2024-12", freq="M")   # MM & KKB PSI referansı
REF_OGR    = pd.period_range("2024-10", "2024-12", freq="M")   # Öğrenci PSI referansı

_vars    = dataiku.get_custom_variables()
END_DATE = pd.Period(_vars.get("RAPOR_AYI") or pd.Timestamp.now(), freq="M")   # RAPOR_AYI='YYYY-MM' (opsiyonel)
SON_AY   = END_DATE - 1
LABELS   = [str(i) for i in range(1, 11)]
print(f"Rapor ayı: {END_DATE} | son değerlendirilen ay: {SON_AY}")

# =========================================================
# 1) VERİYİ OKU
# =========================================================
raw_cols = ["BASVURU_NO", "DONEM", "STATU", "KKB_SKOR", "MM_SKOR", "OGRENCI_MODEL_SKOR",
            "SEGMENTATION_03200", "CALISMA_SEKLI", "KANAL"]
gec_cols = ["BASVURU_NO", "KK_TAKIP_30_3", "KK_TAKIP_90_12"]   # DONEM/STATU alınmıyor → _x/_y çakışması yok

raw_df = dataiku.Dataset("MODEL_PERFORMANS_RAW").get_dataframe(columns=raw_cols)
gec_df = dataiku.Dataset("MODEL_PERFORMANS_GECIKME_DATA").get_dataframe(columns=gec_cols)
print("RAW:", raw_df.shape, "| GECIKME:", gec_df.shape)

def temiz_id(s):
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

raw_df["BASVURU_NO"] = temiz_id(raw_df["BASVURU_NO"])
gec_df["BASVURU_NO"] = temiz_id(gec_df["BASVURU_NO"])
for c in ["KK_TAKIP_30_3", "KK_TAKIP_90_12"]:
    gec_df[c] = pd.to_numeric(gec_df[c], errors="coerce")

dup = gec_df["BASVURU_NO"].duplicated().sum()
if dup:
    print(f"UYARI: GECIKME'de {dup} tekrar eden BASVURU_NO → başvuru bazında max alındı")
    gec_df = gec_df.groupby("BASVURU_NO", as_index=False).max()

# =========================================================
# 2) BİRLEŞTİR
# =========================================================
m_df = raw_df.merge(gec_df, on="BASVURU_NO", how="left")
assert len(m_df) == len(raw_df), "Merge satır sayısını değiştirdi!"
del raw_df, gec_df
print("Gecikme eşleşme oranı:", round(m_df["KK_TAKIP_30_3"].notna().mean(), 3))

# =========================================================
# 3) TİP DÖNÜŞÜMLERİ / TEMİZLİK
# =========================================================
for c in ["KKB_SKOR", "MM_SKOR", "OGRENCI_MODEL_SKOR"]:
    m_df[c] = pd.to_numeric(m_df[c], errors="coerce")

d = m_df["DONEM"]
if pd.api.types.is_datetime64_any_dtype(d):
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
    m_df["DONEM"] = d.dt.to_period("M")
else:
    m_df["DONEM"] = pd.to_datetime(d.astype(str).str.replace(r"\D", "", regex=True).str[:6],
                                   format="%Y%m", errors="coerce").dt.to_period("M")
print("DONEM boş:", m_df["DONEM"].isna().sum(), "| aralık:", m_df["DONEM"].min(), "→", m_df["DONEM"].max())

for c in ["STATU", "CALISMA_SEKLI", "KANAL", "SEGMENTATION_03200"]:
    m_df[c] = m_df[c].astype("string").str.replace("\xa0", "", regex=False).str.strip().replace("", pd.NA)

if (m_df["DONEM"] == SON_AY).sum() == 0:
    raise ValueError(f"{SON_AY} dönemine ait başvuru yok! RAW güncel mi kontrol et.")

# =========================================================
# 4) MODEL ALT KÜMELERİ + REFERANSLAR
# =========================================================
kkb_df = m_df[m_df["KKB_SKOR"] != 0]
oms_df = m_df[m_df["CALISMA_SEKLI"] == "OGRENCI"]   # veri 2024-10'dan başladığı için 11.05.2024 filtresi gereksiz

MODELLER = [
    ("MM_SKOR",            m_df,   m_df[m_df["DONEM"].isin(REF_MM_KKB)]),
    ("KKB_SKOR",           kkb_df, kkb_df[kkb_df["DONEM"].isin(REF_MM_KKB)]),
    ("OGRENCI_MODEL_SKOR", oms_df, oms_df[oms_df["DONEM"].isin(REF_OGR)]),
]
for score, df, ref in MODELLER:
    print(f"{score:20s} popülasyon: {len(df):>10,} | referans dolu skor: {ref[score].notna().sum():>8,}")

# =========================================================
# 5) METRİK FONKSİYONLARI
# =========================================================
def gini(y, s):
    if y.nunique() < 2:
        return np.nan
    return 2 * roc_auc_score(y, s) - 1

END_IDX = END_DATE.year * 12 + END_DATE.month
def ay_yasi(p):                        # dönemden rapor ayına kadar geçen ay
    return END_IDX - (p.dt.year * 12 + p.dt.month)

def psi_by_month(df, ref, score):
    ref_s = ref[score].dropna()
    if ref_s.empty:
        raise ValueError(f"{score}: referans dönemde skor yok!")
    q = ref_s.quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9]).tolist()
    edges  = sorted(set([-999999] + q + [999999]))       # çakışan quantile'lara karşı
    labels = [str(i) for i in range(1, len(edges))]
    if len(labels) < 10:
        print(f"UYARI {score}: tekrar eden quantile → {len(labels)} bin")

    base = pd.cut(ref_s, bins=edges, labels=labels).value_counts(normalize=True)
    base.index = base.index.astype(str)

    p = df[(df["DONEM"] >= BASLANGIC) & (df["DONEM"] < END_DATE) & df[score].notna()]
    p = p.assign(BIN=pd.cut(p[score], bins=edges, labels=labels).astype(str))
    comp = p.groupby("DONEM")["BIN"].value_counts(normalize=True).rename("proportion_comp").reset_index()
    comp["proportion"] = comp["BIN"].map(base)

    v = comp[(comp["proportion_comp"] > 0) & (comp["proportion"] > 0)]     # log(0) → inf engeli
    v = v.assign(PSI=(v["proportion_comp"] - v["proportion"]) * np.log(v["proportion_comp"] / v["proportion"]))
    return v.groupby("DONEM")["PSI"].sum(), comp[["DONEM", "BIN", "proportion_comp"]]

def gini_tablo(df, score, group_col=None):
    keys = ["DONEM"] + ([group_col] if group_col else [])
    b = df[(df["DONEM"] >= BASLANGIC) & (df["DONEM"] < END_DATE) & df[score].notna() & (df["STATU"] == "O")]
    if group_col:
        b = b[b[group_col].notna() & (b[group_col] != "NA")]
    yas = ay_yasi(b["DONEM"])

    parcalar = []
    for takip, min_yas, ad in [("KK_TAKIP_30_3", 4, "Gini 3 Ay 30+"), ("KK_TAKIP_90_12", 13, "Gini 12 Ay 90+")]:
        d = b[(yas >= min_yas) & b[takip].notna()]
        if d.empty:
            continue
        parcalar.append(d.groupby(keys).apply(lambda x: gini(x[takip].astype(int), x[score])).rename(ad))
    res = pd.concat(parcalar, axis=1).reset_index() if parcalar else pd.DataFrame(columns=keys)
    res["model"] = score
    return res

# =========================================================
# 6) HESAPLAMA
# =========================================================
psi_list, bin_dict, gini_list, seg_list, kanal_list = [], {}, [], [], []
for score, df, ref in MODELLER:
    psi, bins = psi_by_month(df, ref, score)
    psi_list.append(psi.rename("PSI").reset_index().assign(model=score))
    bin_dict[score] = bins
    gini_list.append(gini_tablo(df, score))
    seg_list.append(gini_tablo(df, score, "SEGMENTATION_03200"))
    kanal_list.append(gini_tablo(df, score, "KANAL"))
    print(score, "tamam", datetime.now() - t0)

final1   = pd.concat(gini_list, ignore_index=True).merge(pd.concat(psi_list), on=["DONEM", "model"], how="outer")
seg_df   = pd.concat(seg_list,   ignore_index=True)
kanal_df = pd.concat(kanal_list, ignore_index=True)

# =========================================================
# 7) RAPOR TABLOLARI
# =========================================================
METRIK = ["Gini 3 Ay 30+", "Gini 12 Ay 90+", "PSI"]

def pivotla(df, cols, metrics):
    metrics = [m for m in metrics if m in df.columns]
    p = df.pivot_table(index="DONEM", columns=cols, values=metrics, aggfunc="first").abs()
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
seg_tbl      = pivotla(seg_df.rename(columns={"SEGMENTATION_03200": "-"}), ["-", "model"], METRIK[:2])
kanal_tbl    = pivotla(kanal_df.rename(columns={"KANAL": "-"}),           ["-", "model"], METRIK[:2])
bin_mm_      = bin_tablo(bin_dict["MM_SKOR"])
bin_kkb_     = bin_tablo(bin_dict["KKB_SKOR"])

uyari = final1[(final1["DONEM"] == SON_AY) & (final1["PSI"].abs() >= PSI_ESIK)]
warning_html = "<br>".join(
    f"<span style='color:red;font-weight:bold;'>!Dikkat:</span> {r.DONEM} döneminde {r.model} PSI değeri beklenen aralığın üstünde."
    for r in uyari.itertuples())

# Notebook kontrolü
goster(psi_gini_tbl.round(3))
goster(seg_tbl.round(3))
goster(kanal_tbl.round(3))
goster(bin_mm_)
goster(bin_kkb_)
print("Uyarılar:", len(uyari))

# =========================================================
# 8) ÇIKTI DATASET (aylık log)
# =========================================================
long = pd.concat([
    final1.assign(KIRILIM_TIPI="GENEL", KIRILIM="-"),
    seg_df.rename(columns={"SEGMENTATION_03200": "KIRILIM"}).assign(KIRILIM_TIPI="SEGMENT"),
    kanal_df.rename(columns={"KANAL": "KIRILIM"}).assign(KIRILIM_TIPI="KANAL"),
], ignore_index=True).rename(columns={"model": "MODEL", "Gini 3 Ay 30+": "GINI_3AY_30",
                                      "Gini 12 Ay 90+": "GINI_12AY_90"})
long["DONEM"]     = long["DONEM"].astype(str)
long["KIRILIM"]   = long["KIRILIM"].astype(str)
long["RAPOR_AYI"] = str(END_DATE)
long["RUN_TS"]    = pd.Timestamp.now()
long = long.reindex(columns=["RAPOR_AYI", "DONEM", "KIRILIM_TIPI", "KIRILIM", "MODEL",
                             "GINI_3AY_30", "GINI_12AY_90", "PSI", "RUN_TS"])
dataiku.Dataset("MODEL_PERFORMANS_MAIL").write_with_schema(long)
print("Çıktı yazıldı:", long.shape)

# =========================================================
# 9) MAİLLER
# =========================================================
CSS_GENEL = """<style>table{border:1px solid gray;border-collapse:collapse;table-layout:auto;}
th,td{border:1px solid gray;padding:3px;text-align:center;font-size:13px;}</style>"""
CSS_KIRILIM = """<style>table{border:1px solid gray;border-collapse:collapse;table-layout:auto;}
th{border:1px solid gray;padding:2px;text-align:center;word-wrap:break-word;font-size:10px;}
td{border:1px solid gray;padding:2px;text-align:center;word-wrap:break-word;font-size:12px;}</style>"""

ACIKLAMA_GINI = """<p>-<strong>Gini</strong> değerinin <strong>0.4 ile 0.6</strong> arasında olması genellikle iyi,
<strong>0.6</strong> üzeri olması çok iyi, <strong>0.4</strong> altı değerler ise kötü model performansına işaret eder.</p>
<p>-Gini, <strong>TAKIP_30_3</strong> öncü metriğine göre hesaplanmıştır.</p>"""

ACIKLAMA_GENEL = f"""<p>-BIN Dağılımında her bin belirli bir skor aralığını temsil eder, en düşükten en yükseğe doğru sıralanır.
<strong>Bin 1</strong> en düşük skorların bulunduğu aralığı, <strong>Bin 10</strong> en yüksek skorların bulunduğu aralığı içerir.</p>
<p>-<strong>PSI</strong> değeri <strong>&lt;0.1</strong> ise iyi, <strong>0.1-0.2</strong> arasında ise kabul edilebilir,
<strong>0.2</strong>'nin üzeri değerler modelin performansının kötüleştiğini gösterir.</p>{ACIKLAMA_GINI}
<p>-PSI hesaplamasında MM ve KKB için <strong>{REF_MM_KKB[0]} – {REF_MM_KKB[-1]}</strong>,
Öğrenci modeli için <strong>{REF_OGR[0]} – {REF_OGR[-1]}</strong> dönemi baz alınmıştır.</p>"""

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

ALT = '<div style="font-size:14px;color:black;font-weight:normal;">{}</div>'

mail_gonder(
    "(Otomatik Mail): Yeni Kart Satış Model Performans Raporu",
    html_govde(CSS_GENEL,
        f"<div>{warning_html}</div>{renkli_html(psi_gini_tbl)}<br>"
        f"<h3>Müşteri Model Skor BIN Dağılımı</h3>{bin_mm_.to_html()}"
        f"<h3>KKB Skor BIN Dağılımı</h3>{bin_kkb_.to_html()}"
        + ALT.format(ACIKLAMA_GENEL)))

mail_gonder(
    "(Otomatik Mail): Yeni Kart Satış Segment Bazlı Model Performans Raporu",
    html_govde(CSS_KIRILIM, renkli_html(seg_tbl) + ALT.format(ACIKLAMA_GINI)))

mail_gonder(
    "(Otomatik Mail): Yeni Kart Satış Kanal Bazlı Model Performans Raporu",
    html_govde(CSS_KIRILIM, renkli_html(kanal_tbl) + ALT.format(ACIKLAMA_GINI)))

print("Toplam süre:", datetime.now() - t0)
