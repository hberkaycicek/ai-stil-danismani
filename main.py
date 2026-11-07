# -*- coding: utf-8 -*-
import dataiku, time
import pandas as pd
from dataiku import SQLExecutor2
from datetime import datetime

# =========================================================
# PARAMETRELER
# =========================================================
CONN  = "x"
CIKTI = "YENI_KART_SATIS_MPS"
BAS   = pd.Period("2023-01", freq="M")
_vars = dataiku.get_custom_variables()
SON   = pd.Period(_vars.get("RAPOR_AYI") or pd.Timestamp.now(), freq="M") - 1   # son tamamlanan ay

# =========================================================
# TEK AYLIK SQL  ({bas} = ayın 1'i, {bit} = sonraki ayın 1'i)
# =========================================================
SQL = """
WITH T AS (
  SELECT KDS_BOUT_MBB_NO, KDS_BOUT_SM_ID, KDS_BOUT_G1053, KDS_BOUT_ISLEM_ZAMANI,
         ROW_NUMBER() OVER (PARTITION BY KDS_BOUT_MBB_NO
                            ORDER BY TRUNC(KDS_BOUT_ISLEM_ZAMANI) DESC) AS SIRA
  FROM PODS.KDS_TON_ONAY_OUT_LOG
  WHERE KDS_BOUT_G5108 = 'YENI'
    AND KDS_BOUT_G1053 IN ('ONAY','RET')
    AND KDS_BOUT_ISLEM_ZAMANI >= DATE '{bas}'
    AND KDS_BOUT_ISLEM_ZAMANI <  DATE '{bit}'
),
G AS (
  SELECT KDS_BOUT_MBB_NO, KDS_BOUT_ISLEM_ZAMANI,
         ROW_NUMBER() OVER (PARTITION BY KDS_BOUT_MBB_NO
                            ORDER BY TRUNC(KDS_BOUT_ISLEM_ZAMANI) DESC) AS SIRA
  FROM PODS.KDS_TON_ONAY_OUT_LOG
  WHERE KDS_BOUT_G5108 = 'YENI'
    AND KDS_BOUT_G1053 = 'ONAY'
    AND KDS_BOUT_ISLEM_ZAMANI >= DATE '{bas}'
    AND KDS_BOUT_ISLEM_ZAMANI <  DATE '{bit}'
),
A AS (SELECT KDS_BOUT_MBB_NO, KDS_BOUT_ISLEM_ZAMANI FROM G WHERE SIRA = 1),
B AS (
  SELECT IZL_HES_MBB_NO, IZL_HES_KYT_TAR, MAX(IZL_HES_GEC_GUN) AS IZL_HES_GEC_GUN
  FROM (
    SELECT T0.*, T1.TICARI_KK_AP
    FROM (SELECT IZL_HES_MBB_NO, IZL_HES_HES_NO, IZL_HES_KYT_TAR, IZL_HES_HES_TIP, IZL_HES_GEC_GUN
          FROM PODS.IZL_THESAP_GUNLUK
          WHERE IZL_HES_KYT_TAR >= DATE '{bas}' AND IZL_HES_KYT_TAR < ADD_MONTHS(DATE '{bit}', 13)
            AND IZL_HES_HES_TIP IN ('KK')
          UNION
          SELECT IZL_HES_MBB_NO, IZL_HES_HES_NO, IZL_HES_KYT_TAR, IZL_HES_HES_TIP, IZL_HES_GEC_GUN
          FROM PODS.IZL_THESAP_GUNLUK_ARSV
          WHERE IZL_HES_KYT_TAR >= DATE '{bas}' AND IZL_HES_KYT_TAR < ADD_MONTHS(DATE '{bit}', 13)
            AND IZL_HES_HES_TIP IN ('KK')) T0
    LEFT JOIN (
      SELECT * FROM (
        SELECT HESAP_NUMARASI, URUN_TIPI,
               CASE WHEN URUN_TIPI = 5 THEN '000' || HESAP_NUMARASI ELSE HESAP_NUMARASI END AS HESAP_NUMARASI_2,
               CASE WHEN URUN_TIPI = 5 AND SUBSTR(HESAP_NUMARASI_FORMATTED,1,6) IN ('552608','552609','553056','425669') THEN 1
                    WHEN URUN_TIPI = 3 AND MEVCUT_URUN_ID IN (528,529) THEN 1
                    ELSE 0 END AS TICARI_KK_AP
        FROM PDWH.DWH_THESAP)
      WHERE TICARI_KK_AP = 1) T1
      ON TRIM(T1.HESAP_NUMARASI_2) = TRIM(T0.IZL_HES_HES_NO)
    WHERE T1.TICARI_KK_AP IS NULL)
  WHERE IZL_HES_HES_TIP = 'KK'
  GROUP BY IZL_HES_MBB_NO, IZL_HES_KYT_TAR
),
GB AS (
  SELECT A.KDS_BOUT_MBB_NO,
         MAX(CASE WHEN MONTHS_BETWEEN(B.IZL_HES_KYT_TAR, A.KDS_BOUT_ISLEM_ZAMANI) >= 0
                   AND MONTHS_BETWEEN(B.IZL_HES_KYT_TAR, A.KDS_BOUT_ISLEM_ZAMANI) <= 4
                  THEN B.IZL_HES_GEC_GUN ELSE 0 END) AS KK_GGS_4,
         MAX(CASE WHEN MONTHS_BETWEEN(B.IZL_HES_KYT_TAR, A.KDS_BOUT_ISLEM_ZAMANI) >= 0
                   AND MONTHS_BETWEEN(B.IZL_HES_KYT_TAR, A.KDS_BOUT_ISLEM_ZAMANI) <= 12
                  THEN B.IZL_HES_GEC_GUN ELSE 0 END) AS KK_GGS_12
  FROM A LEFT JOIN B
    ON TO_NUMBER(A.KDS_BOUT_MBB_NO) = B.IZL_HES_MBB_NO
   AND B.IZL_HES_KYT_TAR >= A.KDS_BOUT_ISLEM_ZAMANI
  GROUP BY A.KDS_BOUT_MBB_NO
),
C AS (
  SELECT TKR_HES_MBB_NO, TKR_HES_ACLS_TAR, TKR_HES_UYG_KOD
  FROM PODS.TKR_THESAP
  WHERE TKR_HES_ACLS_TAR >= DATE '{bas}' AND TKR_HES_ACLS_TAR < ADD_MONTHS(DATE '{bit}', 13)
),
GC AS (
  SELECT A.KDS_BOUT_MBB_NO,
         MAX(CASE WHEN C.TKR_HES_ACLS_TAR IS NOT NULL AND C.TKR_HES_UYG_KOD IN ('AX0K','AE0K','VM0K')
                   AND MONTHS_BETWEEN(C.TKR_HES_ACLS_TAR, A.KDS_BOUT_ISLEM_ZAMANI) >= 0
                   AND MONTHS_BETWEEN(C.TKR_HES_ACLS_TAR, A.KDS_BOUT_ISLEM_ZAMANI) <= 12
                  THEN 1 ELSE 0 END) AS KK_TAKIP_12
  FROM A LEFT JOIN C ON TO_NUMBER(A.KDS_BOUT_MBB_NO) = C.TKR_HES_MBB_NO
  GROUP BY A.KDS_BOUT_MBB_NO
)
SELECT TO_NUMBER(TO_CHAR(t.KDS_BOUT_ISLEM_ZAMANI,'YYYYMM'))   AS ISLEM_DONEM,
       CASE WHEN t.KDS_BOUT_G1053 = 'ONAY' THEN 1 ELSE 0 END  AS ONAY,
       t2.KDS_TON_ONAY_O5233                                  AS MM_SKOR,
       t2.KDS_TON_ONAY_O0120                                  AS KKB_SKOR,
       CASE WHEN gb.KDS_BOUT_MBB_NO IS NULL THEN NULL
            WHEN gb.KK_GGS_4 > 60 THEN 1 ELSE 0 END           AS KK_TAKIP_60_4,
       CASE WHEN gb.KDS_BOUT_MBB_NO IS NULL THEN NULL
            WHEN gb.KK_GGS_12 > 90 OR gc.KK_TAKIP_12 = 1 THEN 1 ELSE 0 END AS KK_TAKIP_90_12
FROM (SELECT * FROM T WHERE SIRA = 1) t
LEFT JOIN PKRA.KRA_TSM_BATCH_INPUT t2
       ON t.KDS_BOUT_SM_ID  = t2.KDS_TON_ONAY_SM_ID
      AND t.KDS_BOUT_MBB_NO = t2.KDS_TON_ONAY_MBB_NO
LEFT JOIN GB gb ON gb.KDS_BOUT_MBB_NO = t.KDS_BOUT_MBB_NO
LEFT JOIN GC gc ON gc.KDS_BOUT_MBB_NO = t.KDS_BOUT_MBB_NO
"""

# =========================================================
# ÇIKTI ŞEMASI + AYLIK DÖNGÜ
# =========================================================
SCHEMA = [{"name": "ISLEM_DONEM",    "type": "int"},
          {"name": "ONAY",           "type": "int"},
          {"name": "MM_SKOR",        "type": "double"},
          {"name": "KKB_SKOR",       "type": "double"},
          {"name": "KK_TAKIP_60_4",  "type": "double"},
          {"name": "KK_TAKIP_90_12", "type": "double"}]
KOLON = [s["name"] for s in SCHEMA]

ex  = SQLExecutor2(connection=CONN)
out = dataiku.Dataset(CIKTI)
out.write_schema(SCHEMA)

def calistir(sql, deneme=3, bekle=120):
    for i in range(deneme):
        try:
            return ex.query_to_df(sql)
        except Exception as e:
            if "ORA-01652" in str(e) and i < deneme - 1:
                print(f"   TEMP dolu, {bekle} sn sonra tekrar ({i + 2}/{deneme})")
                time.sleep(bekle)
            else:
                raise

t0, toplam, toplam_onay = datetime.now(), 0, 0
with out.get_writer() as w:
    for ay in pd.period_range(BAS, SON, freq="M"):
        t1  = datetime.now()
        bas = ay.start_time.strftime("%Y-%m-%d")
        bit = (ay + 1).start_time.strftime("%Y-%m-%d")
        df = calistir(SQL.format(bas=bas, bit=bit))
        df.columns = [c.upper() for c in df.columns]
        df = df[KOLON].apply(pd.to_numeric, errors="coerce")
        w.write_dataframe(df)
        n, n_onay = len(df), int((df["ONAY"] == 1).sum())
        toplam, toplam_onay = toplam + n, toplam_onay + n_onay
        print(f"{ay} | {n:>11,} satır | onay {n_onay:>10,} | "
              f"takip dolu (onay) {df.loc[df['ONAY'] == 1, 'KK_TAKIP_60_4'].notna().mean():.3f} | "
              f"{datetime.now() - t1}")
        del df

print(f"TOPLAM: {toplam:,} satır | onay {toplam_onay:,} | süre {datetime.now() - t0}")
