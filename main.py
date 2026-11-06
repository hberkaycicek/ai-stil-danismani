import smtplib, dataiku
from email.mime.text import MIMEText
from datetime import datetime

SENDER = "berkay_cicek@akbank.com"
TO     = ["berkay_cicek@akbank.com"]

body = (f"{dataiku.default_project_key()} / LDS_MPS_TRD_HARIC_AYLIK scenario'su hata verdi.\n"
        f"Zaman: {datetime.now():%Y-%m-%d %H:%M}\n"
        f"Detay: Dataiku → Scenarios → LDS_MPS_TRD_HARIC_AYLIK → Last runs")
msg = MIMEText(body, "plain", "utf-8")
msg["From"], msg["To"] = SENDER, ",".join(TO)
msg["Subject"] = "[HATA] LDS Model Performans (TRD HARİÇ) scenario"
with smtplib.SMTP("smtp.akbank.com") as s:
    s.sendmail(SENDER, TO, msg.as_string())
