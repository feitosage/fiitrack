
import smtplib, os
from email.mime.text import MIMEText

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
ALERT_TO = os.getenv("ALERT_TO")

def send_email(subject: str, body: str, to_addr: str = None):
    to_addr = to_addr or ALERT_TO
    if not all([SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, to_addr]):
        return False
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = to_addr
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, [to_addr], msg.as_string())
    return True
