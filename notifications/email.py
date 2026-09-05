"""
Email alert notification service via SMTP with TLS encryption.
"""

from typing import List, Optional
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

from utils.logger import get_logger

logger = get_logger(__name__)


class EmailNotifier:
    """
    Dispatches formatted security and health alerts to ranger email lists.
    """
    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
    ):
        self.host = smtp_host or os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.port = int(smtp_port or os.getenv("SMTP_PORT", 587))
        self.user = smtp_user or os.getenv("SMTP_USER", "")
        self.password = smtp_password or os.getenv("SMTP_PASSWORD", "")
        self.is_configured = bool(self.user and self.password)

    def send_alert(
        self,
        recipients: List[str],
        subject: str,
        message: str,
        snapshot_path: Optional[str] = None,
    ) -> bool:
        if not self.is_configured:
            logger.info(f"[Mock Email] To: {recipients} | Subject: {subject} | Body: {message}")
            return True

        try:
            msg = MIMEMultipart()
            msg["From"] = self.user
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = f"[WILDLIFE ALERT] {subject}"

            body = f"""
            <html>
            <body>
                <h2 style="color: #e74c3c;">Wildlife System Alert Notification</h2>
                <p><strong>Details:</strong> {message}</p>
                <hr>
                <p style="font-size: 0.8em; color: #7f8c8d;">Autonomous AI Wildlife Monitoring System</p>
            </body>
            </html>
            """
            msg.attach(MIMEText(body, "html"))

            if snapshot_path and os.path.exists(snapshot_path):
                with open(snapshot_path, "rb") as f:
                    img_data = f.read()
                img = MIMEImage(img_data)
                img.add_header("Content-Disposition", "attachment", filename=os.path.basename(snapshot_path))
                msg.attach(img)

            with smtplib.SMTP(self.host, self.port, timeout=10) as server:
                server.starttls()
                server.login(self.user, self.password)
                server.send_message(msg)

            logger.info(f"Sent email alert to {recipients}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
