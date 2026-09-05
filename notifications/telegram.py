"""
Telegram Bot notification client for high-priority wildlife alerts.
"""

from typing import Optional
import os
import requests

from utils.logger import get_logger

logger = get_logger(__name__)


class TelegramNotifier:
    """
    Sends alerts directly to Telegram channels or ranger group chats.
    """
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.is_configured = bool(self.bot_token and self.chat_id)

    def send_alert(
        self,
        message: str,
        severity: str = "WARNING",
        snapshot_path: Optional[str] = None,
    ) -> bool:
        if not self.is_configured:
            logger.info(f"[Mock Telegram] Severity: {severity} | Msg: {message}")
            return True

        text = f"🚨 *WILDLIFE ALERT [{severity}]* 🚨\n\n{message}"
        api_url = f"https://api.telegram.org/bot{self.bot_token}"

        try:
            if snapshot_path and os.path.exists(snapshot_path):
                # Send photo with caption
                with open(snapshot_path, "rb") as photo:
                    resp = requests.post(
                        f"{api_url}/sendPhoto",
                        data={"chat_id": self.chat_id, "caption": text, "parse_mode": "Markdown"},
                        files={"photo": photo},
                        timeout=10,
                    )
            else:
                resp = requests.post(
                    f"{api_url}/sendMessage",
                    json={"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"},
                    timeout=10,
                )

            if resp.status_code == 200:
                logger.info("Sent Telegram alert successfully.")
                return True
            else:
                logger.warning(f"Telegram API responded with {resp.status_code}: {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
            return False
