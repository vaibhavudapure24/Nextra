"""
Twilio SMS alert notification client.
"""

from typing import Optional
import os
import requests

from utils.logger import get_logger

logger = get_logger(__name__)


class TwilioSMSNotifier:
    """
    Sends urgent SMS alerts to wildlife rangers using the Twilio REST API.
    """
    def __init__(
        self,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        from_phone: Optional[str] = None,
    ):
        self.account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID", "")
        self.auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN", "")
        self.from_phone = from_phone or os.getenv("TWILIO_PHONE_NUMBER", "")
        self.is_configured = bool(self.account_sid and self.auth_token and self.from_phone)

    def send_sms(self, to_phone: str, message: str) -> bool:
        if not self.is_configured:
            logger.info(f"[Mock Twilio SMS] To: {to_phone} | Msg: {message}")
            return True

        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
        data = {
            "From": self.from_phone,
            "To": to_phone,
            "Body": f"[WILDLIFE ALERT] {message}",
        }

        try:
            resp = requests.post(
                url, data=data, auth=(self.account_sid, self.auth_token), timeout=10
            )
            if resp.status_code in (200, 201):
                logger.info(f"Sent Twilio SMS to {to_phone}")
                return True
            else:
                logger.warning(f"Twilio API error {resp.status_code}: {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Error sending SMS via Twilio: {e}")
            return False
