import requests
import os
import logging

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

def send_telegram_message(text):
    """
    Отправляет сообщение в Telegram через Bot API.
    Возвращает True при успехе, иначе False.
    """
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logging.error("❌ TELEGRAM_TOKEN or CHAT_ID not set")
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=15)
        if response.status_code == 200:
            logging.info("✅ Telegram message sent successfully")
            return True
        else:
            logging.error(f"❌ Telegram API error: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.Timeout:
        logging.error("❌ Telegram API timeout")
        return False
    except requests.exceptions.ConnectionError as e:
        logging.error(f"❌ Connection error: {e}")
        return False
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")
        return False