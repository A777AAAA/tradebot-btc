import os

# TELEGRAM
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# OKX API (Публичный)
OKX_API_KEY    = os.getenv("OKX_API_KEY", "")
OKX_SECRET     = os.getenv("OKX_SECRET", "")
OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE", "")

# ТОРГОВЫЕ ПАРАМЕТРЫ
SYMBOL        = "TON/USDT"
TIMEFRAME     = "1h"
TRADE_AMOUNT  = 10.0

# РИСК МЕНЕДЖМЕНТ
STOP_LOSS_PCT   = 0.015  # -1.5%
TAKE_PROFIT_PCT = 0.030  # +3.0%

# ПОРОГИ СИГНАЛОВ (Сделали строже!)
MIN_CONFIDENCE = 0.75    # Бот купит только если уверен на 75%
STRONG_SIGNAL  = 0.85    # Очень сильный сигнал - 85%
SIGNAL_INTERVAL_MINUTES = 60

def validate_config():
    required = {"TELEGRAM_TOKEN": TELEGRAM_TOKEN, "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID}
    return [key for key, val in required.items() if not val]