"""
Центральный файл конфигурации TradeBot
Все настройки берутся из переменных окружения Render
"""

import os

# ═══════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ═══════════════════════════════════════════
# БИРЖА OKX (только для чтения цен — публичный API)
# ═══════════════════════════════════════════
OKX_API_KEY    = os.getenv("OKX_API_KEY", "")
OKX_SECRET     = os.getenv("OKX_SECRET", "")
OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE", "")

# ═══════════════════════════════════════════
# OPENROUTER (опционально)
# ═══════════════════════════════════════════
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ═══════════════════════════════════════════
# ТОРГОВЫЕ ПАРАМЕТРЫ
# ═══════════════════════════════════════════
SYMBOL        = "TON/USDT"
TIMEFRAME     = "1h"
TRADE_AMOUNT  = float(os.getenv("TRADE_AMOUNT", "10.0"))

# Риск менеджмент
STOP_LOSS_PCT   = 0.015
TAKE_PROFIT_PCT = 0.03
MAX_DAILY_LOSS  = float(os.getenv("MAX_DAILY_LOSS", "30.0"))

# ═══════════════════════════════════════════
# РАСПИСАНИЕ
# ═══════════════════════════════════════════
SIGNAL_INTERVAL_MINUTES = 60
RETRAIN_HOURS = 6

# ═══════════════════════════════════════════
# ПОРОГИ СИГНАЛОВ
# ═══════════════════════════════════════════
MIN_CONFIDENCE = 0.62
STRONG_SIGNAL  = 0.75


def validate_config() -> list:
    """
    Проверяет только ОБЯЗАТЕЛЬНЫЕ переменные.
    OKX ключи не нужны — используем публичный API для цен.
    Режим: Paper Trading (виртуальные сделки).
    """
    required = {
        "TELEGRAM_TOKEN":   TELEGRAM_TOKEN,
        "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID,
    }
    return [key for key, val in required.items() if not val]


if __name__ == "__main__":
    missing = validate_config()
    if missing:
        print(f"⚠️  Отсутствуют переменные: {missing}")
    else:
        print("✅ Все обязательные переменные на месте!")

    print(f"\n📊 Режим: Paper Trading (виртуальные сделки)")
    print(f"📊 Торговые параметры:")
    print(f"   Символ:      {SYMBOL}")
    print(f"   Таймфрейм:   {TIMEFRAME}")
    print(f"   Стоп-лосс:   {STOP_LOSS_PCT*100}%")
    print(f"   Тейк-профит: {TAKE_PROFIT_PCT*100}%")
    print(f"   Переобучение: каждые {RETRAIN_HOURS} часов")