"""
config.py — Централизованная конфигурация TradeBot v4.0
v4.0 изменения:
  - Параметры ансамбля XGBoost + LightGBM
  - Multi-timeframe (1h + 4h + BTC macro-фильтр)
  - Walk-forward параметры
  - Trailing Stop-Loss
  - Расширенные фичи (40+ признаков)
  - Market Regime фильтр
"""

import os

# ═══════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ═══════════════════════════════════════════
# OKX API
# ═══════════════════════════════════════════
OKX_API_KEY    = os.getenv("OKX_API_KEY", "")
OKX_SECRET     = os.getenv("OKX_SECRET", "")
OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE", "")

# ═══════════════════════════════════════════
# ТОРГОВЫЕ ПАРАМЕТРЫ
# ═══════════════════════════════════════════
SYMBOL    = "TON/USDT"
TIMEFRAME = "1h"

# ═══════════════════════════════════════════
# РИСК-МЕНЕДЖМЕНТ
# ═══════════════════════════════════════════
STOP_LOSS_PCT   = 0.015   # 1.5% fallback
TAKE_PROFIT_PCT = 0.030   # 3.0% fallback

# Динамический SL/TP (ATR-based)
ATR_SL_MULT  = 1.5
ATR_TP_MULT  = 3.0
SL_FLOOR_PCT = 0.008
SL_CAP_PCT   = 0.040

# ── Trailing Stop ─────────────────────────
TRAILING_ENABLED        = True
TRAILING_ACTIVATION_PCT = 0.015   # Активация trailing при +1.5%
TRAILING_DISTANCE_PCT   = 0.008   # Trailing тянется на 0.8% за ценой
BREAKEVEN_ACTIVATION    = 0.010   # При +1.0% → SL в безубыток

TRADE_AMOUNT = 10.0               # Для outcome_tracker (legacy)

# ═══════════════════════════════════════════
# ПОРОГИ СИГНАЛОВ
# ═══════════════════════════════════════════
MIN_CONFIDENCE          = 0.70    # Минимум для входа (снижен — ансамбль надёжнее)
STRONG_SIGNAL           = 0.85    # Сильный сигнал → увеличенный размер
SIGNAL_INTERVAL_MINUTES = 60

# ── Ансамбль ─────────────────────────────
ENSEMBLE_CONSENSUS      = True    # Требуем совпадения XGB + LGBM
ENSEMBLE_MIN_VOTES      = 2       # Из 2 моделей оба должны согласиться

# ── Multi-Timeframe фильтр ────────────────
MTF_ENABLED             = True    # 4h подтверждение обязательно
BTC_FILTER_ENABLED      = True    # BTC macro-фильтр
BTC_CORRELATION_THRESH  = -0.03   # BTC 4h change хуже -3% → блокируем BUY

# ── Market Regime ─────────────────────────
REGIME_FILTER_ENABLED   = True    # Не торгуем в choppy-рынке
REGIME_ADX_THRESHOLD    = 20.0    # ADX < 20 = флэт → HOLD

# ═══════════════════════════════════════════
# ПЕРЕОБУЧЕНИЕ
# ═══════════════════════════════════════════
RETRAIN_DAY          = os.getenv("RETRAIN_DAY",  "sunday")
RETRAIN_HOUR         = int(os.getenv("RETRAIN_HOUR", "2"))
RETRAIN_INTERVAL_HRS = 24
MIN_NEW_SAMPLES      = 50

# Walk-forward параметры
WF_TRAIN_DAYS        = 60    # Обучение на 60 днях
WF_TEST_DAYS         = 14    # Тест на следующих 14 днях
WF_STEP_DAYS         = 7     # Шаг скользящего окна

# ═══════════════════════════════════════════
# ML / МОДЕЛЬ
# ═══════════════════════════════════════════
MODEL_PATH      = "ai_brain_xgb.pkl"
MODEL_PATH_LGBM = "ai_brain_lgbm.pkl"
STATS_FILE      = "training_stats.json"

# ── Базовые фичи (1h) ────────────────────
FEATURE_COLS_1H = [
    # RSI
    'RSI_14', 'RSI_7', 'RSI_21',
    # MACD
    'MACD', 'MACD_signal', 'MACD_hist',
    # ATR
    'ATR_pct', 'ATR_norm',
    # Trend
    'ADX', 'BB_pos', 'EMA_ratio_20_50', 'EMA_ratio_20_100',
    # Volume
    'Vol_ratio', 'OBV_norm', 'MFI_14',
    # Candle patterns
    'Body_pct', 'Upper_wick', 'Lower_wick', 'Doji',
    # Returns
    'Return_1h', 'Return_4h', 'Return_12h', 'Return_24h',
    # Stoch RSI
    'StochRSI_K', 'StochRSI_D',
    # Z-score
    'ZScore_20', 'ZScore_50',
    # Williams %R
    'WilliamsR',
    # Session
    'Hour', 'DayOfWeek',
    # Momentum
    'Momentum_10', 'ROC_10',
    # Volatility regime
    'ATR_ratio',   # ATR / ATR(50) — нормализованная волатильность
    'BB_width',    # Ширина полос Боллинджера
]

# ── 4H фичи (multi-timeframe) ─────────────
FEATURE_COLS_4H = [
    'RSI_14_4h', 'RSI_7_4h',
    'MACD_hist_4h',
    'EMA_ratio_4h',
    'ATR_pct_4h',
    'Vol_ratio_4h',
    'Return_4h_tf', 'Return_24h_tf',
    'ADX_4h',
    'BB_pos_4h',
]

# Все фичи для модели
FEATURE_COLS = FEATURE_COLS_1H + FEATURE_COLS_4H

# Обратная совместимость (для старых файлов)
FEATURE_COLS_LEGACY = [
    'RSI_14', 'RSI_7', 'MACD', 'MACD_signal', 'MACD_hist',
    'ATR_pct', 'ADX', 'BB_pos', 'EMA_ratio_20_50', 'Vol_ratio',
    'Body_pct', 'Upper_wick', 'Lower_wick',
    'Return_1h', 'Return_4h', 'Return_12h', 'Return_24h',
    'Hour'
]

# ═══════════════════════════════════════════
# ТАРГЕТ
# ═══════════════════════════════════════════
TARGET_HORIZON   = 6
TARGET_THRESHOLD = 0.012


def validate_config() -> list:
    """Возвращает список отсутствующих ключей. Пустой список = всё OK."""
    required = {
        "TELEGRAM_TOKEN":   TELEGRAM_TOKEN,
        "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID,
    }
    return [key for key, val in required.items() if not val]