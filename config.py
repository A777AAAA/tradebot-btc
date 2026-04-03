"""
config.py — Централизованная конфигурация TradeBot v4.0
ИСПРАВЛЕНО v4.1:
  - MIN_CONFIDENCE снижен 0.70→0.65 (больше сигналов)
  - REGIME_ADX_THRESHOLD снижен 20→18 (меньше блокировок)
  - BTC_CORRELATION_THRESH ужесточён -0.03→-0.04 (меньше ложных блоков)
  - WF_TRAIN_DAYS увеличен 60→90 (лучшее обобщение)
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
STOP_LOSS_PCT   = 0.015
TAKE_PROFIT_PCT = 0.030

ATR_SL_MULT  = 1.5
ATR_TP_MULT  = 3.0
SL_FLOOR_PCT = 0.008
SL_CAP_PCT   = 0.040

# ── Trailing Stop ─────────────────────────
TRAILING_ENABLED        = True
TRAILING_ACTIVATION_PCT = 0.015
TRAILING_DISTANCE_PCT   = 0.008
BREAKEVEN_ACTIVATION    = 0.010

TRADE_AMOUNT = 10.0

# ═══════════════════════════════════════════
# ПОРОГИ СИГНАЛОВ
# ═══════════════════════════════════════════
MIN_CONFIDENCE          = 0.65    # ИСПРАВЛЕНО: было 0.70 → меньше пропускаем сигналы
STRONG_SIGNAL           = 0.82    # ИСПРАВЛЕНО: было 0.85 → достижимее
SIGNAL_INTERVAL_MINUTES = 60

# ── Ансамбль ─────────────────────────────
ENSEMBLE_CONSENSUS      = True
ENSEMBLE_MIN_VOTES      = 2

# ── Multi-Timeframe фильтр ────────────────
MTF_ENABLED             = True
BTC_FILTER_ENABLED      = True
BTC_CORRELATION_THRESH  = -0.04   # ИСПРАВЛЕНО: было -0.03 → меньше ложных блоков

# ── Market Regime ─────────────────────────
REGIME_FILTER_ENABLED   = True
REGIME_ADX_THRESHOLD    = 18.0    # ИСПРАВЛЕНО: было 20.0 → меньше блокировок в боковике

# ═══════════════════════════════════════════
# ПЕРЕОБУЧЕНИЕ
# ═══════════════════════════════════════════
RETRAIN_DAY          = os.getenv("RETRAIN_DAY",  "sunday")
RETRAIN_HOUR         = int(os.getenv("RETRAIN_HOUR", "2"))
RETRAIN_INTERVAL_HRS = 24
MIN_NEW_SAMPLES      = 50

# Walk-forward параметры
WF_TRAIN_DAYS        = 90    # ИСПРАВЛЕНО: было 60 → лучшее обобщение
WF_TEST_DAYS         = 14
WF_STEP_DAYS         = 7

# ═══════════════════════════════════════════
# ML / МОДЕЛЬ
# ═══════════════════════════════════════════
MODEL_PATH      = "ai_brain_xgb.pkl"
MODEL_PATH_LGBM = "ai_brain_lgbm.pkl"
STATS_FILE      = "training_stats.json"

# ── Базовые фичи (1h) ────────────────────
FEATURE_COLS_1H = [
    'RSI_14', 'RSI_7', 'RSI_21',
    'MACD', 'MACD_signal', 'MACD_hist',
    'ATR_pct', 'ATR_norm',
    'ADX', 'BB_pos', 'EMA_ratio_20_50', 'EMA_ratio_20_100',
    'Vol_ratio', 'OBV_norm', 'MFI_14',
    'Body_pct', 'Upper_wick', 'Lower_wick', 'Doji',
    'Return_1h', 'Return_4h', 'Return_12h', 'Return_24h',
    'StochRSI_K', 'StochRSI_D',
    'ZScore_20', 'ZScore_50',
    'WilliamsR',
    'Hour', 'DayOfWeek',
    'Momentum_10', 'ROC_10',
    'ATR_ratio',
    'BB_width',
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

FEATURE_COLS = FEATURE_COLS_1H + FEATURE_COLS_4H

# Обратная совместимость
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
    required = {
        "TELEGRAM_TOKEN":   TELEGRAM_TOKEN,
        "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID,
    }
    return [key for key, val in required.items() if not val]