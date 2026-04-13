"""
config.py — Централизованная конфигурация TradeBot v7.0
v7.0 — Исправлен data leakage + Stacking + Feature Pruning + Новые признаки:
  - Hurst Exponent, VWAP Deviation, Realized Volatility, Order Flow Imbalance
  - Kelly Criterion position sizing
  - Stacking ансамбль (LogReg поверх XGB+LGBM)
  - Исправлен meta-labeling (OOF вместо in-sample)
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
SYMBOL    = "BTC/USDT"
TIMEFRAME = "1h"

# ═══════════════════════════════════════════
# РИСК-МЕНЕДЖМЕНТ
# ═══════════════════════════════════════════
STOP_LOSS_PCT   = 0.015
TAKE_PROFIT_PCT = 0.030

ATR_SL_MULT  = 1.5
ATR_TP_MULT  = 3.5
SL_FLOOR_PCT = 0.008
SL_CAP_PCT   = 0.040

TRAILING_ENABLED        = True
TRAILING_ACTIVATION_PCT = 0.015
TRAILING_DISTANCE_PCT   = 0.008
BREAKEVEN_ACTIVATION    = 0.010

TRADE_AMOUNT = 10.0

# ═══════════════════════════════════════════
# ПОРОГИ СИГНАЛОВ v7.0
# ═══════════════════════════════════════════
MIN_CONFIDENCE          = 0.58
STRONG_SIGNAL           = 0.65
SIGNAL_INTERVAL_MINUTES = 60

CONFIDENCE_PERCENTILE   = 50

MTF_ENABLED             = True
BTC_FILTER_ENABLED      = True
BTC_CORRELATION_THRESH  = -0.04

REGIME_FILTER_ENABLED   = True
REGIME_ADX_THRESHOLD    = 20.0

# ═══════════════════════════════════════════
# ПЕРЕОБУЧЕНИЕ
# ═══════════════════════════════════════════
RETRAIN_DAY          = os.getenv("RETRAIN_DAY",  "sunday")
RETRAIN_HOUR         = int(os.getenv("RETRAIN_HOUR", "2"))
RETRAIN_INTERVAL_HRS = 24
MIN_NEW_SAMPLES      = 50

WF_TRAIN_DAYS = 90
WF_TEST_DAYS  = 14
WF_STEP_DAYS  = 7

# ═══════════════════════════════════════════
# ML / МОДЕЛИ v7.0
# ═══════════════════════════════════════════
MODEL_PATH_BUY_XGB   = "model_buy_xgb.pkl"
MODEL_PATH_BUY_LGBM  = "model_buy_lgbm.pkl"
MODEL_PATH_SELL_XGB  = "model_sell_xgb.pkl"
MODEL_PATH_SELL_LGBM = "model_sell_lgbm.pkl"
MODEL_FEATURES_PATH  = "data/model_features.json"
STATS_FILE           = "training_stats.json"

# Новые модели v7.0
STACK_MODEL_BUY_PATH  = "stack_model_buy.pkl"
STACK_MODEL_SELL_PATH = "stack_model_sell.pkl"
FEATURE_IMPORTANCE_PATH = "feature_importance.json"

# Обратная совместимость
MODEL_PATH      = "model_buy_xgb.pkl"
MODEL_PATH_LGBM = "model_buy_lgbm.pkl"

# ── Фичи 1H (оригинальные) ───────────────
FEATURE_COLS_1H = [
    'RSI_14', 'RSI_7', 'RSI_21',
    'MACD', 'MACD_signal', 'MACD_hist',
    'ATR_pct', 'ATR_norm', 'ATR_ratio',
    'ADX', 'BB_pos', 'BB_width',
    'EMA_ratio_20_50', 'EMA_ratio_20_100',
    'Vol_ratio', 'OBV_norm', 'MFI_14',
    'Body_pct', 'Upper_wick', 'Lower_wick', 'Doji',
    'Return_1h', 'Return_4h', 'Return_12h', 'Return_24h',
    'StochRSI_K', 'StochRSI_D',
    'ZScore_20', 'ZScore_50',
    'WilliamsR',
    'Hour', 'DayOfWeek',
    'Momentum_10', 'ROC_10',
]

# ── Фичи 4H (оригинальные) ───────────────
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

# ── НОВЫЕ ПРОФЕССИОНАЛЬНЫЕ ФИЧИ v7.0 ─────
FEATURE_COLS_V7 = [
    # Trендовость и mean-reversion
    'Hurst',            # Hurst Exponent 1H (H>0.6=тренд, H<0.4=mean-rev)
    'Hurst_4h',         # Hurst Exponent 4H

    # VWAP (институциональная справедливая цена)
    'VWAP_dev_20',      # Отклонение от 20-бар VWAP (%)
    'VWAP_dev_50',      # Отклонение от 50-бар VWAP (%)
    'VWAP_bull_ratio',  # Доля бычьего объёма относительно VWAP

    # Реализованная волатильность
    'RV_20',            # Realized Volatility 20-баров (annualized %)
    'RV_50',            # Realized Volatility 50-баров
    'RV_ratio',         # RV_20 / RV_50 (краткосрочная vs долгосрочная)

    # Order Flow
    'OFI',              # Order Flow Imbalance (покупатели vs продавцы)

    # Производные цены
    'Price_accel',      # Ускорение цены (вторая производная)
    'Vol_cluster',      # Кластеризация волатильности (GARCH-прокси)
]

# Полный список фичей v7.0
FEATURE_COLS = FEATURE_COLS_1H + FEATURE_COLS_4H + FEATURE_COLS_V7

# Legacy (для обратной совместимости)
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