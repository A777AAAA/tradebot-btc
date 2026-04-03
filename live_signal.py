"""
live_signal.py v7.0 — Stacking Ensemble + Новые признаки + Исправленный meta-filter
АРХИТЕКТУРА v7.0:
  Layer 1 — BUY-модель:  XGB_buy + LGBM_buy  -> p_buy_raw
  Layer 1 — SELL-модель: XGB_sell + LGBM_sell -> p_sell_raw
  Layer 1.5 — Stacking:  LogReg(p_xgb, p_lgbm, avg, diff) -> p_buy/p_sell (лучшая калибровка)
  Layer 2 — Meta-filter: meta_model_buy/sell  -> доп. фильтр
  Layer 3 — Фильтры: ADX + 4H MTF + BTC macro + Funding Rate
  Layer 4 — Market Regime: адаптивный порог

НОВОЕ в v7.0:
  - Stacking inference: если stack-модель доступна, используем её вероятности
    вместо простого усреднения XGB+LGBM
  - Новые признаки: Hurst, VWAP_dev, RV, OFI, Price_accel, Vol_cluster
  - Совместимость с авто-пруном: загружаем feature list из model_features.json
"""

import ccxt
import json
import joblib
import logging
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime

from config import (
    MODEL_PATH_BUY_XGB, MODEL_PATH_BUY_LGBM,
    MODEL_PATH_SELL_XGB, MODEL_PATH_SELL_LGBM,
    MODEL_FEATURES_PATH, FEATURE_COLS, FEATURE_COLS_LEGACY,
    MIN_CONFIDENCE, CONFIDENCE_PERCENTILE,
    MTF_ENABLED, BTC_FILTER_ENABLED, BTC_CORRELATION_THRESH,
    REGIME_FILTER_ENABLED, REGIME_ADX_THRESHOLD,
    SYMBOL
)

logger = logging.getLogger(__name__)

OKX_CONFIG = {'options': {'defaultType': 'spot'}, 'timeout': 30000}

META_MODEL_BUY_PATH  = "meta_model_buy.pkl"
META_MODEL_SELL_PATH = "meta_model_sell.pkl"
STACK_MODEL_BUY_PATH  = "stack_model_buy.pkl"
STACK_MODEL_SELL_PATH = "stack_model_sell.pkl"

_confidence_history: list = []
_HISTORY_MAX = 48

_funding_cache: dict = {"rate": 0.0, "oi_change": 0.0, "bias": "neutral", "ts": 0.0}
_FUNDING_TTL = 300


# ===============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ===============================================================

def _get_exchange():
    return ccxt.okx(OKX_CONFIG)


def _to_df(ohlcv: list) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=['ts', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    return df.astype(float)


def load_feature_cols() -> list:
    if os.path.exists(MODEL_FEATURES_PATH):
        with open(MODEL_FEATURES_PATH) as f:
            cols = json.load(f)
        logger.info(f"[Signal] Загружено {len(cols)} фичей из {MODEL_FEATURES_PATH}")
        return cols
    logger.warning("[Signal] features.json не найден — используем FEATURE_COLS из config")
    return FEATURE_COLS


# ===============================================================
# ORDER FLOW — Funding Rate + Open Interest
# ===============================================================

def get_funding_data(symbol_spot: str = "TON/USDT") -> dict:
    global _funding_cache

    now = time.time()
    if now - _funding_cache["ts"] < _FUNDING_TTL:
        return {
            "funding_rate":  _funding_cache["rate"],
            "oi_change_pct": _funding_cache["oi_change"],
            "funding_bias":  _funding_cache["bias"],
        }

    result = {"funding_rate": 0.0, "oi_change_pct": 0.0, "funding_bias": "neutral"}

    try:
        swap_exchange = ccxt.okx({'options': {'defaultType': 'swap'}, 'timeout': 30000})
        swap_symbol   = symbol_spot + ":USDT"

        fr_data      = swap_exchange.fetch_funding_rate(swap_symbol)
        funding_rate = float(fr_data.get("fundingRate", 0.0))

        oi_change_pct = 0.0
        try:
            oi_hist = swap_exchange.fetch_open_interest_history(
                swap_symbol, timeframe='1h', limit=3
            )
            if oi_hist and len(oi_hist) >= 2:
                oi_now  = float(oi_hist[-1].get("openInterestAmount", 1))
                oi_prev = float(oi_hist[-2].get("openInterestAmount", 1))
                if oi_prev > 0:
                    oi_change_pct = (oi_now - oi_prev) / oi_prev * 100
        except Exception:
            pass

        if funding_rate > 0.0001:
            bias = "long_crowded"
        elif funding_rate < -0.0001:
            bias = "short_crowded"
        else:
            bias = "neutral"

        result = {
            "funding_rate":  funding_rate,
            "oi_change_pct": oi_change_pct,
            "funding_bias":  bias,
        }

        _funding_cache.update({
            "rate":      funding_rate,
            "oi_change": oi_change_pct,
            "bias":      bias,
            "ts":        now,
        })

    except Exception as e:
        logger.debug(f"[OrderFlow] Недоступно (нормально для spot): {e}")
        _funding_cache["ts"] = now

    return result


# ===============================================================
# ПРОФЕССИОНАЛЬНЫЕ ПРИЗНАКИ (синхронизированы с auto_trainer v7.0)
# ===============================================================

def _calc_hurst_window(series: np.ndarray, lags_range=range(2, 21)) -> float:
    """Hurst Exponent для последних N значений."""
    if len(series) < 20:
        return 0.5
    try:
        lags = list(lags_range)
        tau  = [max(np.std(np.subtract(series[lag:], series[:-lag])), 1e-9)
                for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return float(max(0.0, min(1.0, poly[0])))
    except Exception:
        return 0.5


def _calc_vwap_dev(df: pd.DataFrame, window: int = 20) -> float:
    """VWAP отклонение последней точки."""
    try:
        tp    = (df['High'] + df['Low'] + df['Close']) / 3
        vwap  = (tp * df['Volume']).rolling(window).sum() / df['Volume'].rolling(window).sum()
        close = df['Close'].iloc[-1]
        vwap_val = float(vwap.iloc[-1])
        return (close - vwap_val) / (vwap_val + 1e-9) * 100
    except Exception:
        return 0.0


def _calc_realized_vol(close: pd.Series, window: int = 20) -> float:
    """Реализованная волатильность (annualized %)."""
    try:
        log_ret = np.log(close / close.shift(1)).dropna()
        if len(log_ret) < window:
            return 0.0
        rv = np.sqrt((log_ret**2).rolling(window).sum().iloc[-1] / window * 8760) * 100
        return float(rv)
    except Exception:
        return 0.0


def _calc_ofi(df: pd.DataFrame) -> float:
    """Order Flow Imbalance (нормализованный)."""
    try:
        close = df['Close']
        high  = df['High']
        low   = df['Low']
        vol   = df['Volume']
        bull  = ((close - low) / (high - low + 1e-9)) * vol
        bear  = ((high - close) / (high - low + 1e-9)) * vol
        ofi   = (bull - bear).rolling(10).sum()
        total = vol.rolling(10).sum()
        return float(ofi.iloc[-1] / (total.iloc[-1] + 1e-9))
    except Exception:
        return 0.0


# ===============================================================
# ИНДИКАТОРЫ 1H
# ===============================================================

def calc_indicators_1h(df: pd.DataFrame) -> pd.DataFrame:
    d     = df.copy()
    close = d['Close']
    high  = d['High']
    low   = d['Low']
    vol   = d['Volume']

    d['Hour']      = d.index.hour
    d['DayOfWeek'] = d.index.dayofweek

    for p in [7, 14, 21]:
        diff  = close.diff()
        g     = diff.clip(lower=0)
        l     = -diff.clip(upper=0)
        avg_g = g.ewm(com=p - 1, min_periods=p).mean()
        avg_l = l.ewm(com=p - 1, min_periods=p).mean()
        d[f'RSI_{p}'] = 100 - (100 / (1 + avg_g / (avg_l + 1e-9)))

    ema12            = close.ewm(span=12, adjust=False).mean()
    ema26            = close.ewm(span=26, adjust=False).mean()
    d['MACD']        = ema12 - ema26
    d['MACD_signal'] = d['MACD'].ewm(span=9, adjust=False).mean()
    d['MACD_hist']   = d['MACD'] - d['MACD_signal']

    tr    = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr14         = tr.ewm(com=13, min_periods=14).mean()
    atr50         = tr.ewm(com=49, min_periods=50).mean()
    d['ATR']      = atr14
    d['ATR_pct']  = (atr14 / (close + 1e-9)) * 100
    d['ATR_norm'] = atr14 / (close + 1e-9)
    d['ATR_ratio']= atr14 / (atr50 + 1e-9)

    sma20         = close.rolling(20).mean()
    std20         = close.rolling(20).std()
    bb_upper      = sma20 + 2 * std20
    bb_lower      = sma20 - 2 * std20
    d['BB_pos']   = (close - bb_lower) / (4 * std20 + 1e-9)
    d['BB_width'] = (bb_upper - bb_lower) / (sma20 + 1e-9)

    ema20  = close.ewm(span=20).mean()
    ema50  = close.ewm(span=50).mean()
    ema100 = close.ewm(span=100).mean()
    d['EMA_ratio_20_50']  = ema20 / (ema50  + 1e-9)
    d['EMA_ratio_20_100'] = ema20 / (ema100 + 1e-9)
    d['EMA_ratio']        = d['EMA_ratio_20_50']

    vol_sma20      = vol.rolling(20).mean()
    d['Vol_ratio'] = vol / (vol_sma20 + 1e-9)

    obv           = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    obv_sma20     = obv.rolling(20).mean()
    d['OBV_norm'] = (obv - obv_sma20) / (obv.rolling(20).std() + 1e-9)

    tp          = (high + low + close) / 3
    mf          = tp * vol
    pos_mf      = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_mf      = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
    d['MFI_14'] = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-9)))

    rsi14           = d['RSI_14']
    stoch_min       = rsi14.rolling(14).min()
    stoch_max       = rsi14.rolling(14).max()
    stoch_k         = (rsi14 - stoch_min) / (stoch_max - stoch_min + 1e-9) * 100
    d['StochRSI_K'] = stoch_k
    d['StochRSI_D'] = stoch_k.rolling(3).mean()

    hw14           = high.rolling(14).max()
    lw14           = low.rolling(14).min()
    d['WilliamsR'] = (hw14 - close) / (hw14 - lw14 + 1e-9) * -100

    d['ZScore_20'] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-9)
    d['ZScore_50'] = (close - close.rolling(50).mean()) / (close.rolling(50).std() + 1e-9)

    up   = high.diff()
    down = -low.diff()
    pdm  = up.where((up > down)   & (up > 0),   0)
    mdm  = down.where((down > up) & (down > 0), 0)
    pdi  = 100 * (pdm.ewm(alpha=1/14).mean() / (atr14 + 1e-9))
    mdi  = 100 * (mdm.ewm(alpha=1/14).mean() / (atr14 + 1e-9))
    dx   = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    d['ADX'] = dx.ewm(alpha=1/14).mean()

    d['Body_pct']   = (close - d['Open']).abs() / (d['Open'] + 1e-9) * 100
    d['Upper_wick'] = (high - d[['Close','Open']].max(axis=1)) / (d['Open'] + 1e-9) * 100
    d['Lower_wick'] = (d[['Close','Open']].min(axis=1) - low) / (d['Open'] + 1e-9) * 100
    d['Doji']       = ((d['Body_pct'] / (high - low + 1e-9)) < 0.1).astype(int)

    d['Momentum_10'] = close - close.shift(10)
    d['ROC_10']      = close.pct_change(10) * 100

    for h in [1, 4, 12, 24]:
        d[f'Return_{h}h'] = close.pct_change(h) * 100

    # ── НОВЫЕ ПРИЗНАКИ v7.0 ──

    # Hurst Exponent
    d['Hurst'] = close.rolling(100, min_periods=50).apply(
        lambda x: _calc_hurst_window(x), raw=True
    )

    # VWAP отклонения
    tp_series = (high + low + close) / 3
    vwap_20   = (tp_series * vol).rolling(20).sum() / vol.rolling(20).sum()
    vwap_50   = (tp_series * vol).rolling(50).sum() / vol.rolling(50).sum()
    d['VWAP_dev_20']   = (close - vwap_20) / (vwap_20 + 1e-9) * 100
    d['VWAP_dev_50']   = (close - vwap_50) / (vwap_50 + 1e-9) * 100
    bull_vol = vol.where(close > vwap_20, 0).rolling(10).sum()
    d['VWAP_bull_ratio'] = bull_vol / (vol.rolling(10).sum() + 1e-9)

    # Реализованная волатильность
    log_ret     = np.log(close / close.shift(1))
    d['RV_20']  = np.sqrt((log_ret**2).rolling(20).sum() / 20 * 8760) * 100
    d['RV_50']  = np.sqrt((log_ret**2).rolling(50).sum() / 50 * 8760) * 100
    d['RV_ratio'] = d['RV_20'] / (d['RV_50'] + 1e-9)

    # Order Flow Imbalance
    bull_frac = (close - low) / (high - low + 1e-9)
    bear_frac = (high - close) / (high - low + 1e-9)
    ofi_raw   = (bull_frac * vol - bear_frac * vol).rolling(10).sum()
    d['OFI']  = ofi_raw / (vol.rolling(10).sum() + 1e-9)

    # Ценовое ускорение
    ret_1h       = close.pct_change(1)
    d['Price_accel'] = ret_1h - ret_1h.shift(1)

    # Кластеризация волатильности
    d['Vol_cluster'] = (log_ret**2).ewm(span=5).mean() / ((log_ret**2).ewm(span=20).mean() + 1e-9)

    return d.dropna()


# ===============================================================
# ИНДИКАТОРЫ 4H
# ===============================================================

def calc_indicators_4h(df4h: pd.DataFrame) -> pd.DataFrame:
    d     = df4h.copy()
    close = d['Close']
    high  = d['High']
    low   = d['Low']
    vol   = d['Volume']

    for p in [7, 14]:
        diff  = close.diff()
        g     = diff.clip(lower=0)
        l     = -diff.clip(upper=0)
        avg_g = g.ewm(com=p - 1, min_periods=p).mean()
        avg_l = l.ewm(com=p - 1, min_periods=p).mean()
        d[f'RSI_{p}_4h'] = 100 - (100 / (1 + avg_g / (avg_l + 1e-9)))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    d['MACD_hist_4h'] = macd - macd.ewm(span=9, adjust=False).mean()

    ema20 = close.ewm(span=20).mean()
    ema50 = close.ewm(span=50).mean()
    d['EMA_ratio_4h'] = ema20 / (ema50 + 1e-9)

    tr    = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.ewm(com=13, min_periods=14).mean()
    d['ATR_pct_4h'] = (atr14 / (close + 1e-9)) * 100

    d['Vol_ratio_4h']  = vol / (vol.rolling(20).mean() + 1e-9)
    d['Return_4h_tf']  = close.pct_change(1) * 100
    d['Return_24h_tf'] = close.pct_change(6) * 100

    up   = high.diff()
    down = -low.diff()
    pdm  = up.where((up > down)   & (up > 0),   0)
    mdm  = down.where((down > up) & (down > 0), 0)
    pdi  = 100 * (pdm.ewm(alpha=1/14).mean() / (atr14 + 1e-9))
    mdi  = 100 * (mdm.ewm(alpha=1/14).mean() / (atr14 + 1e-9))
    dx   = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    d['ADX_4h'] = dx.ewm(alpha=1/14).mean()

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    d['BB_pos_4h'] = (close - (sma20 - 2*std20)) / (4*std20 + 1e-9)

    # Hurst для 4H (трендовость на старшем ТФ)
    d['Hurst_4h'] = close.rolling(60, min_periods=30).apply(
        lambda x: _calc_hurst_window(x, range(2, 15)), raw=True
    )

    return d.dropna()


# ===============================================================
# BTC MACRO FILTER
# ===============================================================

def get_btc_4h_change(exchange) -> float:
    try:
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe='4h', limit=5)
        if not ohlcv or len(ohlcv) < 2:
            return 0.0
        return (float(ohlcv[-1][4]) - float(ohlcv[-2][4])) / float(ohlcv[-2][4]) * 100
    except Exception as e:
        logger.warning(f"[Signal] BTC: {e}")
        return 0.0


# ===============================================================
# MARKET REGIME DETECTION
# ===============================================================

def detect_market_regime(adx: float, atr_ratio: float, bb_width: float) -> dict:
    if adx > 30 and atr_ratio < 1.5:
        return {"regime": "TRENDING",  "mult": 1.00, "note": f"Тренд ADX={adx:.1f}"}
    elif adx < 20 and bb_width < 0.05:
        return {"regime": "RANGING",   "mult": 1.15, "note": f"Боковик ADX={adx:.1f}"}
    elif atr_ratio > 1.8:
        return {"regime": "VOLATILE",  "mult": 1.25, "note": f"Волатильность ATR_r={atr_ratio:.2f}"}
    else:
        return {"regime": "NEUTRAL",   "mult": 1.05, "note": f"Нейтральный ADX={adx:.1f}"}


# ===============================================================
# PERCENTILE FILTER
# ===============================================================

def _percentile_filter(confidence: float) -> bool:
    global _confidence_history
    _confidence_history.append(confidence)
    if len(_confidence_history) > _HISTORY_MAX:
        _confidence_history = _confidence_history[-_HISTORY_MAX:]
    if len(_confidence_history) < 10:
        return True
    threshold = np.percentile(_confidence_history, CONFIDENCE_PERCENTILE)
    return confidence >= threshold


# ===============================================================
# ЗАГРУЗКА МОДЕЛЕЙ
# ===============================================================

def _load_models() -> dict:
    models = {}

    for key, path in [
        ('buy_xgb',   MODEL_PATH_BUY_XGB),
        ('buy_lgbm',  MODEL_PATH_BUY_LGBM),
        ('sell_xgb',  MODEL_PATH_SELL_XGB),
        ('sell_lgbm', MODEL_PATH_SELL_LGBM),
    ]:
        if os.path.exists(path):
            try:
                models[key] = joblib.load(path)
            except Exception as e:
                logger.warning(f"[Signal] Не удалось загрузить {path}: {e}")

    for key, path in [
        ('meta_buy',  META_MODEL_BUY_PATH),
        ('meta_sell', META_MODEL_SELL_PATH),
        ('stack_buy',  STACK_MODEL_BUY_PATH),
        ('stack_sell', STACK_MODEL_SELL_PATH),
    ]:
        if os.path.exists(path):
            try:
                models[key] = joblib.load(path)
                logger.info(f"[Signal] Загружена модель: {path}")
            except Exception as e:
                logger.warning(f"[Signal] {path}: {e}")

    return models


# ===============================================================
# STACKING INFERENCE (новое v7.0)
# ===============================================================

def _apply_stacking(
    models: dict,
    p_xgb: float,
    p_lgbm: float,
    direction: str  # 'buy' или 'sell'
) -> float:
    """
    Применяет stacking-ансамбль если доступен.
    Stacking даёт лучшую калибровку вероятностей.
    Если stack-модель недоступна — возвращает простое среднее.
    """
    stack_key = f'stack_{direction}'
    if stack_key not in models:
        return (p_xgb + p_lgbm) / 2.0

    try:
        stack_bundle = models[stack_key]
        model  = stack_bundle['model']
        scaler = stack_bundle['scaler']

        p_avg  = (p_xgb + p_lgbm) / 2
        p_diff = p_xgb - p_lgbm
        X_stack = np.array([[p_xgb, p_lgbm, p_avg, p_diff]])
        X_scaled = scaler.transform(X_stack)

        p_stacked = float(model.predict_proba(X_scaled)[0][1])
        logger.debug(
            f"[Stack {direction.upper()}] xgb={p_xgb:.1%} lgbm={p_lgbm:.1%} "
            f"avg={p_avg:.1%} → stacked={p_stacked:.1%}"
        )
        return p_stacked
    except Exception as e:
        logger.warning(f"[Signal] Stack {direction} ошибка: {e}")
        return (p_xgb + p_lgbm) / 2.0


# ===============================================================
# META-MODEL INFERENCE
# ===============================================================

def _apply_meta_filter(
    models:     dict,
    X:          np.ndarray,
    p_base:     float,
    model_key:  str,
    signal_dir: str,
) -> tuple:
    if model_key not in models:
        return None, True

    try:
        X_meta  = np.hstack([X, np.array([[p_base]])])
        p_meta  = float(models[model_key].predict_proba(X_meta)[0][1])
        passed  = p_meta >= 0.50
        logger.debug(f"[Meta] {signal_dir}: p_meta={p_meta:.1%} {'OK' if passed else 'BLOCK'}")
        return p_meta, passed
    except Exception as e:
        logger.warning(f"[Meta] Ошибка {model_key}: {e}")
        return None, True


# ===============================================================
# FUNDING RATE КОРРЕКЦИЯ
# ===============================================================

def _apply_funding_correction(
    signal:     str,
    confidence: float,
    funding:    dict,
    threshold:  float,
) -> tuple:
    funding_rate = funding.get("funding_rate", 0.0)
    oi_change    = funding.get("oi_change_pct", 0.0)
    bias         = funding.get("funding_bias", "neutral")
    note         = ""

    if signal == "BUY" and bias == "long_crowded":
        confidence = confidence * 0.90
        note = f"Funding={funding_rate:.4%} long_crowded -10%"

    elif signal == "BUY" and oi_change < -2.0:
        confidence = confidence * 0.92
        note = f"OI_delta={oi_change:+.1f}% -8%"

    elif signal == "SELL" and bias == "short_crowded":
        confidence = confidence * 0.88
        note = f"Funding={funding_rate:.4%} short_crowded -12%"

    if signal != "HOLD" and confidence < threshold:
        note += f" -> ниже порога {threshold:.1%}"
        signal = "HOLD"

    return signal, confidence, note


# ===============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ===============================================================

def get_live_signal(symbol: str = "TON/USDT") -> dict | None:
    """
    Полный pipeline v7.0:
      1. Загружаем модели (базовые + stack + мета)
      2. Получаем 1H + 4H свечи + Funding Rate
      3. Считаем индикаторы (включая новые: Hurst, VWAP, RV, OFI)
      4. Layer 1: XGB + LGBM -> p_xgb, p_lgbm
      4.5. Layer 1.5: Stacking -> p_buy/p_sell (или avg если нет stack)
      5. Market Regime -> адаптивный порог
      6. Layer 2: Meta-model фильтр
      7. Фильтры: percentile -> ADX -> 4H MTF -> BTC -> Funding
      8. Возвращаем результат
    """
    try:
        start_ts = time.time()

        # 1. Модели
        models = _load_models()
        if 'buy_xgb' not in models and 'sell_xgb' not in models:
            logger.warning("[Signal] Нет ни одной модели")
            return None

        feature_cols = load_feature_cols()

        # 2. Данные
        exchange = _get_exchange()
        ohlcv_1h = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=250)

        if not ohlcv_1h or len(ohlcv_1h) < 150:
            logger.warning("[Signal] Мало 1H данных")
            return None

        df1h_feats = calc_indicators_1h(_to_df(ohlcv_1h))
        if df1h_feats.empty:
            return None

        df4h_feats = None
        if MTF_ENABLED:
            try:
                ohlcv_4h = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=100)
                if ohlcv_4h and len(ohlcv_4h) >= 50:
                    df4h_feats = calc_indicators_4h(_to_df(ohlcv_4h))
            except Exception as e:
                logger.warning(f"[Signal] 4H ошибка: {e}")

        funding_data = get_funding_data(symbol)

        # 3. Вектор фичей
        last_1h  = df1h_feats.iloc[-1]
        row_data = {}
        for col in feature_cols:
            if col in df1h_feats.columns:
                row_data[col] = float(last_1h[col])
            elif (col.endswith('_4h') or col.endswith('_4h_tf')) and df4h_feats is not None:
                row_data[col] = float(df4h_feats.iloc[-1][col]) if col in df4h_feats.columns else 0.0
            else:
                row_data[col] = 0.0

        X = np.array([[row_data[c] for c in feature_cols]], dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 4. Layer 1: базовые вероятности
        p_buy_xgb   = float(models['buy_xgb'].predict_proba(X)[0][1])   if 'buy_xgb'   in models else 0.0
        p_buy_lgbm  = float(models['buy_lgbm'].predict_proba(X)[0][1])  if 'buy_lgbm'  in models else p_buy_xgb
        p_sell_xgb  = float(models['sell_xgb'].predict_proba(X)[0][1])  if 'sell_xgb'  in models else 0.0
        p_sell_lgbm = float(models['sell_lgbm'].predict_proba(X)[0][1]) if 'sell_lgbm' in models else p_sell_xgb

        # 4.5. Layer 1.5: Stacking (новое v7.0)
        p_buy  = _apply_stacking(models, p_buy_xgb,  p_buy_lgbm,  'buy')
        p_sell = _apply_stacking(models, p_sell_xgb, p_sell_lgbm, 'sell')

        has_stack = 'stack_buy' in models or 'stack_sell' in models
        models_used = "+".join(filter(None, [
            "XGB"   if 'buy_xgb'   in models else "",
            "LGBM"  if 'buy_lgbm'  in models else "",
            "STACK" if has_stack              else "",
        ]))

        # 5. Market Regime -> адаптивный порог
        adx_1h    = float(last_1h.get('ADX',       25.0))
        atr_ratio = float(last_1h.get('ATR_ratio',  1.0))
        bb_width  = float(last_1h.get('BB_width',  0.05))
        rsi_14    = float(last_1h.get('RSI_14',    50.0))
        vol_ratio = float(last_1h.get('Vol_ratio',  1.0))
        hurst     = float(last_1h.get('Hurst',      0.5))

        regime              = detect_market_regime(adx_1h, atr_ratio, bb_width)
        regime_mult         = regime["mult"]

        # Дополнительный Hurst-фильтр: если рынок случайный (H≈0.5) — поднимаем порог
        if 0.45 <= hurst <= 0.55:
            regime_mult = max(regime_mult, 1.10)
            logger.debug(f"[Signal] Hurst={hurst:.3f} близко к 0.5 — рынок случайный, mult={regime_mult:.2f}")

        effective_threshold = MIN_CONFIDENCE * regime_mult

        # 6. Первичный сигнал
        if p_buy >= effective_threshold and p_sell >= effective_threshold:
            signal, confidence = ("BUY", p_buy) if p_buy >= p_sell else ("SELL", p_sell)
        elif p_buy >= effective_threshold:
            signal, confidence = "BUY",  p_buy
        elif p_sell >= effective_threshold:
            signal, confidence = "SELL", p_sell
        else:
            signal, confidence = "HOLD", max(p_buy, p_sell)

        # 7. Layer 2: Meta-model фильтр
        p_meta_buy   = None
        p_meta_sell  = None
        meta_blocked = False

        if signal == "BUY" and 'meta_buy' in models:
            p_meta_buy, meta_ok = _apply_meta_filter(
                models, X, p_buy, 'meta_buy', 'BUY'
            )
            if not meta_ok:
                logger.info(f"[Signal] Meta-BUY заблокировал: p_meta={p_meta_buy:.1%}")
                signal       = "HOLD"
                meta_blocked = True

        elif signal == "SELL" and 'meta_sell' in models:
            p_meta_sell, meta_ok = _apply_meta_filter(
                models, X, p_sell, 'meta_sell', 'SELL'
            )
            if not meta_ok:
                logger.info(f"[Signal] Meta-SELL заблокировал: p_meta={p_meta_sell:.1%}")
                signal       = "HOLD"
                meta_blocked = True

        # 8. Фильтр-цепочка
        filter_log = []

        if signal != "HOLD":
            if not _percentile_filter(confidence):
                filter_log.append(f"PERCENTILE_LOW_{confidence:.1%}")
                signal = "HOLD"

        if REGIME_FILTER_ENABLED and signal != "HOLD" and adx_1h < REGIME_ADX_THRESHOLD:
            filter_log.append(f"ADX={adx_1h:.1f}<{REGIME_ADX_THRESHOLD}")
            signal = "HOLD"

        mtf_confirmed = True
        if MTF_ENABLED and df4h_feats is not None and signal != "HOLD":
            last_4h      = df4h_feats.iloc[-1]
            rsi_4h       = float(last_4h.get('RSI_14_4h',  50.0))
            ema_ratio_4h = float(last_4h.get('EMA_ratio_4h', 1.0))

            if signal == "BUY":
                mtf_ok = (ema_ratio_4h > 0.995) and (rsi_4h > 40)
            else:
                mtf_ok = (ema_ratio_4h < 1.005) and (rsi_4h < 60)

            if not mtf_ok:
                filter_log.append(f"4H_RSI={rsi_4h:.0f}_EMA={ema_ratio_4h:.4f}")
                signal        = "HOLD"
                mtf_confirmed = False

        btc_change  = 0.0
        btc_blocked = False
        if BTC_FILTER_ENABLED and signal == "BUY":
            btc_change = get_btc_4h_change(exchange)
            if btc_change < BTC_CORRELATION_THRESH:
                filter_log.append(f"BTC_4H={btc_change:+.2f}%")
                signal      = "HOLD"
                btc_blocked = True

        funding_note = ""
        if signal != "HOLD":
            signal, confidence, funding_note = _apply_funding_correction(
                signal, confidence, funding_data, effective_threshold
            )
            if funding_note:
                filter_log.append("FUNDING_ADJ")

        # 9. Финальные данные
        cur_price   = float(last_1h['Close'])
        current_atr = float(last_1h.get('ATR',       0.0))
        change_24h  = float(last_1h.get('Return_24h', 0.0))
        volume      = float(last_1h.get('Volume',     0.0))
        elapsed     = round(time.time() - start_ts, 2)

        p_meta = p_meta_buy if p_meta_buy is not None else p_meta_sell
        if 'meta_buy' in models or 'meta_sell' in models:
            models_used += "+META"

        logger.info(
            f"[Signal v7.0] {signal} | "
            f"p_buy={p_buy:.1%}(xgb={p_buy_xgb:.1%} lgbm={p_buy_lgbm:.1%}) | "
            f"p_sell={p_sell:.1%}(xgb={p_sell_xgb:.1%} lgbm={p_sell_lgbm:.1%}) | "
            f"Hurst={hurst:.3f} | "
            f"meta={'OK' if not meta_blocked else 'BLOCK'} | "
            f"Regime={regime['regime']}x{regime_mult:.2f} | "
            f"ADX={adx_1h:.1f} | 4H={'OK' if mtf_confirmed else 'NO'} | "
            f"BTC={btc_change:+.2f}% | "
            f"Funding={funding_data.get('funding_rate', 0):.4%} | "
            f"Price=${cur_price:.4f} | filters={filter_log} | {elapsed}s"
        )

        return {
            # Основной сигнал
            "signal":        signal,
            "confidence":    round(confidence, 4),

            # Layer 1: базовые вероятности
            "p_buy":         round(p_buy,       4),
            "p_sell":        round(p_sell,      4),
            "p_buy_xgb":     round(p_buy_xgb,   4),
            "p_buy_lgbm":    round(p_buy_lgbm,  4),
            "p_sell_xgb":    round(p_sell_xgb,  4),
            "p_sell_lgbm":   round(p_sell_lgbm, 4),

            # Layer 2: мета-модель
            "p_meta":        round(p_meta, 4) if p_meta is not None else None,
            "meta_blocked":  meta_blocked,
            "models_used":   models_used,

            # Новые метрики v7.0
            "hurst":         round(hurst, 3),

            # Рыночный контекст
            "price":         cur_price,
            "atr":           current_atr,
            "change_24h":    change_24h,
            "volume":        volume,
            "adx":           round(adx_1h, 2),
            "rsi14":         round(rsi_14,  2),

            # Режим рынка
            "regime":        regime["regime"],
            "regime_note":   regime["note"],
            "regime_mult":   regime_mult,
            "eff_threshold": round(effective_threshold, 4),

            # Order Flow
            "funding_rate":  funding_data.get("funding_rate",  0.0),
            "oi_change_pct": funding_data.get("oi_change_pct", 0.0),
            "funding_bias":  funding_data.get("funding_bias", "neutral"),
            "funding_note":  funding_note,

            # Фильтры
            "mtf_confirmed": mtf_confirmed,
            "btc_change_4h": btc_change,
            "btc_blocked":   btc_blocked,
            "filter_log":    filter_log,

            # Совместимость
            "xgb_signal":   signal,
            "lgbm_signal":  signal,

            # Служебное
            "inference_ms": int(elapsed * 1000),
            "timestamp":    datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"[Signal] Ошибка: {e}", exc_info=True)
        return None