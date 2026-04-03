"""
live_signal.py v5.0 — Двойные бинарные классификаторы
АРХИТЕКТУРА:
  - BUY-модель:  p_buy  = avg(XGB_buy_proba,  LGBM_buy_proba)
  - SELL-модель: p_sell = avg(XGB_sell_proba, LGBM_sell_proba)
  - Сигнал: если p_buy > MIN_CONFIDENCE → BUY, p_sell > MIN_CONFIDENCE → SELL
  - Динамический порог: торгуем только топ-35% сигналов по уверенности
  - Все прежние фильтры: ADX, 4H MTF, BTC macro
"""

import ccxt
import json
import joblib
import logging
import os
import numpy as np
import pandas as pd

from config import (
    MODEL_PATH_BUY_XGB, MODEL_PATH_BUY_LGBM,
    MODEL_PATH_SELL_XGB, MODEL_PATH_SELL_LGBM,
    MODEL_FEATURES_PATH, FEATURE_COLS, FEATURE_COLS_LEGACY,
    MIN_CONFIDENCE, CONFIDENCE_PERCENTILE,
    MTF_ENABLED, BTC_FILTER_ENABLED, BTC_CORRELATION_THRESH,
    REGIME_FILTER_ENABLED, REGIME_ADX_THRESHOLD
)

logger = logging.getLogger(__name__)

OKX_CONFIG = {'options': {'defaultType': 'spot'}, 'timeout': 30000}

# Скользящий буфер уверенности для перцентильного фильтра
_confidence_history = []
_HISTORY_MAX = 48   # 48 часов


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

    tr  = pd.concat([
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

    sma20        = close.rolling(20).mean()
    std20        = close.rolling(20).std()
    bb_upper     = sma20 + 2 * std20
    bb_lower     = sma20 - 2 * std20
    d['BB_pos']  = (close - bb_lower) / (4 * std20 + 1e-9)
    d['BB_width'] = (bb_upper - bb_lower) / (sma20 + 1e-9)

    ema20  = close.ewm(span=20).mean()
    ema50  = close.ewm(span=50).mean()
    ema100 = close.ewm(span=100).mean()
    d['EMA_ratio_20_50']  = ema20 / (ema50 + 1e-9)
    d['EMA_ratio_20_100'] = ema20 / (ema100 + 1e-9)
    d['EMA_ratio']        = d['EMA_ratio_20_50']

    vol_sma20      = vol.rolling(20).mean()
    d['Vol_ratio'] = vol / (vol_sma20 + 1e-9)

    obv           = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    obv_sma20     = obv.rolling(20).mean()
    d['OBV_norm'] = (obv - obv_sma20) / (obv.rolling(20).std() + 1e-9)

    tp            = (high + low + close) / 3
    mf            = tp * vol
    pos_mf        = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_mf        = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
    d['MFI_14']   = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-9)))

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

    return d.dropna()


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

    return d.dropna()


def get_btc_4h_change(exchange) -> float:
    try:
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe='4h', limit=5)
        if not ohlcv or len(ohlcv) < 2:
            return 0.0
        return (float(ohlcv[-1][4]) - float(ohlcv[-2][4])) / float(ohlcv[-2][4]) * 100
    except Exception as e:
        logger.warning(f"[Signal] BTC: {e}")
        return 0.0


def _percentile_filter(confidence: float) -> bool:
    """Возвращает True если уверенность выше порогового перцентиля."""
    global _confidence_history
    _confidence_history.append(confidence)
    if len(_confidence_history) > _HISTORY_MAX:
        _confidence_history = _confidence_history[-_HISTORY_MAX:]

    if len(_confidence_history) < 10:
        return True   # мало истории — пропускаем фильтр

    threshold = np.percentile(_confidence_history, CONFIDENCE_PERCENTILE)
    return confidence >= threshold


def _load_models() -> dict:
    """Загружает все 4 бинарные модели."""
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

    return models


def get_live_signal(symbol: str = "TON/USDT") -> dict | None:
    try:
        # 1. Загружаем модели
        models = _load_models()

        if 'buy_xgb' not in models and 'sell_xgb' not in models:
            logger.warning("[Signal] Модели не найдены")
            return None

        feature_cols = load_feature_cols()

        # 2. Данные
        exchange = _get_exchange()
        ohlcv_1h = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=200)

        if not ohlcv_1h or len(ohlcv_1h) < 100:
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

        # 4. Вероятности от бинарных моделей
        # BUY
        p_buy_list = []
        if 'buy_xgb' in models:
            p_buy_list.append(float(models['buy_xgb'].predict_proba(X)[0][1]))
        if 'buy_lgbm' in models:
            p_buy_list.append(float(models['buy_lgbm'].predict_proba(X)[0][1]))
        p_buy = float(np.mean(p_buy_list)) if p_buy_list else 0.0

        # SELL
        p_sell_list = []
        if 'sell_xgb' in models:
            p_sell_list.append(float(models['sell_xgb'].predict_proba(X)[0][1]))
        if 'sell_lgbm' in models:
            p_sell_list.append(float(models['sell_lgbm'].predict_proba(X)[0][1]))
        p_sell = float(np.mean(p_sell_list)) if p_sell_list else 0.0

        models_used = "+".join(
            ([f"XGB"] if 'buy_xgb' in models else []) +
            ([f"LGBM"] if 'buy_lgbm' in models else [])
        )

        # 5. Определяем сигнал
        # Если оба сильные — берём максимальный
        if p_buy >= MIN_CONFIDENCE and p_sell >= MIN_CONFIDENCE:
            if p_buy >= p_sell:
                signal, confidence = "BUY", p_buy
            else:
                signal, confidence = "SELL", p_sell
        elif p_buy >= MIN_CONFIDENCE:
            signal, confidence = "BUY", p_buy
        elif p_sell >= MIN_CONFIDENCE:
            signal, confidence = "SELL", p_sell
        else:
            signal, confidence = "HOLD", max(p_buy, p_sell)

        # 6. Перцентильный фильтр
        if signal != "HOLD":
            if not _percentile_filter(confidence):
                logger.info(
                    f"[Signal] 📊 Перцентильный фильтр: conf={confidence:.1%} "
                    f"< {CONFIDENCE_PERCENTILE}th перцентиль → HOLD"
                )
                signal = "HOLD"

        # 7. ADX фильтр
        adx_1h = float(last_1h.get('ADX', 25.0))
        if REGIME_FILTER_ENABLED and signal != "HOLD" and adx_1h < REGIME_ADX_THRESHOLD:
            logger.info(f"[Signal] 🔕 ADX={adx_1h:.1f} < {REGIME_ADX_THRESHOLD} → HOLD")
            signal = "HOLD"

        # 8. Multi-TF 4H фильтр
        mtf_confirmed = True
        if MTF_ENABLED and df4h_feats is not None and signal != "HOLD":
            last_4h      = df4h_feats.iloc[-1]
            rsi_4h       = float(last_4h.get('RSI_14_4h', 50))
            ema_ratio_4h = float(last_4h.get('EMA_ratio_4h', 1.0))

            if signal == "BUY":
                mtf_ok = (ema_ratio_4h > 0.995) and (rsi_4h > 40)
            else:
                mtf_ok = (ema_ratio_4h < 1.005) and (rsi_4h < 60)

            if not mtf_ok:
                logger.info(
                    f"[Signal] 🔕 4H фильтр: RSI={rsi_4h:.1f} "
                    f"EMA_ratio={ema_ratio_4h:.4f} → HOLD"
                )
                signal        = "HOLD"
                mtf_confirmed = False

        # 9. BTC macro-фильтр
        btc_change  = 0.0
        btc_blocked = False
        if BTC_FILTER_ENABLED and signal == "BUY":
            btc_change = get_btc_4h_change(exchange)
            if btc_change < BTC_CORRELATION_THRESH:
                logger.info(f"[Signal] 🔕 BTC 4H={btc_change:+.2f}% → блок BUY")
                signal      = "HOLD"
                btc_blocked = True

        # 10. Мета-данные
        cur_price   = float(last_1h['Close'])
        current_atr = float(last_1h.get('ATR', 0.0))
        change_24h  = float(last_1h.get('Return_24h', 0.0))
        volume      = float(last_1h.get('Volume', 0.0))

        logger.info(
            f"[Signal] {signal} | "
            f"p_buy={p_buy:.1%} p_sell={p_sell:.1%} | "
            f"Conf={confidence:.1%} | Models={models_used} | "
            f"ADX={adx_1h:.1f} | 4H={'✅' if mtf_confirmed else '❌'} | "
            f"BTC={btc_change:+.2f}% | Price=${cur_price:.4f}"
        )

        return {
            "signal":        signal,
            "confidence":    confidence,
            "p_buy":         p_buy,
            "p_sell":        p_sell,
            "price":         cur_price,
            "atr":           current_atr,
            "change_24h":    change_24h,
            "volume":        volume,
            "adx":           adx_1h,
            "models_used":   models_used,
            "mtf_confirmed": mtf_confirmed,
            "btc_change_4h": btc_change,
            "btc_blocked":   btc_blocked,
            # Совместимость с app.py
            "xgb_signal":    signal,
            "lgbm_signal":   signal,
        }

    except Exception as e:
        logger.error(f"[Signal] Ошибка: {e}", exc_info=True)
        return None