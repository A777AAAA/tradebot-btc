"""
live_signal.py v4.0 — Многомодельный сигнал с Multi-TF фильтрацией
v4.0 изменения:
  - Ансамбль XGBoost + LightGBM (консенсус обеих моделей)
  - 4H timeframe фильтр: сигнал открывается только при совпадении тренда
  - BTC macro-фильтр: блокируем BUY при падении BTC > 3% на 4H
  - Market Regime фильтр: ADX < 20 = флэт = HOLD
  - 40+ признаков (динамическая загрузка feature_cols из JSON)
  - Уверенность ансамбля = avg(XGB_prob, LGBM_prob)
"""

import ccxt
import pandas as pd
import numpy as np
import logging
import joblib
import json
import os
import time

from config import (
    MODEL_PATH, MODEL_PATH_LGBM,
    FEATURE_COLS, FEATURE_COLS_LEGACY,
    MTF_ENABLED, BTC_FILTER_ENABLED, BTC_CORRELATION_THRESH,
    REGIME_FILTER_ENABLED, REGIME_ADX_THRESHOLD
)

logger = logging.getLogger(__name__)

OKX_CONFIG = {'options': {'defaultType': 'spot'}, 'timeout': 30000}


# ─────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────
def _get_exchange():
    return ccxt.okx(OKX_CONFIG)


def _to_df(ohlcv: list) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=['ts', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    return df.astype(float)


# ─────────────────────────────────────────────
# Загрузка фичей из сохранённого JSON
# ─────────────────────────────────────────────
def load_feature_cols() -> list:
    features_path = MODEL_PATH.replace('.pkl', '_features.json')
    if os.path.exists(features_path):
        with open(features_path) as f:
            cols = json.load(f)
        logger.info(f"[Signal] Загружено {len(cols)} фичей из {features_path}")
        return cols
    # Фоллбэк
    logger.warning("[Signal] features.json не найден — используем FEATURE_COLS из config")
    return FEATURE_COLS


# ─────────────────────────────────────────────
# 1H индикаторы
# ─────────────────────────────────────────────
def calc_indicators_1h(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
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
    atr14        = tr.ewm(com=13, min_periods=14).mean()
    atr50        = tr.ewm(com=49, min_periods=50).mean()
    d['ATR']     = atr14
    d['ATR_pct'] = (atr14 / (close + 1e-9)) * 100
    d['ATR_norm'] = atr14 / (close + 1e-9)
    d['ATR_ratio'] = atr14 / (atr50 + 1e-9)

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

    obv        = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    obv_sma20  = obv.rolling(20).mean()
    d['OBV_norm'] = (obv - obv_sma20) / (obv.rolling(20).std() + 1e-9)

    tp       = (high + low + close) / 3
    mf       = tp * vol
    pos_mf   = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_mf   = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
    d['MFI_14'] = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-9)))

    rsi14     = d['RSI_14']
    stoch_min = rsi14.rolling(14).min()
    stoch_max = rsi14.rolling(14).max()
    stoch_k   = (rsi14 - stoch_min) / (stoch_max - stoch_min + 1e-9) * 100
    d['StochRSI_K'] = stoch_k
    d['StochRSI_D'] = stoch_k.rolling(3).mean()

    hw14 = high.rolling(14).max()
    lw14 = low.rolling(14).min()
    d['WilliamsR'] = (hw14 - close) / (hw14 - lw14 + 1e-9) * -100

    d['ZScore_20'] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-9)
    d['ZScore_50'] = (close - close.rolling(50).mean()) / (close.rolling(50).std() + 1e-9)

    up   = high.diff()
    down = -low.diff()
    pdm  = up.where((up > down)   & (up > 0),   0)
    mdm  = down.where((down > up) & (down > 0), 0)
    pdi  = 100 * (pdm.ewm(alpha=1 / 14).mean() / (atr14 + 1e-9))
    mdi  = 100 * (mdm.ewm(alpha=1 / 14).mean() / (atr14 + 1e-9))
    dx   = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    d['ADX'] = dx.ewm(alpha=1 / 14).mean()

    d['Body_pct']   = (close - d['Open']).abs() / (d['Open'] + 1e-9) * 100
    d['Upper_wick'] = (high - d[['Close', 'Open']].max(axis=1)) / (d['Open'] + 1e-9) * 100
    d['Lower_wick'] = (d[['Close', 'Open']].min(axis=1) - low) / (d['Open'] + 1e-9) * 100
    body_range      = (high - low + 1e-9)
    d['Doji']       = (d['Body_pct'] / body_range < 0.1).astype(int)

    d['Momentum_10'] = close - close.shift(10)
    d['ROC_10']      = close.pct_change(10) * 100

    for h in [1, 4, 12, 24]:
        d[f'Return_{h}h'] = close.pct_change(h) * 100

    return d.dropna()


# ─────────────────────────────────────────────
# 4H индикаторы (для фильтра и фичей модели)
# ─────────────────────────────────────────────
def calc_indicators_4h(df4h: pd.DataFrame) -> pd.DataFrame:
    d = df4h.copy()
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
    pdi  = 100 * (pdm.ewm(alpha=1 / 14).mean() / (atr14 + 1e-9))
    mdi  = 100 * (mdm.ewm(alpha=1 / 14).mean() / (atr14 + 1e-9))
    dx   = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    d['ADX_4h'] = dx.ewm(alpha=1 / 14).mean()

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    d['BB_pos_4h'] = (close - (sma20 - 2 * std20)) / (4 * std20 + 1e-9)

    return d.dropna()


# ─────────────────────────────────────────────
# BTC macro-фильтр
# ─────────────────────────────────────────────
def get_btc_4h_change(exchange) -> float:
    """Возвращает 4H изменение BTC в % (последняя закрытая свеча)."""
    try:
        ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe='4h', limit=5)
        if not ohlcv or len(ohlcv) < 2:
            return 0.0
        prev_close = float(ohlcv[-2][4])
        last_close = float(ohlcv[-1][4])
        return (last_close - prev_close) / prev_close * 100
    except Exception as e:
        logger.warning(f"[Signal] BTC фильтр — ошибка: {e}")
        return 0.0


# ─────────────────────────────────────────────
# Получение сигнала ансамбля
# ─────────────────────────────────────────────
def get_live_signal(symbol: str = "TON/USDT") -> dict | None:
    try:
        # 1. Загрузка моделей
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"[Signal] XGBoost модель не найдена: {MODEL_PATH}")
            return None

        xgb_model = joblib.load(MODEL_PATH)

        lgbm_model = None
        if os.path.exists(MODEL_PATH_LGBM):
            try:
                lgbm_model = joblib.load(MODEL_PATH_LGBM)
            except Exception:
                pass

        # 2. Загружаем список фичей
        feature_cols = load_feature_cols()

        # 3. Загружаем 1H данные (200 свечей = достаточно для всех индикаторов)
        exchange = _get_exchange()
        ohlcv_1h = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=200)

        if not ohlcv_1h or len(ohlcv_1h) < 100:
            logger.warning("[Signal] Мало 1H данных")
            return None

        df1h = _to_df(ohlcv_1h)
        df1h_feats = calc_indicators_1h(df1h)

        if df1h_feats.empty:
            return None

        # 4. Загружаем 4H данные
        df4h_feats = None
        if MTF_ENABLED:
            try:
                ohlcv_4h = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=100)
                if ohlcv_4h and len(ohlcv_4h) >= 50:
                    df4h = _to_df(ohlcv_4h)
                    df4h_feats = calc_indicators_4h(df4h)
            except Exception as e:
                logger.warning(f"[Signal] 4H данные — ошибка: {e}")

        # 5. Собираем признаки: 1H + 4H (последняя строка)
        last_1h  = df1h_feats.iloc[-1]
        row_data = {}

        for col in feature_cols:
            if col in df1h_feats.columns:
                row_data[col] = float(last_1h[col])
            elif col.endswith('_4h') or col.endswith('_4h_tf'):
                # 4H признак
                if df4h_feats is not None and col in df4h_feats.columns:
                    row_data[col] = float(df4h_feats.iloc[-1][col])
                else:
                    row_data[col] = 0.0  # fallback
            else:
                row_data[col] = 0.0

        X = np.array([[row_data[c] for c in feature_cols]], dtype=np.float32)

        # 6. XGBoost предсказание
        xgb_pred  = int(xgb_model.predict(X)[0])
        xgb_probs = xgb_model.predict_proba(X)[0]

        # 7. LightGBM предсказание (если доступен)
        lgbm_pred  = None
        lgbm_probs = None
        if lgbm_model is not None:
            try:
                lgbm_pred  = int(lgbm_model.predict(X)[0])
                lgbm_probs = lgbm_model.predict_proba(X)[0]
            except Exception as e:
                logger.warning(f"[Signal] LGBM предсказание — ошибка: {e}")

        # 8. Ансамбль: усредняем вероятности
        if lgbm_probs is not None:
            ensemble_probs = (xgb_probs + lgbm_probs) / 2.0
            models_used    = "XGB+LGBM"
        else:
            ensemble_probs = xgb_probs
            models_used    = "XGB"

        pred_class = int(np.argmax(ensemble_probs))
        confidence = float(np.max(ensemble_probs))

        signal_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        signal     = signal_map.get(pred_class, "HOLD")

        # 9. Консенсус-фильтр: оба должны согласиться
        if lgbm_pred is not None and signal != "HOLD":
            lgbm_signal = signal_map.get(lgbm_pred, "HOLD")
            xgb_signal  = signal_map.get(xgb_pred, "HOLD")
            if xgb_signal != lgbm_signal:
                logger.info(
                    f"[Signal] ⚠️ Нет консенсуса: XGB={xgb_signal} LGBM={lgbm_signal} → HOLD"
                )
                signal     = "HOLD"
                confidence = float(ensemble_probs[0])  # p_hold

        # 10. Market Regime фильтр
        adx_1h = float(last_1h.get('ADX', 25.0))
        if REGIME_FILTER_ENABLED and signal != "HOLD":
            if adx_1h < REGIME_ADX_THRESHOLD:
                logger.info(
                    f"[Signal] 🔕 Regime filter: ADX={adx_1h:.1f} < {REGIME_ADX_THRESHOLD} → HOLD"
                )
                signal = "HOLD"

        # 11. Multi-TF 4H фильтр
        mtf_confirmed   = True
        mtf_description = "N/A"

        if MTF_ENABLED and df4h_feats is not None and signal != "HOLD":
            last_4h      = df4h_feats.iloc[-1]
            rsi_4h       = float(last_4h.get('RSI_14_4h', 50))
            ema_ratio_4h = float(last_4h.get('EMA_ratio_4h', 1.0))
            adx_4h       = float(last_4h.get('ADX_4h', 25))

            if signal == "BUY":
                # Подтверждение покупки: 4H в восходящем тренде
                mtf_ok = (ema_ratio_4h > 1.0) and (rsi_4h > 45)
            else:  # SELL
                # Подтверждение продажи: 4H в нисходящем тренде
                mtf_ok = (ema_ratio_4h < 1.0) and (rsi_4h < 55)

            mtf_description = (
                f"RSI_4h={rsi_4h:.1f} "
                f"EMA_ratio_4h={ema_ratio_4h:.4f} "
                f"ADX_4h={adx_4h:.1f}"
            )

            if not mtf_ok:
                logger.info(
                    f"[Signal] 🔕 4H фильтр отклонил {signal}: {mtf_description}"
                )
                signal      = "HOLD"
                mtf_confirmed = False

        # 12. BTC macro-фильтр
        btc_change = 0.0
        btc_blocked = False

        if BTC_FILTER_ENABLED and signal == "BUY":
            btc_change = get_btc_4h_change(exchange)
            if btc_change < BTC_CORRELATION_THRESH:
                logger.info(
                    f"[Signal] 🔕 BTC macro-фильтр: BTC 4H = {btc_change:+.2f}% → блок BUY"
                )
                signal     = "HOLD"
                btc_blocked = True

        # 13. Итоговые данные
        cur_price   = float(last_1h['Close'])
        current_atr = float(last_1h.get('ATR', 0.0))
        change_24h  = float(last_1h.get('Return_24h', 0.0))
        volume      = float(last_1h.get('Volume', 0.0))

        p_hold = float(ensemble_probs[0])
        p_buy  = float(ensemble_probs[1])
        p_sell = float(ensemble_probs[2])

        logger.info(
            f"[Signal] {signal} | "
            f"HOLD={p_hold:.1%} BUY={p_buy:.1%} SELL={p_sell:.1%} | "
            f"Conf={confidence:.1%} | Models={models_used} | "
            f"ADX={adx_1h:.1f} | 4H={'✅' if mtf_confirmed else '❌'} | "
            f"BTC_4h={btc_change:+.2f}% | "
            f"Price=${cur_price:.4f}"
        )

        return {
            "signal":         signal,
            "confidence":     confidence,
            "price":          cur_price,
            "p_hold":         p_hold,
            "p_buy":          p_buy,
            "p_sell":         p_sell,
            "change_24h":     change_24h,
            "atr":            current_atr,
            "volume":         volume,
            "adx":            adx_1h,
            # Мета-информация фильтров
            "models_used":    models_used,
            "mtf_confirmed":  mtf_confirmed,
            "mtf_info":       mtf_description,
            "btc_change_4h":  btc_change,
            "btc_blocked":    btc_blocked,
            "xgb_signal":     signal_map.get(xgb_pred, "HOLD"),
            "lgbm_signal":    signal_map.get(lgbm_pred, "HOLD") if lgbm_pred is not None else "N/A",
        }

    except Exception as e:
        logger.error(f"[Signal] Ошибка: {e}", exc_info=True)
        return None