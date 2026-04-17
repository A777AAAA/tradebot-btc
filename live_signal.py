"""
live_signal.py v8.1 — Реальный Order Book OFI + Regime-Switching + Калиброванные модели
ИЗМЕНЕНИЯ v8.0 vs v7.0:
  - РЕАЛЬНЫЙ ORDER BOOK OFI: OKX API /api/v5/market/books
    Вместо OHLCV-прокси — настоящий дисбаланс bid/ask топ-10 уровней.
    Это один из сильнейших краткосрочных предикторов (используется в HFT).
  - REGIME-SWITCHING ПОРОГИ:
    Тренд (Hurst > 0.6):      порог снижен на 5% (momentum работает)
    Mean-Rev (Hurst < 0.4):   порог снижен на 5% (осцилляторы работают)
    Случайный (0.4-0.6):      порог повышен на 10% (плохая предсказуемость)
    Это ключевое отличие топовых систем от любительских ботов.
  - КАЛИБРОВАННЫЕ МОДЕЛИ: загружает calibrated_model_buy/sell.pkl (v8.0)
    Если доступны — использует их вместо raw XGB.
    Это делает пороги 0.58/0.62/0.70/0.75 реалистичными.
  - Улучшенный composite score: взвешенное голосование с учётом режима
  - Все v7.0 улучшения сохранены
"""

import requests
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

import importlib

def _get_cfg(param, default):
    try:
        import config as _cfg
        importlib.reload(_cfg)
        return getattr(_cfg, param, default)
    except Exception:
        return default

logger = logging.getLogger(__name__)

OKX_REST = 'https://www.okx.com'

META_MODEL_BUY_PATH   = "meta_model_buy.pkl"
META_MODEL_SELL_PATH  = "meta_model_sell.pkl"
STACK_MODEL_BUY_PATH  = "stack_model_buy.pkl"
STACK_MODEL_SELL_PATH = "stack_model_sell.pkl"
CALIB_MODEL_BUY_PATH  = "calibrated_model_buy.pkl"   # v8.0
CALIB_MODEL_SELL_PATH = "calibrated_model_sell.pkl"   # v8.0

_confidence_history: list = []
_HISTORY_MAX = 48

_funding_cache: dict = {"rate": 0.0, "oi_change": 0.0, "bias": "neutral", "ts": 0.0}
_FUNDING_TTL = 300

# v8.0: кэш Order Book OFI (обновляем не чаще раза в 2 минуты)
_ob_ofi_cache: dict = {"ofi": 0.0, "bid_ask_ratio": 1.0, "ts": 0.0}
_OB_OFI_TTL = 120


# ===============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ===============================================================

def _okx_get(url: str, retries: int = 3) -> dict:
    """GET запрос к OKX с retry + exponential backoff при 429."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=15, proxies={"http": "http://smmlbrex:gblito9kdke7@31.59.20.176:6754", "https": "http://smmlbrex:gblito9kdke7@31.59.20.176:6754"})
            if r.status_code == 429:
                wait = 2 ** attempt * 5
                logger.warning(f"[Signal] Rate limit 429 — ждём {wait}с")
                time.sleep(wait)
                continue
            if r.status_code != 200:
                logger.error(f"[Signal] HTTP {r.status_code}: {url}")
                return {}
            return r.json()
        except requests.exceptions.Timeout:
            logger.warning(f"[Signal] Timeout (попытка {attempt+1})")
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"[Signal] Ошибка запроса: {e}")
            time.sleep(2 ** attempt)
    return {}


def _fetch_candles(inst_id: str, bar: str = "1H", limit: int = 250, after: str = None) -> list:
    """Получаем свечи через публичный REST OKX с retry."""
    url = f"{OKX_REST}/api/v5/market/candles?instId={inst_id}&bar={bar}&limit={limit}"
    if after:
        url += f"&after={after}"
    return _okx_get(url).get("data", [])


def _to_df_rest(data: list) -> pd.DataFrame:
    """Конвертируем REST ответ в DataFrame"""
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=['ts','Open','High','Low','Close','Volume','VolCcy','VolCcyQuote','Confirm'])
    df = df[['ts','Open','High','Low','Close','Volume']].copy()
    df['ts'] = pd.to_datetime(df['ts'].astype(float), unit='ms')
    df.set_index('ts', inplace=True)
    for col in ['Open','High','Low','Close','Volume']:
        df[col] = df[col].astype(float)
    return df.sort_index()


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
# v8.0 NEW: РЕАЛЬНЫЙ ORDER BOOK OFI
# ===============================================================

def get_orderbook_ofi(symbol: str = "BTC/USDT", depth: int = 10) -> dict:
    """
    Реальный Order Flow Imbalance из стакана OKX.

    OFI = (Bid_Volume_top10 - Ask_Volume_top10) / (Bid_Volume_top10 + Ask_Volume_top10)
    OFI > 0: давление покупателей (bullish)
    OFI < 0: давление продавцов (bearish)
    |OFI| > 0.3: сильный дисбаланс

    Дополнительно считаем:
    - bid_ask_ratio: соотношение суммарных объёмов
    - spread_pct: спред как % от цены
    - weighted_mid: взвешенная средняя цена

    Это то, что используют HFT-системы и профессиональные маркет-мейкеры.
    """
    global _ob_ofi_cache

    now = time.time()
    if now - _ob_ofi_cache["ts"] < _OB_OFI_TTL:
        return {
            "ob_ofi":          _ob_ofi_cache["ofi"],
            "bid_ask_ratio":   _ob_ofi_cache["bid_ask_ratio"],
            "spread_pct":      _ob_ofi_cache.get("spread_pct", 0.0),
            "weighted_mid":    _ob_ofi_cache.get("weighted_mid", 0.0),
            "ob_source":       "cache",
        }

    result = {
        "ob_ofi":        0.0,
        "bid_ask_ratio": 1.0,
        "spread_pct":    0.0,
        "weighted_mid":  0.0,
        "ob_source":     "fallback",
    }

    try:
        # OKX REST API для стакана
        import requests as req
        inst_id = symbol.replace("/", "-")
        url = f"https://www.okx.com/api/v5/market/books?instId={inst_id}&sz={depth}"
        r = req.get(url, timeout=5)
        data = r.json().get("data", [])

        if not data or not data[0]:
            logger.debug("[Signal] Order Book: нет данных")
            _ob_ofi_cache["ts"] = now
            return result

        book = data[0]
        bids = book.get("bids", [])[:depth]   # [[price, qty, ?, ?], ...]
        asks = book.get("asks", [])[:depth]

        if not bids or not asks:
            _ob_ofi_cache["ts"] = now
            return result

        bid_prices = [float(b[0]) for b in bids]
        bid_vols   = [float(b[1]) for b in bids]
        ask_prices = [float(a[0]) for a in asks]
        ask_vols   = [float(a[1]) for a in asks]

        total_bid = sum(bid_vols)
        total_ask = sum(ask_vols)
        total     = total_bid + total_ask

        if total == 0:
            _ob_ofi_cache["ts"] = now
            return result

        # Нормализованный OFI [-1, +1]
        ofi = (total_bid - total_ask) / total

        # Bid/Ask ratio
        bid_ask_ratio = total_bid / (total_ask + 1e-9)

        # Спред
        best_bid = bid_prices[0]
        best_ask = ask_prices[0]
        mid_price = (best_bid + best_ask) / 2
        spread_pct = (best_ask - best_bid) / (mid_price + 1e-9) * 100

        # Взвешенная средняя цена (Weighted Mid Price)
        # Цена смещается к стороне с бо́льшим объёмом
        weighted_mid = (best_bid * total_ask + best_ask * total_bid) / (total + 1e-9)

        result = {
            "ob_ofi":        round(float(ofi), 4),
            "bid_ask_ratio": round(float(bid_ask_ratio), 3),
            "spread_pct":    round(float(spread_pct), 4),
            "weighted_mid":  round(float(weighted_mid), 6),
            "ob_source":     "live",
        }

        _ob_ofi_cache.update({
            "ofi":           ofi,
            "bid_ask_ratio": bid_ask_ratio,
            "spread_pct":    spread_pct,
            "weighted_mid":  weighted_mid,
            "ts":            now,
        })

        logger.info(
            f"[OrderBook] OFI={ofi:+.3f} "
            f"Bid/Ask={bid_ask_ratio:.2f} "
            f"Spread={spread_pct:.3f}% "
            f"WMid={weighted_mid:.4f}"
        )

    except Exception as e:
        logger.debug(f"[Signal] Order Book ошибка: {e}")
        _ob_ofi_cache["ts"] = now

    return result


# ===============================================================
# ORDER FLOW — Funding Rate + Open Interest
# ===============================================================

def get_funding_data(symbol_spot: str = "BTC/USDT") -> dict:
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
        # OKX REST: funding rate
        inst_id_swap = symbol_spot.replace("/", "-") + "-SWAP"
        fr_url = f"{OKX_REST}/api/v5/public/funding-rate?instId={inst_id_swap}"
        fr_d_raw = _okx_get(fr_url)
        fr_d   = fr_d_raw.get("data", [{}])
        funding_rate = float(fr_d[0].get("fundingRate", 0.0)) if fr_d else 0.0

        oi_change_pct = 0.0
        try:
            oi_url = f"{OKX_REST}/api/v5/public/open-interest?instId={inst_id_swap}&limit=3"
            oi_d_raw = _okx_get(oi_url)
            oi_d   = oi_d_raw.get("data", [])
            if len(oi_d) >= 2:
                oi_now  = float(oi_d[0].get("oi", 1))
                oi_prev = float(oi_d[1].get("oi", 1))
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
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ПРИЗНАКОВ
# ===============================================================

def _calc_hurst_window(series: np.ndarray, lags_range=range(2, 21)) -> float:
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

    # ── ПРОФЕССИОНАЛЬНЫЕ ПРИЗНАКИ v7.0+ ──

    d['Hurst'] = close.rolling(100, min_periods=50).apply(
        lambda x: _calc_hurst_window(x), raw=True
    )

    tp_series = (high + low + close) / 3
    vwap_20   = (tp_series * vol).rolling(20).sum() / vol.rolling(20).sum()
    vwap_50   = (tp_series * vol).rolling(50).sum() / vol.rolling(50).sum()
    d['VWAP_dev_20']     = (close - vwap_20) / (vwap_20 + 1e-9) * 100
    d['VWAP_dev_50']     = (close - vwap_50) / (vwap_50 + 1e-9) * 100
    bull_vol = vol.where(close > vwap_20, 0).rolling(10).sum()
    d['VWAP_bull_ratio'] = bull_vol / (vol.rolling(10).sum() + 1e-9)

    log_ret     = np.log(close / close.shift(1))
    d['RV_20']  = np.sqrt((log_ret**2).rolling(20).sum() / 20 * 8760) * 100
    d['RV_50']  = np.sqrt((log_ret**2).rolling(50).sum() / 50 * 8760) * 100
    d['RV_ratio'] = d['RV_20'] / (d['RV_50'] + 1e-9)

    bull_frac = (close - low) / (high - low + 1e-9)
    bear_frac = (high - close) / (high - low + 1e-9)
    ofi_raw   = (bull_frac * vol - bear_frac * vol).rolling(10).sum()
    d['OFI']  = ofi_raw / (vol.rolling(10).sum() + 1e-9)

    ret_1h       = close.pct_change(1)
    d['Price_accel'] = ret_1h - ret_1h.shift(1)

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

    d['Hurst_4h'] = close.rolling(60, min_periods=30).apply(
        lambda x: _calc_hurst_window(x, range(2, 15)), raw=True
    )

    return d.dropna()


# ===============================================================
# BTC MACRO FILTER
# ===============================================================

def get_btc_4h_change(exchange=None) -> float:
    try:
        data_btc = _fetch_candles("BTC-USDT", "4H", 5)
        ohlcv = [[int(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4]), float(d[5])] for d in data_btc] if data_btc else []
        if not ohlcv or len(ohlcv) < 2:
            return 0.0
        return (float(ohlcv[-1][4]) - float(ohlcv[-2][4])) / float(ohlcv[-2][4]) * 100
    except Exception as e:
        logger.warning(f"[Signal] BTC: {e}")
        return 0.0


# ===============================================================
# v8.0 NEW: REGIME-SWITCHING ПОРОГИ
# ===============================================================

def get_regime_adjusted_threshold(
    hurst: float,
    adx: float,
    atr_ratio: float,
    bb_width: float,
    base_threshold: float
) -> tuple:
    """
    Адаптивный порог уверенности в зависимости от режима рынка.

    Логика:
    - TREND (Hurst > 0.6, ADX > 25): momentum работает хорошо
      → снижаем порог на 5-8% (больше сигналов при высоком качестве)
    - MEAN_REV (Hurst < 0.4, ADX < 20): осцилляторы работают
      → снижаем порог на 3-5%
    - RANDOM (0.4 ≤ Hurst ≤ 0.6): рынок непредсказуем
      → повышаем порог на 8-12% (меньше ложных сигналов)
    - VOLATILE (ATR_ratio > 1.8): высокая волатильность
      → повышаем порог на 5% (слипеж больше)

    Это имитирует то, как Renaissance/Two Sigma переключаются между
    стратегиями в зависимости от рыночного режима.
    """
    # Определяем режим
    if hurst > 0.62 and adx > 25:
        regime    = "TREND"
        threshold = base_threshold * 0.93   # -7%
        note      = f"TREND Hurst={hurst:.3f} ADX={adx:.1f} → порог снижен"
    elif hurst < 0.40 and adx < 22:
        regime    = "MEAN_REV"
        threshold = base_threshold * 0.96   # -4%
        note      = f"MEAN_REV Hurst={hurst:.3f} ADX={adx:.1f} → порог снижен"
    elif 0.43 <= hurst <= 0.57:
        regime    = "RANDOM"
        threshold = base_threshold * 1.10   # +10%
        note      = f"RANDOM Hurst={hurst:.3f} → порог повышен"
    elif atr_ratio > 1.8:
        regime    = "VOLATILE"
        threshold = base_threshold * 1.06   # +6%
        note      = f"VOLATILE ATR_ratio={atr_ratio:.2f} → порог повышен"
    else:
        regime    = "NEUTRAL"
        threshold = base_threshold * 1.02   # +2%
        note      = f"NEUTRAL Hurst={hurst:.3f} ADX={adx:.1f}"

    threshold = min(max(threshold, 0.50), 0.90)  # ограничиваем [0.50, 0.90]

    return regime, round(threshold, 4), note


def detect_market_regime(adx: float, atr_ratio: float, bb_width: float) -> dict:
    """Базовый детектор режима (для совместимости с app.py)."""
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
        ('meta_buy',    META_MODEL_BUY_PATH),
        ('meta_sell',   META_MODEL_SELL_PATH),
        ('stack_buy',   STACK_MODEL_BUY_PATH),
        ('stack_sell',  STACK_MODEL_SELL_PATH),
        ('calib_buy',   CALIB_MODEL_BUY_PATH),   # v8.0
        ('calib_sell',  CALIB_MODEL_SELL_PATH),   # v8.0
    ]:
        if os.path.exists(path):
            try:
                models[key] = joblib.load(path)
                logger.info(f"[Signal] Загружена модель: {path}")
            except Exception as e:
                logger.warning(f"[Signal] {path}: {e}")

    return models


# ===============================================================
# STACKING INFERENCE
# ===============================================================

def _apply_stacking(models: dict, p_xgb: float, p_lgbm: float, direction: str) -> float:
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
        return p_stacked
    except Exception as e:
        logger.warning(f"[Signal] Stack {direction} ошибка: {e}")
        return (p_xgb + p_lgbm) / 2.0


# ===============================================================
# v8.0: КАЛИБРОВАННЫЙ INFERENCE
# ===============================================================

def _get_calibrated_prob(models: dict, X: np.ndarray, direction: str,
                          p_raw: float) -> float:
    """
    Если доступна калиброванная модель — используем её вероятности.
    Калиброванные вероятности более точно отражают реальный win rate.
    """
    calib_key = f'calib_{direction}'
    if calib_key not in models:
        return p_raw

    try:
        p_calib = float(models[calib_key].predict_proba(X)[0][1])
        logger.debug(
            f"[Calib {direction.upper()}] raw={p_raw:.1%} → calibrated={p_calib:.1%}"
        )
        return p_calib
    except Exception as e:
        logger.warning(f"[Signal] Calibrated {direction} ошибка: {e}")
        return p_raw


# ===============================================================
# META-MODEL INFERENCE
# ===============================================================

def _apply_meta_filter(models: dict, X: np.ndarray, p_base: float,
                        model_key: str, signal_dir: str) -> tuple:
    if model_key not in models:
        return None, True

    try:
        X_meta  = np.hstack([X, np.array([[p_base]])])
        p_meta  = float(models[model_key].predict_proba(X_meta)[0][1])
        passed  = p_meta >= 0.50
        return p_meta, passed
    except Exception as e:
        logger.warning(f"[Meta] Ошибка {model_key}: {e}")
        return None, True


# ===============================================================
# FUNDING RATE КОРРЕКЦИЯ
# ===============================================================

def _apply_funding_correction(signal: str, confidence: float,
                               funding: dict, threshold: float) -> tuple:
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
# v8.0: OB OFI BOOST
# ===============================================================

def _apply_ob_ofi_boost(signal: str, confidence: float,
                          ob_data: dict) -> tuple:
    """
    Корректируем уверенность на основе реального Order Book OFI.

    OFI > 0.2 при BUY  → +5% boost (стакан подтверждает)
    OFI < -0.2 при BUY → -5% снижение (стакан против)
    И наоборот для SELL.
    """
    ob_ofi = ob_data.get("ob_ofi", 0.0)
    note   = ""

    if signal == "BUY":
        if ob_ofi > 0.2:
            confidence = min(confidence * 1.05, 0.99)
            note = f"OB_OFI={ob_ofi:+.3f} ↑ BUY boost"
        elif ob_ofi < -0.2:
            confidence = confidence * 0.95
            note = f"OB_OFI={ob_ofi:+.3f} ↓ BUY reduce"
    elif signal == "SELL":
        if ob_ofi < -0.2:
            confidence = min(confidence * 1.05, 0.99)
            note = f"OB_OFI={ob_ofi:+.3f} ↑ SELL boost"
        elif ob_ofi > 0.2:
            confidence = confidence * 0.95
            note = f"OB_OFI={ob_ofi:+.3f} ↓ SELL reduce"

    return confidence, note


# ===============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ===============================================================

def get_live_signal(symbol: str = "BTC/USDT") -> dict | None:
    """
    Полный pipeline v8.0:
      1. Загружаем модели (базовые + calibrated + stack + мета)
      2. Получаем 1H + 4H свечи + Funding Rate + Order Book OFI
      3. Считаем индикаторы
      4. Layer 1: XGB + LGBM -> p_xgb, p_lgbm
      4.5. Calibration: если есть calibrated_model → используем его
      4.7. Stacking -> p_buy/p_sell
      5. v8.0 Regime-Switching: адаптивный порог по Hurst + ADX
      6. Layer 2: Meta-model фильтр
      7. v8.0 OB OFI boost: корректировка по реальному стакану
      8. Фильтры: percentile → ADX → 4H MTF → BTC → Funding
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
        inst = symbol.replace("/", "-")
        data_1h = _fetch_candles(inst, "1H", 300)
        ohlcv_1h = [[int(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4]), float(d[5])] for d in data_1h] if data_1h else []

        if not ohlcv_1h or len(ohlcv_1h) < 150:
            logger.warning("[Signal] Мало 1H данных")
            return None

        df1h_feats = calc_indicators_1h(_to_df(ohlcv_1h))
        if df1h_feats.empty:
            return None

        df4h_feats = None
        if MTF_ENABLED:
            try:
                data_4h = _fetch_candles(inst, "4H", 100)
                ohlcv_4h = [[int(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4]), float(d[5])] for d in data_4h] if data_4h else []
                if ohlcv_4h and len(ohlcv_4h) >= 50:
                    df4h_feats = calc_indicators_4h(_to_df(ohlcv_4h))
            except Exception as e:
                logger.warning(f"[Signal] 4H ошибка: {e}")

        funding_data = get_funding_data(symbol)

        # v8.0: Реальный Order Book OFI
        ob_data = get_orderbook_ofi(symbol)

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

        # 4.5 v8.0: Калиброванные вероятности
        p_buy_cal  = _get_calibrated_prob(models, X, 'buy',  p_buy_xgb)
        p_sell_cal = _get_calibrated_prob(models, X, 'sell', p_sell_xgb)

        # 4.7. Stacking (если нет калибровки — используем stacking)
        if 'calib_buy' in models:
            p_buy  = p_buy_cal
        else:
            p_buy  = _apply_stacking(models, p_buy_xgb,  p_buy_lgbm,  'buy')

        if 'calib_sell' in models:
            p_sell = p_sell_cal
        else:
            p_sell = _apply_stacking(models, p_sell_xgb, p_sell_lgbm, 'sell')

        has_stack   = 'stack_buy'  in models or 'stack_sell'  in models
        has_calib   = 'calib_buy'  in models or 'calib_sell'  in models
        models_used = "+".join(filter(None, [
            "XGB"   if 'buy_xgb'   in models else "",
            "LGBM"  if 'buy_lgbm'  in models else "",
            "CALIB" if has_calib               else "",
            "STACK" if has_stack and not has_calib else "",
        ]))

        # 5. v8.0 Regime-Switching: адаптивный порог
        adx_1h    = float(last_1h.get('ADX',       25.0))
        atr_ratio = float(last_1h.get('ATR_ratio',  1.0))
        bb_width  = float(last_1h.get('BB_width',  0.05))
        rsi_14    = float(last_1h.get('RSI_14',    50.0))
        hurst     = float(last_1h.get('Hurst',      0.5))

        # v8.0: Regime-Switching пороги вместо простого множителя
        regime, effective_threshold, regime_note = get_regime_adjusted_threshold(
            hurst, adx_1h, atr_ratio, bb_width, MIN_CONFIDENCE
        )

        # Совместимость с app.py (использует regime_mult)
        legacy_regime = detect_market_regime(adx_1h, atr_ratio, bb_width)

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
            p_meta_buy, meta_ok = _apply_meta_filter(models, X, p_buy, 'meta_buy', 'BUY')
            if not meta_ok:
                logger.info(f"[Signal] Meta-BUY заблокировал: p_meta={p_meta_buy:.1%}")
                signal       = "HOLD"
                meta_blocked = True

        elif signal == "SELL" and 'meta_sell' in models:
            p_meta_sell, meta_ok = _apply_meta_filter(models, X, p_sell, 'meta_sell', 'SELL')
            if not meta_ok:
                logger.info(f"[Signal] Meta-SELL заблокировал: p_meta={p_meta_sell:.1%}")
                signal       = "HOLD"
                meta_blocked = True

        # v8.0: OB OFI boost (до фильтров)
        ob_note = ""
        if signal != "HOLD":
            confidence, ob_note = _apply_ob_ofi_boost(signal, confidence, ob_data)
            if ob_note:
                logger.info(f"[Signal] {ob_note}")

        # 8. Фильтр-цепочка
        filter_log = []

        if signal != "HOLD":
            if not _percentile_filter(confidence):
                filter_log.append(f"PERCENTILE_LOW_{confidence:.1%}")
                signal = "HOLD"

        if REGIME_FILTER_ENABLED and signal != "HOLD" and adx_1h < _get_cfg("REGIME_ADX_THRESHOLD", 18.0):
            filter_log.append(f"ADX={adx_1h:.1f}<{_get_cfg('REGIME_ADX_THRESHOLD', 18.0)}")
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
            btc_change = get_btc_4h_change()
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
            f"[Signal v8.0] {signal} | "
            f"p_buy={p_buy:.1%}(cal={p_buy_cal:.1%}) | "
            f"p_sell={p_sell:.1%}(cal={p_sell_cal:.1%}) | "
            f"Hurst={hurst:.3f} | Regime={regime} | thresh={effective_threshold:.1%} | "
            f"OB_OFI={ob_data.get('ob_ofi', 0):+.3f} | "
            f"meta={'OK' if not meta_blocked else 'BLOCK'} | "
            f"ADX={adx_1h:.1f} | 4H={'OK' if mtf_confirmed else 'NO'} | "
            f"BTC={btc_change:+.2f}% | filters={filter_log} | {elapsed}s"
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
            "p_buy_cal":     round(p_buy_cal,   4),  # v8.0
            "p_sell_cal":    round(p_sell_cal,  4),  # v8.0

            # Layer 2: мета-модель
            "p_meta":        round(p_meta, 4) if p_meta is not None else None,
            "meta_blocked":  meta_blocked,
            "models_used":   models_used,

            # v8.0: Order Book
            "ob_ofi":        ob_data.get("ob_ofi", 0.0),
            "bid_ask_ratio": ob_data.get("bid_ask_ratio", 1.0),
            "spread_pct":    ob_data.get("spread_pct", 0.0),
            "ob_note":       ob_note,

            # Hurst + Regime v8.0
            "hurst":         round(hurst, 3),
            "regime":        regime,
            "regime_note":   regime_note,
            "regime_mult":   legacy_regime["mult"],  # совместимость
            "eff_threshold": round(effective_threshold, 4),

            # Рыночный контекст
            "price":         cur_price,
            "atr":           current_atr,
            "change_24h":    change_24h,
            "volume":        volume,
            "adx":           round(adx_1h, 2),
            "rsi14":         round(rsi_14,  2),

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