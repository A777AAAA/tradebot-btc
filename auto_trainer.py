"""
auto_trainer.py v8.3 — OFI_5 + OFI_20 + OFI_ratio + Funding_rate + Funding_bias
ИЗМЕНЕНИЯ v8.3 vs v8.1:
  - OFI_5:  Order Flow Imbalance за 5 баров (краткосрочное давление)
  - OFI_20: Order Flow Imbalance за 20 баров (среднесрочное давление)
  - OFI_ratio: OFI_5 / OFI_20 — ускорение/замедление потока
  - Funding_rate: ставка финансирования (из config, заполняется нулями если недоступна)
  - Funding_bias: знак и величина funding (bullish/bearish)
  - HOLD класс в Triple Barrier (3-классовая задача)
  - Все v8.1 улучшения сохранены
"""

import os
import json
import joblib
import logging
import requests
import time
import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from xgboost import XGBClassifier
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

from config import (
    MODEL_PATH_BUY_XGB, MODEL_PATH_BUY_LGBM,
    MODEL_PATH_SELL_XGB, MODEL_PATH_SELL_LGBM,
    MODEL_FEATURES_PATH, STATS_FILE,
    FEATURE_COLS, FEATURE_COLS_LEGACY,
    TARGET_HORIZON, TARGET_THRESHOLD,
    WF_TRAIN_DAYS, WF_TEST_DAYS, WF_STEP_DAYS,
    ATR_SL_MULT, ATR_TP_MULT,
)

logger = logging.getLogger(__name__)

META_MODEL_BUY_PATH     = "meta_model_buy.pkl"
META_MODEL_SELL_PATH    = "meta_model_sell.pkl"
STACK_MODEL_BUY_PATH    = "stack_model_buy.pkl"
STACK_MODEL_SELL_PATH   = "stack_model_sell.pkl"
CALIB_MODEL_BUY_PATH    = "calibrated_model_buy.pkl"
CALIB_MODEL_SELL_PATH   = "calibrated_model_sell.pkl"
FEATURE_IMPORTANCE_PATH = "feature_importance.json"

FEATURE_IMPORTANCE_THRESHOLD = 0.005

BARS_1H = 8000
BARS_4H = 4000

# ─────────────────────────────────────────────
# Загрузка OHLCV
# ─────────────────────────────────────────────
def fetch_ohlcv(symbol: str = "BTC-USDT", bar: str = "1H", bars: int = 8000) -> pd.DataFrame:
    from okx_client import get_candles_multi, candles_to_df
    try:
        raw = get_candles_multi(symbol, bar, bars)
        if not raw:
            logger.error(f"[Trainer] Нет данных {bar}")
            return pd.DataFrame()
        df = candles_to_df(raw)
        logger.info(f"[Trainer] ✅ Загружено {len(df)} свечей ({bar})")
        return df
    except Exception as e:
        logger.error(f"[Trainer] Ошибка загрузки {bar}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# Funding Rate (OKX, с fallback на нули)
# ─────────────────────────────────────────────
def fetch_funding_rate(symbol: str = "BTC-USDT-SWAP", limit: int = 100) -> pd.DataFrame:
    """
    Загружает историю funding rate с OKX.
    Если недоступно (блокировка Aeza) — возвращает пустой DF,
    признаки заполнятся нулями.
    """
    try:
        url = f"https://www.okx.com/api/v5/public/funding-rate-history"
        params = {"instId": symbol, "limit": limit}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("code") != "0" or not data.get("data"):
            return pd.DataFrame()
        rows = []
        for item in data["data"]:
            ts = int(item["fundingTime"]) // 1000
            rate = float(item["fundingRate"])
            rows.append({"ts": ts, "funding_rate": rate})
        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        df = df.set_index("datetime").sort_index()
        return df[["funding_rate"]]
    except Exception as e:
        logger.warning(f"[Trainer] Funding rate недоступен: {e}")
        return pd.DataFrame()


def merge_funding(df1h: pd.DataFrame, symbol: str = "BTC-USDT-SWAP") -> pd.DataFrame:
    """
    Добавляет Funding_rate и Funding_bias в 1H датафрейм.
    Funding выходит раз в 8 часов — forward fill на все бары.
    Если данные недоступны — заполняет нулями.
    """
    d = df1h.copy()
    funding_df = fetch_funding_rate(symbol)

    if funding_df.empty:
        logger.warning("[Trainer] Funding: заполняем нулями")
        d["Funding_rate"] = 0.0
        d["Funding_bias"] = 0.0
    else:
        # Reindex с forward fill
        funding_reindexed = funding_df.reindex(d.index, method="ffill")
        d["Funding_rate"] = funding_reindexed["funding_rate"].fillna(0.0)
        # Funding_bias: нормализованный знак (>0 = лонги платят шортам = перегрев)
        d["Funding_bias"] = np.sign(d["Funding_rate"]) * np.log1p(d["Funding_rate"].abs() * 10000)
        logger.info(f"[Trainer] Funding: {len(funding_df)} записей, mean={d['Funding_rate'].mean():.6f}")

    return d


# ─────────────────────────────────────────────
# Hurst, VWAP, RV
# ─────────────────────────────────────────────
def calc_hurst_exponent(ts: pd.Series, lags_range: range = range(2, 21)) -> pd.Series:
    def hurst_single(series):
        if len(series) < 20:
            return 0.5
        try:
            lags = list(lags_range)
            tau  = [max(np.std(np.subtract(series[lag:], series[:-lag])), 1e-9) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return max(0.0, min(1.0, poly[0]))
        except Exception:
            return 0.5
    return ts.rolling(window=100, min_periods=50).apply(lambda x: hurst_single(x), raw=True)


def calc_vwap_features(df: pd.DataFrame) -> pd.DataFrame:
    d     = df.copy()
    close = d['Close']
    high  = d['High']
    low   = d['Low']
    vol   = d['Volume']
    tp    = (high + low + close) / 3
    vwap_20 = (tp * vol).rolling(20).sum() / vol.rolling(20).sum()
    d['VWAP_dev_20'] = (close - vwap_20) / (vwap_20 + 1e-9) * 100
    vwap_50 = (tp * vol).rolling(50).sum() / vol.rolling(50).sum()
    d['VWAP_dev_50'] = (close - vwap_50) / (vwap_50 + 1e-9) * 100
    bull_vol  = vol.where(close > vwap_20, 0).rolling(10).sum()
    total_vol = vol.rolling(10).sum() + 1e-9
    d['VWAP_bull_ratio'] = bull_vol / total_vol
    return d


def calc_realized_volatility(close: pd.Series) -> pd.DataFrame:
    log_ret = np.log(close / close.shift(1))
    result  = pd.DataFrame(index=close.index)
    result['RV_20']    = np.sqrt((log_ret**2).rolling(20).sum() / 20) * np.sqrt(8760) * 100
    result['RV_50']    = np.sqrt((log_ret**2).rolling(50).sum() / 50) * np.sqrt(8760) * 100
    result['RV_ratio'] = result['RV_20'] / (result['RV_50'] + 1e-9)
    return result


# ─────────────────────────────────────────────
# v8.3: OFI расширенный (5, 20, ratio)
# ─────────────────────────────────────────────
def calc_ofi_extended(df: pd.DataFrame) -> pd.DataFrame:
    """
    OFI_5:   краткосрочное давление (5 баров) — быстрый сигнал
    OFI_20:  среднесрочное давление (20 баров) — фильтр тренда
    OFI_ratio: OFI_5 / OFI_20 — ускорение потока (импульс)
    OFI (базовый, 10 баров) — оставляем для совместимости
    """
    close = df['Close']
    high  = df['High']
    low   = df['Low']
    vol   = df['Volume']

    bull_fraction = (close - low) / (high - low + 1e-9)
    bear_fraction = (high - close) / (high - low + 1e-9)
    bull_vol = vol * bull_fraction
    bear_vol = vol * bear_fraction
    raw_flow = bull_vol - bear_vol

    d = df.copy()

    # Базовый OFI (10 баров) — совместимость
    ofi_10 = raw_flow.rolling(10).sum()
    vol_10 = vol.rolling(10).sum() + 1e-9
    d['OFI'] = ofi_10 / vol_10

    # OFI_5 — краткосрочный
    ofi_5 = raw_flow.rolling(5).sum()
    vol_5 = vol.rolling(5).sum() + 1e-9
    d['OFI_5'] = ofi_5 / vol_5

    # OFI_20 — среднесрочный
    ofi_20 = raw_flow.rolling(20).sum()
    vol_20 = vol.rolling(20).sum() + 1e-9
    d['OFI_20'] = ofi_20 / vol_20

    # OFI_ratio — ускорение потока
    d['OFI_ratio'] = d['OFI_5'] / (d['OFI_20'].abs() + 1e-9)

    return d


# ─────────────────────────────────────────────
# Индикаторы 1H (v8.3)
# ─────────────────────────────────────────────
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

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    d['MACD']        = ema12 - ema26
    d['MACD_signal'] = d['MACD'].ewm(span=9, adjust=False).mean()
    d['MACD_hist']   = d['MACD'] - d['MACD_signal']

    tr    = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(com=13, min_periods=14).mean()
    atr50 = tr.ewm(com=49, min_periods=50).mean()
    d['ATR']       = atr14
    d['ATR_pct']   = (atr14 / (close + 1e-9)) * 100
    d['ATR_norm']  = atr14 / (close + 1e-9)
    d['ATR_ratio'] = atr14 / (atr50 + 1e-9)

    sma20    = close.rolling(20).mean()
    std20    = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
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

    tp     = (high + low + close) / 3
    mf     = tp * vol
    pos_mf = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
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
    pdm  = up.where((up > down)   & (up > 0), 0)
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

    d['Hurst'] = calc_hurst_exponent(close)
    d = calc_vwap_features(d)

    rv = calc_realized_volatility(close)
    d['RV_20']    = rv['RV_20']
    d['RV_50']    = rv['RV_50']
    d['RV_ratio'] = rv['RV_ratio']

    # v8.3: расширенный OFI
    d = calc_ofi_extended(d)

    d['Price_accel'] = close.pct_change(1) - close.pct_change(1).shift(1)
    log_ret = np.log(close / close.shift(1))
    d['Vol_cluster'] = (log_ret**2).ewm(span=5).mean() / ((log_ret**2).ewm(span=20).mean() + 1e-9)

    return d


# ─────────────────────────────────────────────
# Индикаторы 4H
# ─────────────────────────────────────────────
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

    tr    = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(com=13, min_periods=14).mean()
    d['ATR_pct_4h']    = (atr14 / (close + 1e-9)) * 100
    d['Vol_ratio_4h']  = vol / (vol.rolling(20).mean() + 1e-9)
    d['Return_4h_tf']  = close.pct_change(1) * 100
    d['Return_24h_tf'] = close.pct_change(6) * 100

    up   = high.diff()
    down = -low.diff()
    pdm  = up.where((up > down)   & (up > 0), 0)
    mdm  = down.where((down > up) & (down > 0), 0)
    pdi  = 100 * (pdm.ewm(alpha=1/14).mean() / (atr14 + 1e-9))
    mdi  = 100 * (mdm.ewm(alpha=1/14).mean() / (atr14 + 1e-9))
    dx   = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    d['ADX_4h'] = dx.ewm(alpha=1/14).mean()

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    d['BB_pos_4h'] = (close - (sma20 - 2*std20)) / (4*std20 + 1e-9)
    d['Hurst_4h']  = calc_hurst_exponent(close, range(2, 15))

    return d


def merge_timeframes(df1h: pd.DataFrame, df4h: pd.DataFrame) -> pd.DataFrame:
    cols_4h        = [c for c in df4h.columns if c.endswith('_4h') or c.endswith('_4h_tf')]
    df4h_sub       = df4h[cols_4h].copy()
    df4h_reindexed = df4h_sub.reindex(df1h.index, method='ffill')
    df_merged      = pd.concat([df1h.copy(), df4h_reindexed], axis=1)
    df_merged      = df_merged.dropna(subset=cols_4h)
    return df_merged


# ─────────────────────────────────────────────
# TRIPLE BARRIER + HOLD класс (v8.3)
# ─────────────────────────────────────────────
def triple_barrier_labels(df: pd.DataFrame,
                           horizon: int = None,
                           tp_mult: float = None,
                           sl_mult: float = None) -> pd.DataFrame:
    """
    Возвращает Target_BUY и Target_SELL (0/1) + Target_HOLD (True/False).
    HOLD = ни TP, ни SL не достигнуты за горизонт.
    """
    if horizon is None: horizon = TARGET_HORIZON
    if tp_mult  is None: tp_mult = ATR_TP_MULT
    if sl_mult  is None: sl_mult = ATR_SL_MULT

    close = df['Close'].values
    atr   = df['ATR'].values
    high  = df['High'].values
    low   = df['Low'].values
    n     = len(df)

    target_buy  = np.full(n, np.nan)
    target_sell = np.full(n, np.nan)
    target_hold = np.zeros(n, dtype=bool)  # v8.3: HOLD класс

    for i in range(n - horizon):
        entry = close[i]
        atr_i = atr[i]

        tp_buy  = entry + atr_i * tp_mult
        sl_buy  = entry - atr_i * sl_mult
        tp_sell = entry - atr_i * tp_mult
        sl_sell = entry + atr_i * sl_mult

        buy_result  = np.nan
        sell_result = np.nan

        for j in range(i + 1, min(i + horizon + 1, n)):
            h = high[j]
            l = low[j]

            if np.isnan(buy_result):
                if h >= tp_buy and l <= sl_buy:
                    buy_result = 1 if close[j-1] < entry + atr_i * 0.5 else 0
                elif h >= tp_buy:
                    buy_result = 1
                elif l <= sl_buy:
                    buy_result = 0

            if np.isnan(sell_result):
                if l <= tp_sell and h >= sl_sell:
                    sell_result = 1 if close[j-1] > entry - atr_i * 0.5 else 0
                elif l <= tp_sell:
                    sell_result = 1
                elif h >= sl_sell:
                    sell_result = 0

            if not np.isnan(buy_result) and not np.isnan(sell_result):
                break

        # HOLD: ни одна граница не пробита за горизонт
        if np.isnan(buy_result) and np.isnan(sell_result):
            target_hold[i] = True
            buy_result  = 0
            sell_result = 0

        target_buy[i]  = buy_result
        target_sell[i] = sell_result

    df = df.copy()
    df['Target_BUY']  = target_buy
    df['Target_SELL'] = target_sell
    df['Target_HOLD'] = target_hold

    total     = n - horizon
    buy_valid = int(np.sum(~np.isnan(target_buy[:total])))
    buy_pos   = int(np.nansum(target_buy[:total]))
    hold_cnt  = int(target_hold[:total].sum())

    logger.info(
        f"[Trainer] Triple Barrier: "
        f"BUY pos={buy_pos}/{buy_valid} ({buy_pos/(buy_valid+1e-9):.1%}) | "
        f"HOLD={hold_cnt} ({hold_cnt/(total+1e-9):.1%})"
    )
    return df


# ─────────────────────────────────────────────
# SMOTE
# ─────────────────────────────────────────────
def apply_smote(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    if not SMOTE_AVAILABLE:
        return X_train, y_train
    pos = y_train.sum()
    neg = len(y_train) - pos
    if pos / (neg + 1e-9) > 0.4:
        return X_train, y_train
    try:
        smote = SMOTE(sampling_strategy=0.8, random_state=42, k_neighbors=5)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        logger.info(f"[Trainer] SMOTE: {len(y_train)}→{len(y_res)}")
        return X_res, y_res
    except Exception as e:
        logger.warning(f"[Trainer] SMOTE ошибка: {e}")
        return X_train, y_train


# ─────────────────────────────────────────────
# Feature Pruning
# ─────────────────────────────────────────────
def prune_features(model_xgb, model_lgbm, feature_cols: list,
                   threshold: float = FEATURE_IMPORTANCE_THRESHOLD) -> list:
    n = len(feature_cols)
    importance = np.zeros(n)
    if model_xgb is not None:
        try:
            imp = model_xgb.feature_importances_
            if len(imp) == n:
                importance += imp / (imp.sum() + 1e-9)
        except Exception:
            pass
    if model_lgbm is not None and LGBM_AVAILABLE:
        try:
            imp = model_lgbm.feature_importances_
            if len(imp) == n:
                importance += imp / (imp.sum() + 1e-9)
        except Exception:
            pass
    total = importance.sum()
    if total > 0:
        importance = importance / total
    else:
        return feature_cols
    importance_dict = {feature_cols[i]: float(importance[i]) for i in range(n)}
    try:
        with open(FEATURE_IMPORTANCE_PATH, 'w') as f:
            json.dump(dict(sorted(importance_dict.items(), key=lambda x: -x[1])), f, indent=2)
    except Exception:
        pass
    kept = [feature_cols[i] for i in range(n) if importance[i] >= threshold]
    if len(kept) < 15:
        kept = [feature_cols[i] for i in np.argsort(importance)[::-1][:20]]
    removed = [f for f in feature_cols if f not in kept]
    logger.info(f"[Trainer] Pruning: {n}→{len(kept)} (убрано {len(removed)})")
    return kept


# ─────────────────────────────────────────────
# Optuna + XGBoost
# ─────────────────────────────────────────────
def tune_xgboost(X_train, y_train, X_val, y_val, n_trials: int = 50) -> dict:
    tscv = TimeSeriesSplit(n_splits=3)
    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 150, 600),
            'max_depth':        trial.suggest_int('max_depth', 3, 7),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.12, log=True),
            'subsample':        trial.suggest_float('subsample', 0.55, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.55, 0.95),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 25),
            'gamma':            trial.suggest_float('gamma', 0.0, 0.7),
            'reg_alpha':        trial.suggest_float('reg_alpha', 0.0, 1.5),
            'reg_lambda':       trial.suggest_float('reg_lambda', 0.5, 4.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 6.0),
        }
        scores = []
        for tr_idx, val_idx in tscv.split(X_train, y_train):
            X_tr, X_v = X_train[tr_idx], X_train[val_idx]
            y_tr, y_v = y_train[tr_idx], y_train[val_idx]
            if y_tr.sum() < 5 or y_v.sum() < 2:
                continue
            m = XGBClassifier(**params, eval_metric='logloss', use_label_encoder=False, verbosity=0)
            m.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
            scores.append(precision_score(y_v, m.predict(X_v), zero_division=0))
        return float(np.mean(scores)) if scores else 0.0
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"[Trainer] Optuna best: {study.best_value:.3f}")
    return study.best_params


def train_binary_xgb(X_train, y_train, X_test, y_test, best_params=None) -> tuple:
    if best_params is None:
        best_params = {'n_estimators':400,'max_depth':4,'learning_rate':0.03,
                       'subsample':0.75,'colsample_bytree':0.70,'min_child_weight':10,
                       'gamma':0.2,'reg_alpha':0.3,'reg_lambda':2.0,'scale_pos_weight':2.0}
    model = XGBClassifier(**best_params, eval_metric='logloss', use_label_encoder=False, verbosity=0)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return model, {
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_test, y_pred, zero_division=0)),
        'f1':        float(f1_score(y_test, y_pred, zero_division=0)),
        'accuracy':  float(accuracy_score(y_test, y_pred)),
        'roc_auc':   float(roc_auc_score(y_test, y_proba)) if y_test.sum() > 0 else 0.0,
    }


def train_binary_lgbm(X_train, y_train, X_test, y_test) -> tuple:
    if not LGBM_AVAILABLE:
        return None, None
    scale_pos = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9)
    model = lgb.LGBMClassifier(n_estimators=400, max_depth=4, learning_rate=0.03,
                                subsample=0.75, colsample_bytree=0.70, min_child_samples=20,
                                reg_alpha=0.3, reg_lambda=2.0, scale_pos_weight=min(scale_pos, 5.0),
                                objective='binary', verbosity=-1, n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return model, {
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_test, y_pred, zero_division=0)),
        'f1':        float(f1_score(y_test, y_pred, zero_division=0)),
        'accuracy':  float(accuracy_score(y_test, y_pred)),
        'roc_auc':   float(roc_auc_score(y_test, y_proba)) if y_test.sum() > 0 else 0.0,
    }


def train_binary_cat(X_train, y_train, X_test, y_test) -> tuple:
    if not CATBOOST_AVAILABLE:
        return None, None
    scale_pos = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9)
    model = CatBoostClassifier(iterations=400, depth=4, learning_rate=0.03,
                                l2_leaf_reg=3.0, subsample=0.75,
                                scale_pos_weight=min(scale_pos, 5.0),
                                eval_metric="AUC", verbose=0, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return model, {
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_test, y_pred, zero_division=0)),
        'f1':        float(f1_score(y_test, y_pred, zero_division=0)),
        'accuracy':  float(accuracy_score(y_test, y_pred)),
        'roc_auc':   float(roc_auc_score(y_test, y_proba)) if y_test.sum() > 0 else 0.0,
    }


# ─────────────────────────────────────────────
# Калибровка
# ─────────────────────────────────────────────
def calibrate_model(model, X_train, y_train, X_test, y_test, label="BUY") -> tuple:
    try:
        tscv = TimeSeriesSplit(n_splits=3)
        calibrated = CalibratedClassifierCV(estimator=model, method='isotonic', cv=tscv)
        calibrated.fit(X_train, y_train)
        y_proba_raw = model.predict_proba(X_test)[:, 1]
        y_proba_cal = calibrated.predict_proba(X_test)[:, 1]
        y_pred_cal  = calibrated.predict(X_test)
        raw_auc = float(roc_auc_score(y_test, y_proba_raw)) if y_test.sum() > 0 else 0.0
        cal_auc = float(roc_auc_score(y_test, y_proba_cal)) if y_test.sum() > 0 else 0.0
        prec_cal = float(precision_score(y_test, y_pred_cal, zero_division=0))
        logger.info(f"[Trainer] Calib {label}: AUC {raw_auc:.3f}→{cal_auc:.3f} prec={prec_cal:.1%}")
        return calibrated, {'precision': prec_cal, 'roc_auc': cal_auc, 'raw_auc': raw_auc}
    except Exception as e:
        logger.warning(f"[Trainer] Калибровка {label} ошибка: {e}")
        return model, {}


# ─────────────────────────────────────────────
# Stacking
# ─────────────────────────────────────────────
def train_stacking_ensemble(model_xgb, model_lgbm, X_train, y_train,
                             X_test, y_test, label="BUY", model_cat=None) -> tuple:
    if not LGBM_AVAILABLE or model_lgbm is None:
        return None, None
    try:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        p_xgb_oof  = np.zeros(len(X_train))
        p_lgbm_oof = np.zeros(len(X_train))
        p_cat_oof  = np.zeros(len(X_train))
        for _, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr = y_train[tr_idx]
            if y_tr.sum() < 5: continue
            xf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                eval_metric='logloss', use_label_encoder=False, verbosity=0)
            xf.fit(X_tr, y_tr, verbose=False)
            p_xgb_oof[val_idx] = xf.predict_proba(X_val)[:, 1]
            if LGBM_AVAILABLE:
                lf = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, verbosity=-1, n_jobs=-1)
                lf.fit(X_tr, y_tr)
                p_lgbm_oof[val_idx] = lf.predict_proba(X_val)[:, 1]
            if CATBOOST_AVAILABLE:
                cf = CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05, verbose=0)
                cf.fit(X_tr, y_tr)
                p_cat_oof[val_idx] = cf.predict_proba(X_val)[:, 1]

        p_avg_oof  = (p_xgb_oof + p_lgbm_oof + p_cat_oof) / 3
        p_diff_oof = p_xgb_oof - p_lgbm_oof
        X_stack_train = np.column_stack([p_xgb_oof, p_lgbm_oof, p_cat_oof, p_avg_oof, p_diff_oof])

        p_xgb  = model_xgb.predict_proba(X_test)[:, 1].reshape(-1, 1)
        p_lgbm = model_lgbm.predict_proba(X_test)[:, 1].reshape(-1, 1)
        p_cat  = model_cat.predict_proba(X_test)[:, 1].reshape(-1, 1) if (CATBOOST_AVAILABLE and model_cat is not None) else p_xgb
        X_stack_test = np.hstack([p_xgb, p_lgbm, p_cat, (p_xgb+p_lgbm+p_cat)/3, p_xgb-p_lgbm])

        scaler = StandardScaler()
        X_st_tr = scaler.fit_transform(X_stack_train)
        X_st_te = scaler.transform(X_stack_test)

        stack_model = RidgeClassifierCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=5)
        stack_model.fit(X_st_tr, y_train)
        y_pred  = stack_model.predict(X_st_te)
        d = stack_model.decision_function(X_st_te)
        y_proba = 1 / (1 + np.exp(-d))
        metrics = {
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall':    float(recall_score(y_test, y_pred, zero_division=0)),
            'roc_auc':   float(roc_auc_score(y_test, y_proba)) if y_test.sum() > 0 else 0.0,
        }
        logger.info(f"[Trainer] Stack {label}: prec={metrics['precision']:.1%} auc={metrics['roc_auc']:.3f}")
        return {'model': stack_model, 'scaler': scaler}, metrics
    except Exception as e:
        logger.warning(f"[Trainer] Stack {label} ошибка: {e}")
        return None, None


# ─────────────────────────────────────────────
# Meta-Labeling
# ─────────────────────────────────────────────
def train_meta_model(X_train, y_true_train, X_test, y_true_test, side_model) -> tuple:
    try:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        oof_proba = np.zeros(len(X_train))
        for _, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_true_train)):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr = y_true_train[tr_idx]
            if y_tr.sum() < 5:
                oof_proba[val_idx] = 0.5; continue
            fm = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                               subsample=0.75, colsample_bytree=0.7, min_child_weight=10,
                               gamma=0.2, eval_metric='logloss', use_label_encoder=False, verbosity=0)
            fm.fit(X_tr, y_tr, verbose=False)
            oof_proba[val_idx] = fm.predict_proba(X_val)[:, 1]
        oof_pred_binary = (oof_proba >= 0.50).astype(int)
        y_meta_train = ((oof_pred_binary == 1) & (y_true_train == 1)).astype(int)
        if y_meta_train.sum() < 10:
            logger.warning("[Trainer] Meta-model: мало примеров, пропуск")
            return None, None
        X_meta_train = np.hstack([X_train, oof_proba.reshape(-1, 1)])
        p_test = side_model.predict_proba(X_test)[:, 1]
        X_meta_test = np.hstack([X_test, p_test.reshape(-1, 1)])
        y_meta_test = ((p_test >= 0.50).astype(int) == 1) & (y_true_test == 1)
        y_meta_test = y_meta_test.astype(int)
        X_meta_sm, y_meta_sm = apply_smote(X_meta_train, y_meta_train)
        meta_model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                   subsample=0.75, colsample_bytree=0.7, min_child_weight=10,
                                   gamma=0.3, eval_metric='logloss', use_label_encoder=False, verbosity=0)
        meta_model.fit(X_meta_sm, y_meta_sm, eval_set=[(X_meta_test, y_meta_test)], verbose=False)
        y_pred  = meta_model.predict(X_meta_test)
        y_proba = meta_model.predict_proba(X_meta_test)[:, 1]
        if y_meta_test.sum() > 0:
            metrics = {'precision': float(precision_score(y_meta_test, y_pred, zero_division=0)),
                       'recall':    float(recall_score(y_meta_test, y_pred, zero_division=0)),
                       'roc_auc':   float(roc_auc_score(y_meta_test, y_proba))}
        else:
            metrics = {'precision': 0.0, 'recall': 0.0, 'roc_auc': 0.0}
        logger.info(f"[Trainer] Meta-model: prec={metrics['precision']:.1%}")
        return meta_model, metrics
    except Exception as e:
        logger.warning(f"[Trainer] Meta-model ошибка: {e}")
        return None, None


# ─────────────────────────────────────────────
# Kelly
# ─────────────────────────────────────────────
def calc_kelly_criterion(win_rate, avg_win_pct, avg_loss_pct):
    if avg_loss_pct <= 0 or win_rate <= 0: return 0.10
    odds  = avg_win_pct / avg_loss_pct
    kelly = (odds * win_rate - (1 - win_rate)) / odds
    return round(max(0.0, min(kelly, 0.5)) / 2, 3)


def calc_kelly_from_wf_returns(trade_returns):
    if len(trade_returns) < 10: return 0.10
    arr    = np.array(trade_returns)
    wins   = arr[arr > 0]
    losses = arr[arr < 0]
    if len(wins) == 0 or len(losses) == 0: return 0.10
    return calc_kelly_criterion(len(wins)/len(arr), float(wins.mean()), float(abs(losses.mean())))


# ─────────────────────────────────────────────
# Walk-Forward
# ─────────────────────────────────────────────
def walk_forward_binary(X, y, train_size, test_size, step):
    results = []
    all_returns = []
    n = len(X)
    start = train_size
    while start + test_size <= n:
        X_tr = X[start-train_size:start]
        y_tr = y[start-train_size:start]
        X_te = X[start:start+test_size]
        y_te = y[start:start+test_size]
        if y_tr.sum() < 5 or y_te.sum() < 3:
            start += step; continue
        X_tr_sm, y_tr_sm = apply_smote(X_tr, y_tr)
        spw = min((len(y_tr) - y_tr.sum()) / (y_tr.sum() + 1e-9), 5.0)
        m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.03,
                          subsample=0.75, colsample_bytree=0.7, min_child_weight=10,
                          gamma=0.2, reg_alpha=0.3, reg_lambda=2.0, scale_pos_weight=spw,
                          eval_metric='logloss', use_label_encoder=False, verbosity=0)
        m.fit(X_tr_sm, y_tr_sm, verbose=False)
        y_pred = m.predict(X_te)
        prec = precision_score(y_te, y_pred, zero_division=0)
        rec  = recall_score(y_te, y_pred, zero_division=0)
        tp_pct = ATR_TP_MULT * 0.015
        sl_pct = ATR_SL_MULT * 0.015
        hourly_returns = []
        for pred, true in zip(y_pred, y_te):
            if pred == 1:
                r = (tp_pct if true == 1 else -sl_pct) * 100
                hourly_returns.append(r)
                all_returns.append(r)
            else:
                hourly_returns.append(0.0)
        r = np.array(hourly_returns)
        sharpe = float(r.mean() / (r.std() + 1e-9) * np.sqrt(8760)) if r.std() > 0 else 0.0
        results.append({'precision': prec, 'recall': rec, 'sharpe': sharpe})
        start += step
    if not results:
        return {'wf_precision':0.0,'wf_recall':0.0,'wf_sharpe':0.0,'wf_folds':0,'wf_trade_returns':[]}
    return {
        'wf_precision':     round(float(np.mean([r['precision'] for r in results])), 4),
        'wf_recall':        round(float(np.mean([r['recall']    for r in results])), 4),
        'wf_sharpe':        round(float(np.mean([r['sharpe']    for r in results])), 3),
        'wf_folds':         len(results),
        'wf_trade_returns': all_returns,
    }


def get_available_features(df, desired):
    available = [c for c in desired if c in df.columns]
    missing   = [c for c in desired if c not in df.columns]
    if missing:
        logger.warning(f"[Trainer] Отсутствуют: {missing}")
    return available


# v8.3: расширенный список признаков
FEATURE_COLS_V83_EXTRA = [
    'Hurst', 'VWAP_dev_20', 'VWAP_dev_50', 'VWAP_bull_ratio',
    'RV_20', 'RV_50', 'RV_ratio',
    'OFI', 'OFI_5', 'OFI_20', 'OFI_ratio',          # v8.3 NEW
    'Funding_rate', 'Funding_bias',                   # v8.3 NEW
    'Price_accel', 'Vol_cluster', 'Hurst_4h',
    'Target_HOLD',                                    # v8.3 HOLD признак
]


# ─────────────────────────────────────────────
# ГЛАВНАЯ: train_model v8.3
# ─────────────────────────────────────────────
def train_model() -> dict:
    logger.info("[Trainer] 🚀 v8.3: OFI_5+OFI_20+OFI_ratio + Funding + HOLD class")

    # 1. Данные
    df1h_raw = fetch_ohlcv("BTC-USDT", "1H", BARS_1H)
    df4h_raw = fetch_ohlcv("BTC-USDT", "4H", BARS_4H)
    if df1h_raw.empty:
        return {"success": False, "error": "Нет 1H данных"}

    # 2. Индикаторы
    df1h = calc_indicators_1h(df1h_raw)

    # v8.3: Funding
    df1h = merge_funding(df1h, "BTC-USDT-SWAP")

    if not df4h_raw.empty:
        df4h = calc_indicators_4h(df4h_raw)
        df   = merge_timeframes(df1h, df4h)
        logger.info(f"[Trainer] ✅ 1H+4H объединены | строк: {len(df)}")
    else:
        logger.warning("[Trainer] ⚠️ 4H недоступны")
        df = df1h

    df = df.dropna()
    if len(df) < 300:
        return {"success": False, "error": f"Мало данных: {len(df)}"}

    # 3. Triple Barrier + HOLD
    logger.info("[Trainer] 🎯 Triple Barrier (с HOLD классом)...")
    df = triple_barrier_labels(df)

    df_buy  = df[~df['Target_BUY'].isna()].copy()
    df_sell = df[~df['Target_SELL'].isna()].copy()

    # 4. Признаки v8.3
    all_feature_cols = FEATURE_COLS + [f for f in FEATURE_COLS_V83_EXTRA if f not in FEATURE_COLS]
    feature_cols = get_available_features(df, all_feature_cols)
    if len(feature_cols) < 10:
        feature_cols = get_available_features(df, FEATURE_COLS_LEGACY)
        logger.warning(f"[Trainer] Legacy: {len(feature_cols)} признаков")
    else:
        logger.info(f"[Trainer] Используем {len(feature_cols)} признаков (v8.3)")

    X_buy  = np.nan_to_num(df_buy[feature_cols].values.astype(np.float32),  nan=0.0, posinf=0.0, neginf=0.0)
    y_buy  = df_buy['Target_BUY'].values.astype(int)
    X_sell = np.nan_to_num(df_sell[feature_cols].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y_sell = df_sell['Target_SELL'].values.astype(int)

    # 5. Train/test split
    split_buy  = int(len(X_buy)  * 0.8)
    split_sell = int(len(X_sell) * 0.8)
    X_buy_train,  X_buy_test  = X_buy[:split_buy],   X_buy[split_buy:]
    y_buy_train,  y_buy_test  = y_buy[:split_buy],   y_buy[split_buy:]
    X_sell_train, X_sell_test = X_sell[:split_sell], X_sell[split_sell:]
    y_sell_train, y_sell_test = y_sell[:split_sell], y_sell[split_sell:]

    logger.info(f"[Trainer] BUY train={len(X_buy_train)} pos={y_buy_train.sum()} | SELL train={len(X_sell_train)} pos={y_sell_train.sum()}")

    # 6. SMOTE
    X_buy_sm,  y_buy_sm  = apply_smote(X_buy_train,  y_buy_train)
    X_sell_sm, y_sell_sm = apply_smote(X_sell_train, y_sell_train)

    # 7. Optuna
    logger.info("[Trainer] 🔬 Optuna (50 trials)...")
    best_params = tune_xgboost(X_buy_sm, y_buy_sm, X_buy_test, y_buy_test, n_trials=50)

    # 8. Модели BUY
    buy_xgb,  buy_xgb_m  = train_binary_xgb(X_buy_sm, y_buy_sm, X_buy_test, y_buy_test, best_params)
    buy_lgbm, buy_lgbm_m = train_binary_lgbm(X_buy_sm, y_buy_sm, X_buy_test, y_buy_test)
    buy_cat,  buy_cat_m  = train_binary_cat(X_buy_sm, y_buy_sm, X_buy_test, y_buy_test)
    logger.info(f"[Trainer] BUY XGB={buy_xgb_m['precision']:.1%} LGBM={buy_lgbm_m['precision'] if buy_lgbm_m else 0:.1%} CAT={buy_cat_m['precision'] if buy_cat_m else 0:.1%}")

    # 9. Модели SELL
    sell_xgb,  sell_xgb_m  = train_binary_xgb(X_sell_sm, y_sell_sm, X_sell_test, y_sell_test, best_params)
    sell_lgbm, sell_lgbm_m = train_binary_lgbm(X_sell_sm, y_sell_sm, X_sell_test, y_sell_test)
    sell_cat,  sell_cat_m  = train_binary_cat(X_sell_sm, y_sell_sm, X_sell_test, y_sell_test)

    # 10. Pruning
    feature_cols_pruned = prune_features(buy_xgb, buy_lgbm, feature_cols)
    if len(feature_cols_pruned) < len(feature_cols) * 0.85:
        idx = [feature_cols.index(f) for f in feature_cols_pruned if f in feature_cols]
        X_bp = X_buy[:, idx];  X_sp = X_sell[:, idx]
        X_bp_tr, X_bp_te = X_bp[:split_buy],   X_bp[split_buy:]
        X_sp_tr, X_sp_te = X_sp[:split_sell],  X_sp[split_sell:]
        X_bp_sm, y_bp_sm = apply_smote(X_bp_tr, y_buy_train)
        X_sp_sm, y_sp_sm = apply_smote(X_sp_tr, y_sell_train)
        bxp, bxpm = train_binary_xgb(X_bp_sm, y_bp_sm, X_bp_te, y_buy_test, best_params)
        blp, blpm = train_binary_lgbm(X_bp_sm, y_bp_sm, X_bp_te, y_buy_test)
        sxp, sxpm = train_binary_xgb(X_sp_sm, y_sp_sm, X_sp_te, y_sell_test, best_params)
        slp, slpm = train_binary_lgbm(X_sp_sm, y_sp_sm, X_sp_te, y_sell_test)
        if bxpm['precision'] >= buy_xgb_m['precision']:
            buy_xgb, buy_xgb_m, buy_lgbm, buy_lgbm_m = bxp, bxpm, blp, blpm
            sell_xgb, sell_xgb_m, sell_lgbm, sell_lgbm_m = sxp, sxpm, slp, slpm
            feature_cols = feature_cols_pruned
            X_buy_train, X_buy_test = X_bp_tr, X_bp_te
            X_sell_train, X_sell_test = X_sp_tr, X_sp_te
            logger.info(f"[Trainer] ✅ Pruned: {len(feature_cols)} признаков")

    # 11. Stacking
    stack_buy,  stack_buy_m  = train_stacking_ensemble(buy_xgb,  buy_lgbm,  X_buy_train,  y_buy_train,  X_buy_test,  y_buy_test,  "BUY",  model_cat=buy_cat)
    stack_sell, stack_sell_m = train_stacking_ensemble(sell_xgb, sell_lgbm, X_sell_train, y_sell_train, X_sell_test, y_sell_test, "SELL", model_cat=sell_cat)

    # 12. Meta-labeling
    meta_buy,  meta_buy_m  = train_meta_model(X_buy_train,  y_buy_train,  X_buy_test,  y_buy_test,  buy_xgb)
    meta_sell, meta_sell_m = train_meta_model(X_sell_train, y_sell_train, X_sell_test, y_sell_test, sell_xgb)

    # 13. Калибровка
    calib_buy,  calib_buy_m  = calibrate_model(buy_xgb,  X_buy_train,  y_buy_train,  X_buy_test,  y_buy_test,  "BUY")
    calib_sell, calib_sell_m = calibrate_model(sell_xgb, X_sell_train, y_sell_train, X_sell_test, y_sell_test, "SELL")

    # 14. Walk-Forward
    n_s = len(X_buy)
    wf_train = max(int(n_s * 0.55), 100)
    wf_test  = max(int(n_s * 0.12), 30)
    wf_step  = max(int(n_s * 0.08), 20)
    logger.info(f"[Trainer] 📊 Walk-Forward (train={wf_train} test={wf_test} step={wf_step})...")
    wf_buy  = walk_forward_binary(X_buy,  y_buy,  wf_train, wf_test, wf_step)
    wf_sell = walk_forward_binary(X_sell, y_sell, wf_train, wf_test, wf_step)
    logger.info(f"[Trainer] WF BUY: prec={wf_buy['wf_precision']:.1%} sharpe={wf_buy['wf_sharpe']:.2f}")
    logger.info(f"[Trainer] WF SELL: prec={wf_sell['wf_precision']:.1%} sharpe={wf_sell['wf_sharpe']:.2f}")

    # 15. Kelly
    all_wf_returns = wf_buy.get('wf_trade_returns', []) + wf_sell.get('wf_trade_returns', [])
    kelly_f = calc_kelly_from_wf_returns(all_wf_returns) if len(all_wf_returns) >= 10 else calc_kelly_criterion(wf_buy['wf_precision'], ATR_TP_MULT * 1.5, ATR_SL_MULT * 1.5)
    logger.info(f"[Trainer] Kelly (Half): {kelly_f:.1%}")

    # 16. Сохранение
    joblib.dump(buy_xgb,  MODEL_PATH_BUY_XGB)
    joblib.dump(sell_xgb, MODEL_PATH_SELL_XGB)
    if buy_lgbm:  joblib.dump(buy_lgbm,  MODEL_PATH_BUY_LGBM)
    if sell_lgbm: joblib.dump(sell_lgbm, MODEL_PATH_SELL_LGBM)
    if meta_buy:  joblib.dump(meta_buy,  META_MODEL_BUY_PATH)
    if meta_sell: joblib.dump(meta_sell, META_MODEL_SELL_PATH)
    if stack_buy:  joblib.dump(stack_buy,  STACK_MODEL_BUY_PATH)
    if stack_sell: joblib.dump(stack_sell, STACK_MODEL_SELL_PATH)
    if calib_buy:  joblib.dump(calib_buy,  CALIB_MODEL_BUY_PATH)
    if calib_sell: joblib.dump(calib_sell, CALIB_MODEL_SELL_PATH)
    with open(MODEL_FEATURES_PATH, 'w') as f:
        json.dump(feature_cols, f)

    avg_buy_prec  = (buy_xgb_m['precision'] + (buy_lgbm_m['precision']  if buy_lgbm_m  else buy_xgb_m['precision']))  / 2
    avg_sell_prec = (sell_xgb_m['precision'] + (sell_lgbm_m['precision'] if sell_lgbm_m else sell_xgb_m['precision'])) / 2
    avg_buy_auc   = (buy_xgb_m['roc_auc']   + (buy_lgbm_m['roc_auc']   if buy_lgbm_m  else buy_xgb_m['roc_auc']))    / 2
    avg_sell_auc  = (sell_xgb_m['roc_auc']  + (sell_lgbm_m['roc_auc']  if sell_lgbm_m else sell_xgb_m['roc_auc']))   / 2

    stats = {
        "success": True, "version": "8.3",
        "labeling": "triple_barrier+hold",
        "n_features": len(feature_cols),
        "n_samples": len(df_buy),
        "bars_loaded": BARS_1H,
        "kelly_fraction": kelly_f,
        "avg_buy_precision": avg_buy_prec,
        "avg_sell_precision": avg_sell_prec,
        "avg_buy_auc": avg_buy_auc,
        "avg_sell_auc": avg_sell_auc,
        "wf_buy_precision":  wf_buy['wf_precision'],
        "wf_sell_precision": wf_sell['wf_precision'],
        "wf_buy_sharpe":     wf_buy['wf_sharpe'],
        "wf_sell_sharpe":    wf_sell['wf_sharpe'],
        "wf_folds": wf_buy['wf_folds'],
        "wf_trade_count": len(all_wf_returns),
        "stack_buy_precision":  stack_buy_m['precision']  if stack_buy_m  else None,
        "stack_sell_precision": stack_sell_m['precision'] if stack_sell_m else None,
        "meta_buy_precision":   meta_buy_m['precision']   if meta_buy_m   else None,
        "calib_buy_auc":  calib_buy_m.get('roc_auc')  if calib_buy_m  else None,
        "calib_sell_auc": calib_sell_m.get('roc_auc') if calib_sell_m else None,
        "wf_sharpe_buy": wf_buy['wf_sharpe'],
        "avg_wf_sharpe_buy": wf_buy['wf_sharpe'],
        "xgb_precision": avg_buy_prec,
        "ensemble_precision": (avg_buy_prec + avg_sell_prec) / 2,
        "wf_precision": (wf_buy['wf_precision'] + wf_sell['wf_precision']) / 2,
        "wf_accuracy": 0.0,
        "lgbm_available": LGBM_AVAILABLE,
        "smote_available": SMOTE_AVAILABLE,
        "meta_labeling": meta_buy is not None,
        "stacking": stack_buy is not None,
        "calibration": calib_buy is not None,
        "ofi_extended": True,
        "funding_features": True,
        "hold_class": True,
    }
    with open(STATS_FILE, 'w') as f:
        json.dump({k: v for k, v in stats.items() if k != 'wf_trade_returns'}, f, indent=2)

    logger.info(
        f"[Trainer] ✅ v8.3 Готово! "
        f"BUY prec={avg_buy_prec:.1%} | SELL prec={avg_sell_prec:.1%} | "
        f"WF Sharpe={wf_buy['wf_sharpe']:.2f} | Kelly={kelly_f:.1%} | "
        f"Features={len(feature_cols)} | OFI_ext=✅ Funding=✅ HOLD=✅"
    )
    return {**stats, "model": buy_xgb, "lgbm_model": buy_lgbm}


def _git_push_if_better(stats):
    try:
        import subprocess
        prec = stats.get("avg_buy_precision", 0)
        sharpe = stats.get("wf_buy_sharpe", 0)
        kelly  = stats.get("kelly_fraction", 0)
        msg = f"auto: BUY_prec={prec:.1%} Sharpe={sharpe:.2f} Kelly={kelly:.1%} v8.3"
        repo = "/root/TradingCons"
        files = ["app.py","live_signal.py","auto_trainer.py","config.py"]
        subprocess.run(["git","-C",repo,"add"] + files, check=True)
        subprocess.run(["git","-C",repo,"commit","-m",msg], capture_output=True)
        subprocess.run(["git","-C",repo,"push","tradebot-btc","main"], check=True, capture_output=True)
        logger.info(f"[Git] ✅ Запушено: {msg}")
    except Exception as e:
        logger.warning(f"[Git] Push пропущен: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    result = train_model()
    if result.get("success"):
        print(f"\n✅ v8.3 Готово!")
        print(f"   Features: {result['n_features']}")
        print(f"   BUY prec: {result['avg_buy_precision']:.1%}")
        print(f"   WF Sharpe: {result['wf_buy_sharpe']:.2f}")
        print(f"   Kelly: {result['kelly_fraction']:.1%}")
        print(f"   OFI_extended: ✅  Funding: ✅  HOLD: ✅")
    else:
        print(f"❌ {result.get('error')}")
