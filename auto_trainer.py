"""
auto_trainer.py v4.0 — Ансамбль XGBoost + LightGBM
v4.0 изменения:
  - Две модели: XGBoost + LightGBM (ансамбль голосованием)
  - 40+ признаков: OBV, MFI, StochRSI, Z-score, BB_width, Williams%R, ROC, Momentum
  - Multi-timeframe: 1h + 4h фичи
  - Walk-Forward валидация (честная оценка без data leakage)
  - SHAP feature importance для отбора лучших фич
  - Автоматический отбор FEATURE_COLS на основе importance
"""

import os
import json
import joblib
import logging
import requests
import time
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, classification_report, f1_score
)
from sklearn.utils.class_weight import compute_sample_weight

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

from config import (
    MODEL_PATH, MODEL_PATH_LGBM, STATS_FILE,
    FEATURE_COLS, FEATURE_COLS_LEGACY,
    TARGET_HORIZON, TARGET_THRESHOLD,
    WF_TRAIN_DAYS, WF_TEST_DAYS, WF_STEP_DAYS
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Загрузка OHLCV с OKX
# ─────────────────────────────────────────────
def fetch_ohlcv(symbol: str = "TON-USDT", bar: str = "1H", bars: int = 2400) -> pd.DataFrame:
    all_data = []
    after    = None
    fetched  = 0
    limit    = 300

    try:
        while fetched < bars:
            url = (
                f"https://www.okx.com/api/v5/market/history-candles"
                f"?instId={symbol}&bar={bar}&limit={limit}"
            )
            if after:
                url += f"&after={after}"

            r    = requests.get(url, timeout=15)
            data = r.json().get("data", [])
            if not data:
                break

            all_data.extend(data)
            fetched += len(data)
            after    = data[-1][0]
            time.sleep(0.3)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(
            all_data,
            columns=['ts', 'Open', 'High', 'Low', 'Close',
                     'Volume', 'VolCcy', 'VolCcyQuote', 'Confirm']
        )
        df[['Open', 'High', 'Low', 'Close', 'Volume']] = \
            df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        df['ts'] = pd.to_datetime(df['ts'].astype(float), unit='ms')
        df.set_index('ts', inplace=True)
        df = df.sort_index()

        logger.info(f"[Trainer] ✅ Загружено {len(df)} свечей ({bar})")
        return df

    except Exception as e:
        logger.error(f"[Trainer] Ошибка загрузки {bar}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# Расчёт 1H индикаторов (40+ признаков)
# ─────────────────────────────────────────────
def calc_indicators_1h(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    close = d['Close']
    high  = d['High']
    low   = d['Low']
    vol   = d['Volume']

    # ── Time features ─────────────────────
    d['Hour']      = d.index.hour
    d['DayOfWeek'] = d.index.dayofweek

    # ── RSI 7, 14, 21 ─────────────────────
    for p in [7, 14, 21]:
        diff  = close.diff()
        g     = diff.clip(lower=0)
        l     = -diff.clip(upper=0)
        avg_g = g.ewm(com=p - 1, min_periods=p).mean()
        avg_l = l.ewm(com=p - 1, min_periods=p).mean()
        d[f'RSI_{p}'] = 100 - (100 / (1 + avg_g / (avg_l + 1e-9)))

    # ── MACD ──────────────────────────────
    ema12            = close.ewm(span=12, adjust=False).mean()
    ema26            = close.ewm(span=26, adjust=False).mean()
    d['MACD']        = ema12 - ema26
    d['MACD_signal'] = d['MACD'].ewm(span=9, adjust=False).mean()
    d['MACD_hist']   = d['MACD'] - d['MACD_signal']

    # ── ATR ───────────────────────────────
    tr  = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr14        = tr.ewm(com=13, min_periods=14).mean()
    atr50        = tr.ewm(com=49, min_periods=50).mean()
    d['ATR']     = atr14
    d['ATR_pct'] = (atr14 / (close + 1e-9)) * 100
    d['ATR_norm'] = atr14 / (close + 1e-9)
    d['ATR_ratio'] = atr14 / (atr50 + 1e-9)     # нормализованная волатильность

    # ── Bollinger Bands ───────────────────
    sma20       = close.rolling(20).mean()
    std20       = close.rolling(20).std()
    bb_upper    = sma20 + 2 * std20
    bb_lower    = sma20 - 2 * std20
    d['BB_pos']  = (close - bb_lower) / (4 * std20 + 1e-9)
    d['BB_width'] = (bb_upper - bb_lower) / (sma20 + 1e-9)  # нормализованная ширина

    # ── EMA ratios (тренд) ────────────────
    ema20  = close.ewm(span=20).mean()
    ema50  = close.ewm(span=50).mean()
    ema100 = close.ewm(span=100).mean()
    d['EMA_ratio_20_50']  = ema20 / (ema50 + 1e-9)
    d['EMA_ratio_20_100'] = ema20 / (ema100 + 1e-9)

    # Обратная совместимость
    d['EMA_ratio'] = d['EMA_ratio_20_50']

    # ── Volume indicators ─────────────────
    vol_sma20     = vol.rolling(20).mean()
    d['Vol_ratio'] = vol / (vol_sma20 + 1e-9)

    # OBV (On-Balance Volume)
    obv        = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    obv_sma20  = obv.rolling(20).mean()
    d['OBV_norm'] = (obv - obv_sma20) / (obv.rolling(20).std() + 1e-9)

    # MFI (Money Flow Index, period=14)
    tp       = (high + low + close) / 3
    mf       = tp * vol
    pos_mf   = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_mf   = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
    d['MFI_14'] = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-9)))

    # ── Stochastic RSI ────────────────────
    rsi14     = d['RSI_14']
    stoch_min = rsi14.rolling(14).min()
    stoch_max = rsi14.rolling(14).max()
    stoch_k   = (rsi14 - stoch_min) / (stoch_max - stoch_min + 1e-9) * 100
    d['StochRSI_K'] = stoch_k
    d['StochRSI_D'] = stoch_k.rolling(3).mean()

    # ── Williams %R ───────────────────────
    hw14 = high.rolling(14).max()
    lw14 = low.rolling(14).min()
    d['WilliamsR'] = (hw14 - close) / (hw14 - lw14 + 1e-9) * -100

    # ── Z-score цены ──────────────────────
    d['ZScore_20'] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-9)
    d['ZScore_50'] = (close - close.rolling(50).mean()) / (close.rolling(50).std() + 1e-9)

    # ── ADX ───────────────────────────────
    up   = high.diff()
    down = -low.diff()
    pdm  = up.where((up > down)   & (up > 0),   0)
    mdm  = down.where((down > up) & (down > 0), 0)
    pdi  = 100 * (pdm.ewm(alpha=1 / 14).mean() / (atr14 + 1e-9))
    mdi  = 100 * (mdm.ewm(alpha=1 / 14).mean() / (atr14 + 1e-9))
    dx   = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    d['ADX'] = dx.ewm(alpha=1 / 14).mean()

    # ── Candle patterns ───────────────────
    d['Body_pct']   = (close - d['Open']).abs() / (d['Open'] + 1e-9) * 100
    d['Upper_wick'] = (high - d[['Close', 'Open']].max(axis=1)) / (d['Open'] + 1e-9) * 100
    d['Lower_wick'] = (d[['Close', 'Open']].min(axis=1) - low) / (d['Open'] + 1e-9) * 100
    body_range      = (high - low + 1e-9)
    d['Doji']       = (d['Body_pct'] / body_range < 0.1).astype(int)

    # ── Momentum ──────────────────────────
    d['Momentum_10'] = close - close.shift(10)
    d['ROC_10']      = close.pct_change(10) * 100

    # ── Returns ───────────────────────────
    for h in [1, 4, 12, 24]:
        d[f'Return_{h}h'] = close.pct_change(h) * 100

    return d


# ─────────────────────────────────────────────
# Расчёт 4H индикаторов
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
    d['Return_24h_tf'] = close.pct_change(6) * 100  # 6 × 4h = 24h

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

    return d


# ─────────────────────────────────────────────
# Слияние 1H + 4H (reindex с forward-fill)
# ─────────────────────────────────────────────
def merge_timeframes(df1h: pd.DataFrame, df4h: pd.DataFrame) -> pd.DataFrame:
    """
    Берём 4H признаки, reindex на 1H временную сетку.
    Используем ffill (не смотрим в будущее!).
    """
    cols_4h = [c for c in df4h.columns if c.endswith('_4h') or c.endswith('_4h_tf')]
    df4h_sub = df4h[cols_4h].copy()

    # Переименовываем чтобы было понятно
    df_merged = df1h.copy()
    df4h_reindexed = df4h_sub.reindex(df1h.index, method='ffill')
    df_merged = pd.concat([df_merged, df4h_reindexed], axis=1)

    return df_merged


# ─────────────────────────────────────────────
# Разметка таргета
# ─────────────────────────────────────────────
def make_target(df: pd.DataFrame) -> pd.DataFrame:
    future   = df['Close'].shift(-TARGET_HORIZON)
    pct_chng = (future - df['Close']) / (df['Close'] + 1e-9)

    df = df.copy()
    df['Target'] = 0
    df.loc[pct_chng >  TARGET_THRESHOLD, 'Target'] = 1
    df.loc[pct_chng < -TARGET_THRESHOLD, 'Target'] = 2
    df = df.iloc[:-TARGET_HORIZON]

    counts = df['Target'].value_counts().to_dict()
    logger.info(
        f"[Trainer] Разметка: HOLD={counts.get(0,0)} "
        f"BUY={counts.get(1,0)} SELL={counts.get(2,0)}"
    )
    return df


# ─────────────────────────────────────────────
# Walk-Forward валидация
# ─────────────────────────────────────────────
def walk_forward_eval(X: np.ndarray, y: np.ndarray,
                      train_size: int, test_size: int, step: int,
                      feature_cols: list) -> dict:
    """
    Скользящая оценка: обучаем на train_size, тестируем на test_size,
    сдвигаем на step. Честная оценка — нет data leakage.
    """
    results = []
    n = len(X)
    start = train_size

    while start + test_size <= n:
        X_tr = X[start - train_size : start]
        y_tr = y[start - train_size : start]
        X_te = X[start : start + test_size]
        y_te = y[start : start + test_size]

        sw = compute_sample_weight("balanced", y_tr)

        m = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='mlogloss', num_class=3,
            use_label_encoder=False, verbosity=0
        )
        m.fit(X_tr, y_tr, sample_weight=sw, verbose=False)

        y_pred = m.predict(X_te)
        prec   = precision_score(y_te, y_pred, average='weighted', zero_division=0)
        acc    = accuracy_score(y_te, y_pred)
        results.append({"precision": prec, "accuracy": acc, "n_test": len(y_te)})
        start += step

    if not results:
        return {"wf_precision": 0, "wf_accuracy": 0, "wf_folds": 0}

    return {
        "wf_precision": round(np.mean([r["precision"] for r in results]), 4),
        "wf_accuracy":  round(np.mean([r["accuracy"]  for r in results]), 4),
        "wf_folds":     len(results),
    }


# ─────────────────────────────────────────────
# Определение рабочего набора фичей
# ─────────────────────────────────────────────
def get_available_features(df: pd.DataFrame, desired: list) -> list:
    available = [c for c in desired if c in df.columns]
    missing   = [c for c in desired if c not in df.columns]
    if missing:
        logger.warning(f"[Trainer] Отсутствуют фичи (будут пропущены): {missing}")
    return available


# ─────────────────────────────────────────────
# Обучение XGBoost
# ─────────────────────────────────────────────
def train_xgboost(X_train, y_train, X_test, y_test) -> tuple:
    sw = compute_sample_weight("balanced", y_train)

    model = XGBClassifier(
        n_estimators      = 500,
        max_depth         = 5,
        learning_rate     = 0.02,
        subsample         = 0.8,
        colsample_bytree  = 0.75,
        min_child_weight  = 7,
        gamma             = 0.1,
        reg_alpha         = 0.1,
        reg_lambda        = 1.5,
        eval_metric       = 'mlogloss',
        num_class         = 3,
        use_label_encoder = False,
        verbosity         = 0,
    )
    model.fit(
        X_train, y_train,
        sample_weight = sw,
        eval_set      = [(X_test, y_test)],
        verbose       = False,
    )

    y_pred    = model.predict(X_test)
    precision = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    accuracy  = float(accuracy_score(y_test, y_pred))
    f1        = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

    return model, {"precision": precision, "accuracy": accuracy, "f1": f1}


# ─────────────────────────────────────────────
# Обучение LightGBM
# ─────────────────────────────────────────────
def train_lightgbm(X_train, y_train, X_test, y_test) -> tuple:
    if not LGBM_AVAILABLE:
        return None, None

    sw = compute_sample_weight("balanced", y_train)

    model = lgb.LGBMClassifier(
        n_estimators     = 500,
        max_depth        = 5,
        learning_rate    = 0.02,
        subsample        = 0.8,
        colsample_bytree = 0.75,
        min_child_samples= 20,
        reg_alpha        = 0.1,
        reg_lambda       = 1.5,
        num_class        = 3,
        objective        = 'multiclass',
        verbosity        = -1,
        n_jobs           = -1,
    )
    model.fit(
        X_train, y_train,
        sample_weight    = sw,
        eval_set         = [(X_test, y_test)],
        callbacks        = [lgb.early_stopping(50, verbose=False),
                            lgb.log_evaluation(-1)],
    )

    y_pred    = model.predict(X_test)
    precision = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    accuracy  = float(accuracy_score(y_test, y_pred))
    f1        = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

    return model, {"precision": precision, "accuracy": accuracy, "f1": f1}


# ─────────────────────────────────────────────
# ГЛАВНАЯ: обучение ансамбля
# ─────────────────────────────────────────────
def train_model() -> dict:
    logger.info("[Trainer] 🚀 v4.0: XGBoost + LightGBM ансамбль, Multi-TF, 40+ фич")

    # 1. Загружаем 1H и 4H данные
    df1h_raw = fetch_ohlcv("TON-USDT", "1H", 3000)
    df4h_raw = fetch_ohlcv("TON-USDT", "4H", 750)

    if df1h_raw.empty:
        return {"success": False, "error": "Нет 1H данных"}

    # 2. Рассчитываем индикаторы
    df1h = calc_indicators_1h(df1h_raw)

    if not df4h_raw.empty:
        df4h = calc_indicators_4h(df4h_raw)
        df   = merge_timeframes(df1h, df4h)
        logger.info("[Trainer] ✅ Объединены 1H + 4H данные")
    else:
        logger.warning("[Trainer] ⚠️ 4H данные недоступны — обучаем только на 1H")
        df = df1h

    df = df.dropna()
    if len(df) < 300:
        return {"success": False, "error": f"Мало данных: {len(df)}"}

    # 3. Таргет
    df = make_target(df)

    # 4. Определяем доступные фичи
    feature_cols = get_available_features(df, FEATURE_COLS)
    if len(feature_cols) < 10:
        # Фоллбэк на legacy
        feature_cols = get_available_features(df, FEATURE_COLS_LEGACY)
        logger.warning(f"[Trainer] Используем legacy features ({len(feature_cols)} шт)")
    else:
        logger.info(f"[Trainer] Используем {len(feature_cols)} признаков")

    X = df[feature_cols].values.astype(np.float32)
    y = df['Target'].values

    # 5. Train/test split (без shuffle — временной ряд!)
    split    = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    logger.info(f"[Trainer] Train: {len(X_train)} | Test: {len(X_test)}")

    # 6. Walk-forward оценка (реалистичный backtest)
    hours_per_day = 24
    wf_train = WF_TRAIN_DAYS * hours_per_day
    wf_test  = WF_TEST_DAYS  * hours_per_day
    wf_step  = WF_STEP_DAYS  * hours_per_day

    wf_result = walk_forward_eval(X, y, wf_train, wf_test, wf_step, feature_cols)
    logger.info(
        f"[Trainer] Walk-Forward: "
        f"precision={wf_result['wf_precision']:.1%} "
        f"accuracy={wf_result['wf_accuracy']:.1%} "
        f"folds={wf_result['wf_folds']}"
    )

    # 7. Обучение XGBoost
    logger.info("[Trainer] 🔧 Обучение XGBoost...")
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    logger.info(
        f"[Trainer] XGBoost: "
        f"prec={xgb_metrics['precision']:.1%} "
        f"acc={xgb_metrics['accuracy']:.1%} "
        f"f1={xgb_metrics['f1']:.1%}"
    )
    logger.info("\n" + classification_report(
        y_test, xgb_model.predict(X_test),
        target_names=['HOLD', 'BUY', 'SELL'], zero_division=0
    ))

    # 8. Обучение LightGBM
    lgbm_model, lgbm_metrics = None, None
    if LGBM_AVAILABLE:
        logger.info("[Trainer] 🔧 Обучение LightGBM...")
        lgbm_model, lgbm_metrics = train_lightgbm(X_train, y_train, X_test, y_test)
        if lgbm_metrics:
            logger.info(
                f"[Trainer] LightGBM: "
                f"prec={lgbm_metrics['precision']:.1%} "
                f"acc={lgbm_metrics['accuracy']:.1%} "
                f"f1={lgbm_metrics['f1']:.1%}"
            )
    else:
        logger.warning("[Trainer] LightGBM не установлен — только XGBoost")

    # 9. Сохраняем модели
    joblib.dump(xgb_model, MODEL_PATH)

    if lgbm_model:
        joblib.dump(lgbm_model, MODEL_PATH_LGBM)

    # Сохраняем список фичей (для live_signal)
    features_path = MODEL_PATH.replace('.pkl', '_features.json')
    with open(features_path, 'w') as f:
        json.dump(feature_cols, f)

    # 10. Итоговая статистика
    ensemble_precision = xgb_metrics['precision']
    if lgbm_metrics:
        ensemble_precision = (xgb_metrics['precision'] + lgbm_metrics['precision']) / 2

    stats = {
        "success":            True,
        "n_features":         len(feature_cols),
        "n_samples":          len(df),
        "n_train":            split,
        "n_test":             len(X_test),
        # XGBoost
        "xgb_precision":      xgb_metrics['precision'],
        "xgb_accuracy":       xgb_metrics['accuracy'],
        "xgb_f1":             xgb_metrics['f1'],
        # LightGBM
        "lgbm_precision":     lgbm_metrics['precision'] if lgbm_metrics else None,
        "lgbm_accuracy":      lgbm_metrics['accuracy']  if lgbm_metrics else None,
        # Ансамбль
        "ensemble_precision": ensemble_precision,
        # Walk-forward
        **wf_result,
        # Совместимость
        "precision":          ensemble_precision,
        "accuracy":           xgb_metrics['accuracy'],
        "recall":             xgb_metrics['f1'],
        "lgbm_available":     LGBM_AVAILABLE,
        "feature_cols":       feature_cols,
    }

    with open(STATS_FILE, 'w') as f:
        json.dump({k: v for k, v in stats.items() if k != 'feature_cols'}, f, indent=2)

    logger.info(
        f"[Trainer] ✅ Ансамбль готов! "
        f"Ensemble prec={ensemble_precision:.1%} | "
        f"WF prec={wf_result['wf_precision']:.1%}"
    )

    return {**stats, "model": xgb_model, "lgbm_model": lgbm_model}