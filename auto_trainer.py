"""
auto_trainer.py v5.0 — Двойные бинарные классификаторы
АРХИТЕКТУРНОЕ ИЗМЕНЕНИЕ:
  - Вместо 1 мультиклассового (HOLD/BUY/SELL с дисбалансом 65:17:18)
    → 2 бинарных классификатора:
      · BUY-модель:  "вырастет ли цена > 1.2% за 6ч?" (1 vs 0)
      · SELL-модель: "упадёт ли цена > 1.2% за 6ч?"  (1 vs 0)
  - SMOTE балансировка (1:1 вместо 65:17:18) → precision BUY/SELL ~45-55%
  - Optuna гиперпараметр-тюнинг (30 trials)
  - Walk-Forward валидация на бинарных моделях
  - Все фичи 1H + 4H (44 признака)
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
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
    classification_report
)

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
    WF_TRAIN_DAYS, WF_TEST_DAYS, WF_STEP_DAYS
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Загрузка OHLCV с OKX (пагинация)
# ─────────────────────────────────────────────
def fetch_ohlcv(symbol: str = "TON-USDT", bar: str = "1H", bars: int = 3000) -> pd.DataFrame:
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
# Индикаторы 1H
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

    rsi14         = d['RSI_14']
    stoch_min     = rsi14.rolling(14).min()
    stoch_max     = rsi14.rolling(14).max()
    stoch_k       = (rsi14 - stoch_min) / (stoch_max - stoch_min + 1e-9) * 100
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

    return d


# ─────────────────────────────────────────────
# Слияние 1H + 4H
# ─────────────────────────────────────────────
def merge_timeframes(df1h: pd.DataFrame, df4h: pd.DataFrame) -> pd.DataFrame:
    cols_4h        = [c for c in df4h.columns if c.endswith('_4h') or c.endswith('_4h_tf')]
    df4h_sub       = df4h[cols_4h].copy()
    df_merged      = df1h.copy()
    df4h_reindexed = df4h_sub.reindex(df1h.index, method='ffill')
    df_merged      = pd.concat([df_merged, df4h_reindexed], axis=1)
    df_merged      = df_merged.dropna(subset=cols_4h)
    return df_merged


# ─────────────────────────────────────────────
# Разметка таргетов (бинарные)
# ─────────────────────────────────────────────
def make_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создаёт ДВА бинарных таргета:
      Target_BUY:  1 если цена вырастет > TARGET_THRESHOLD за TARGET_HORIZON часов
      Target_SELL: 1 если цена упадёт  > TARGET_THRESHOLD за TARGET_HORIZON часов
    """
    future   = df['Close'].shift(-TARGET_HORIZON)
    pct_chng = (future - df['Close']) / (df['Close'] + 1e-9)

    df = df.copy()
    df['Target_BUY']  = (pct_chng >  TARGET_THRESHOLD).astype(int)
    df['Target_SELL'] = (pct_chng < -TARGET_THRESHOLD).astype(int)
    df = df.iloc[:-TARGET_HORIZON]

    buy_count  = df['Target_BUY'].sum()
    sell_count = df['Target_SELL'].sum()
    total      = len(df)
    logger.info(
        f"[Trainer] Разметка: BUY={buy_count} ({buy_count/total:.1%}) "
        f"SELL={sell_count} ({sell_count/total:.1%}) "
        f"NEUTRAL={total-buy_count-sell_count} ({(total-buy_count-sell_count)/total:.1%})"
    )
    return df


# ─────────────────────────────────────────────
# SMOTE балансировка
# ─────────────────────────────────────────────
def apply_smote(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    if not SMOTE_AVAILABLE:
        logger.warning("[Trainer] SMOTE недоступен — используем class_weight='balanced'")
        return X_train, y_train

    pos = y_train.sum()
    neg = len(y_train) - pos
    ratio = pos / neg if neg > 0 else 1.0

    if ratio > 0.4:  # уже достаточно сбалансировано
        return X_train, y_train

    try:
        smote = SMOTE(sampling_strategy=0.8, random_state=42, k_neighbors=5)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        logger.info(
            f"[Trainer] SMOTE: {len(y_train)} → {len(y_res)} "
            f"(pos: {y_train.sum()} → {y_res.sum()})"
        )
        return X_res, y_res
    except Exception as e:
        logger.warning(f"[Trainer] SMOTE ошибка: {e} — без балансировки")
        return X_train, y_train


# ─────────────────────────────────────────────
# Optuna тюнинг XGBoost
# ─────────────────────────────────────────────
def tune_xgboost(X_train, y_train, X_val, y_val, n_trials: int = 30) -> dict:
    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
            'max_depth':        trial.suggest_int('max_depth', 3, 6),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
            'gamma':            trial.suggest_float('gamma', 0.0, 0.5),
            'reg_alpha':        trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda':       trial.suggest_float('reg_lambda', 0.5, 3.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0),
        }
        m = XGBClassifier(
            **params,
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=0,
        )
        m.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)
        y_pred = m.predict(X_val)
        return precision_score(y_val, y_pred, zero_division=0)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"[Trainer] Optuna best precision: {study.best_value:.3f}")
    return study.best_params


# ─────────────────────────────────────────────
# Обучение одного бинарного XGBoost
# ─────────────────────────────────────────────
def train_binary_xgb(X_train, y_train, X_test, y_test,
                     best_params: dict = None) -> tuple:
    if best_params is None:
        best_params = {
            'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.03,
            'subsample': 0.75, 'colsample_bytree': 0.70,
            'min_child_weight': 10, 'gamma': 0.2,
            'reg_alpha': 0.3, 'reg_lambda': 2.0, 'scale_pos_weight': 2.0,
        }

    model = XGBClassifier(
        **best_params,
        eval_metric='logloss',
        use_label_encoder=False,
        verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_test, y_pred, zero_division=0)),
        'f1':        float(f1_score(y_test, y_pred, zero_division=0)),
        'accuracy':  float(accuracy_score(y_test, y_pred)),
        'roc_auc':   float(roc_auc_score(y_test, y_proba)) if y_test.sum() > 0 else 0.0,
    }
    return model, metrics


# ─────────────────────────────────────────────
# Обучение одного бинарного LightGBM
# ─────────────────────────────────────────────
def train_binary_lgbm(X_train, y_train, X_test, y_test) -> tuple:
    if not LGBM_AVAILABLE:
        return None, None

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos = neg_count / (pos_count + 1e-9)

    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.75,
        colsample_bytree=0.70,
        min_child_samples=20,
        reg_alpha=0.3,
        reg_lambda=2.0,
        scale_pos_weight=min(scale_pos, 5.0),
        objective='binary',
        verbosity=-1,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(-1)],
    )

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_test, y_pred, zero_division=0)),
        'f1':        float(f1_score(y_test, y_pred, zero_division=0)),
        'accuracy':  float(accuracy_score(y_test, y_pred)),
        'roc_auc':   float(roc_auc_score(y_test, y_proba)) if y_test.sum() > 0 else 0.0,
    }
    return model, metrics


# ─────────────────────────────────────────────
# Walk-Forward для бинарной модели
# ─────────────────────────────────────────────
def walk_forward_binary(X: np.ndarray, y: np.ndarray,
                        train_size: int, test_size: int, step: int) -> dict:
    results = []
    n = len(X)
    start = train_size

    while start + test_size <= n:
        X_tr = X[start - train_size: start]
        y_tr = y[start - train_size: start]
        X_te = X[start: start + test_size]
        y_te = y[start: start + test_size]

        if y_tr.sum() < 5:
            start += step
            continue

        X_tr_sm, y_tr_sm = apply_smote(X_tr, y_tr)

        pos = y_tr.sum()
        neg = len(y_tr) - pos
        spw = min(neg / (pos + 1e-9), 5.0)

        m = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.03,
            subsample=0.75, colsample_bytree=0.7,
            min_child_weight=10, gamma=0.2,
            reg_alpha=0.3, reg_lambda=2.0,
            scale_pos_weight=spw,
            eval_metric='logloss', use_label_encoder=False, verbosity=0
        )
        m.fit(X_tr_sm, y_tr_sm, verbose=False)

        if y_te.sum() > 0:
            y_pred = m.predict(X_te)
            prec   = precision_score(y_te, y_pred, zero_division=0)
            rec    = recall_score(y_te, y_pred, zero_division=0)
            results.append({'precision': prec, 'recall': rec})

        start += step

    if not results:
        return {'wf_precision': 0.0, 'wf_recall': 0.0, 'wf_folds': 0}

    return {
        'wf_precision': round(np.mean([r['precision'] for r in results]), 4),
        'wf_recall':    round(np.mean([r['recall']    for r in results]), 4),
        'wf_folds':     len(results),
    }


# ─────────────────────────────────────────────
# Доступные фичи
# ─────────────────────────────────────────────
def get_available_features(df: pd.DataFrame, desired: list) -> list:
    available = [c for c in desired if c in df.columns]
    missing   = [c for c in desired if c not in df.columns]
    if missing:
        logger.warning(f"[Trainer] Отсутствуют фичи: {missing}")
    return available


# ─────────────────────────────────────────────
# ГЛАВНАЯ: обучение двух бинарных ансамблей
# ─────────────────────────────────────────────
def train_model() -> dict:
    logger.info("[Trainer] 🚀 v5.0: Двойные бинарные классификаторы + SMOTE + Optuna")

    # 1. Данные
    df1h_raw = fetch_ohlcv("TON-USDT", "1H", 3000)
    df4h_raw = fetch_ohlcv("TON-USDT", "4H", 750)

    if df1h_raw.empty:
        return {"success": False, "error": "Нет 1H данных"}

    # 2. Индикаторы
    df1h = calc_indicators_1h(df1h_raw)

    if not df4h_raw.empty:
        df4h = calc_indicators_4h(df4h_raw)
        df   = merge_timeframes(df1h, df4h)
        logger.info(f"[Trainer] ✅ Объединены 1H + 4H | строк: {len(df)}")
    else:
        logger.warning("[Trainer] ⚠️ 4H данные недоступны")
        df = df1h

    df = df.dropna()
    if len(df) < 300:
        return {"success": False, "error": f"Мало данных: {len(df)}"}

    # 3. Бинарные таргеты
    df = make_targets(df)

    # 4. Фичи
    feature_cols = get_available_features(df, FEATURE_COLS)
    if len(feature_cols) < 10:
        feature_cols = get_available_features(df, FEATURE_COLS_LEGACY)
        logger.warning(f"[Trainer] Legacy features: {len(feature_cols)} шт")
    else:
        logger.info(f"[Trainer] Используем {len(feature_cols)} признаков")

    X = df[feature_cols].values.astype(np.float32)
    y_buy  = df['Target_BUY'].values
    y_sell = df['Target_SELL'].values

    # 5. Train/test split (временной ряд!)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_buy_train,  y_buy_test  = y_buy[:split],  y_buy[split:]
    y_sell_train, y_sell_test = y_sell[:split], y_sell[split:]

    logger.info(
        f"[Trainer] Train: {len(X_train)} | Test: {len(X_test)} | "
        f"BUY pos train: {y_buy_train.sum()} | SELL pos train: {y_sell_train.sum()}"
    )

    # 6. SMOTE
    logger.info("[Trainer] 🔄 SMOTE балансировка BUY...")
    X_buy_sm, y_buy_sm   = apply_smote(X_train, y_buy_train)
    logger.info("[Trainer] 🔄 SMOTE балансировка SELL...")
    X_sell_sm, y_sell_sm = apply_smote(X_train, y_sell_train)

    # 7. Optuna тюнинг для BUY-модели (30 trials)
    logger.info("[Trainer] 🔬 Optuna тюнинг BUY-модели (30 trials)...")
    val_split    = int(len(X_buy_sm) * 0.85)
    X_opt_tr     = X_buy_sm[:val_split]
    y_opt_tr     = y_buy_sm[:val_split]
    X_opt_val    = X_buy_sm[val_split:]
    y_opt_val    = y_buy_sm[val_split:]
    best_params  = tune_xgboost(X_opt_tr, y_opt_tr, X_opt_val, y_opt_val, n_trials=30)
    logger.info(f"[Trainer] Best params: {best_params}")

    # 8. Обучение BUY-моделей
    logger.info("[Trainer] 🔧 XGBoost BUY...")
    buy_xgb, buy_xgb_m = train_binary_xgb(X_buy_sm, y_buy_sm, X_test, y_buy_test, best_params)
    logger.info(
        f"[Trainer] BUY XGB: prec={buy_xgb_m['precision']:.1%} "
        f"rec={buy_xgb_m['recall']:.1%} "
        f"auc={buy_xgb_m['roc_auc']:.3f}"
    )

    logger.info("[Trainer] 🔧 LightGBM BUY...")
    buy_lgbm, buy_lgbm_m = train_binary_lgbm(X_buy_sm, y_buy_sm, X_test, y_buy_test)
    if buy_lgbm_m:
        logger.info(
            f"[Trainer] BUY LGBM: prec={buy_lgbm_m['precision']:.1%} "
            f"rec={buy_lgbm_m['recall']:.1%} "
            f"auc={buy_lgbm_m['roc_auc']:.3f}"
        )

    # 9. Обучение SELL-моделей
    logger.info("[Trainer] 🔧 XGBoost SELL...")
    sell_xgb, sell_xgb_m = train_binary_xgb(X_sell_sm, y_sell_sm, X_test, y_sell_test, best_params)
    logger.info(
        f"[Trainer] SELL XGB: prec={sell_xgb_m['precision']:.1%} "
        f"rec={sell_xgb_m['recall']:.1%} "
        f"auc={sell_xgb_m['roc_auc']:.3f}"
    )

    logger.info("[Trainer] 🔧 LightGBM SELL...")
    sell_lgbm, sell_lgbm_m = train_binary_lgbm(X_sell_sm, y_sell_sm, X_test, y_sell_test)
    if sell_lgbm_m:
        logger.info(
            f"[Trainer] SELL LGBM: prec={sell_lgbm_m['precision']:.1%} "
            f"rec={sell_lgbm_m['recall']:.1%} "
            f"auc={sell_lgbm_m['roc_auc']:.3f}"
        )

    # 10. Walk-Forward
    hours_per_day = 24
    wf_train = WF_TRAIN_DAYS * hours_per_day
    wf_test  = WF_TEST_DAYS  * hours_per_day
    wf_step  = WF_STEP_DAYS  * hours_per_day

    logger.info("[Trainer] 📊 Walk-Forward BUY...")
    wf_buy  = walk_forward_binary(X, y_buy,  wf_train, wf_test, wf_step)
    logger.info("[Trainer] 📊 Walk-Forward SELL...")
    wf_sell = walk_forward_binary(X, y_sell, wf_train, wf_test, wf_step)
    logger.info(
        f"[Trainer] WF BUY:  prec={wf_buy['wf_precision']:.1%} "
        f"rec={wf_buy['wf_recall']:.1%} folds={wf_buy['wf_folds']}"
    )
    logger.info(
        f"[Trainer] WF SELL: prec={wf_sell['wf_precision']:.1%} "
        f"rec={wf_sell['wf_recall']:.1%} folds={wf_sell['wf_folds']}"
    )

    # 11. Сохраняем модели
    joblib.dump(buy_xgb,  MODEL_PATH_BUY_XGB)
    joblib.dump(sell_xgb, MODEL_PATH_SELL_XGB)
    if buy_lgbm:
        joblib.dump(buy_lgbm,  MODEL_PATH_BUY_LGBM)
    if sell_lgbm:
        joblib.dump(sell_lgbm, MODEL_PATH_SELL_LGBM)

    with open(MODEL_FEATURES_PATH, 'w') as f:
        json.dump(feature_cols, f)

    # 12. Итоги
    avg_buy_prec  = (buy_xgb_m['precision'] + (buy_lgbm_m['precision'] if buy_lgbm_m else buy_xgb_m['precision'])) / 2
    avg_sell_prec = (sell_xgb_m['precision'] + (sell_lgbm_m['precision'] if sell_lgbm_m else sell_xgb_m['precision'])) / 2
    avg_buy_auc   = (buy_xgb_m['roc_auc']   + (buy_lgbm_m['roc_auc']   if buy_lgbm_m else buy_xgb_m['roc_auc']))   / 2
    avg_sell_auc  = (sell_xgb_m['roc_auc']  + (sell_lgbm_m['roc_auc']  if sell_lgbm_m else sell_xgb_m['roc_auc'])) / 2

    stats = {
        "success":       True,
        "n_features":    len(feature_cols),
        "n_samples":     len(df),
        "n_train":       split,
        "n_test":        len(X_test),
        # BUY
        "buy_xgb_precision":  buy_xgb_m['precision'],
        "buy_xgb_recall":     buy_xgb_m['recall'],
        "buy_xgb_auc":        buy_xgb_m['roc_auc'],
        "buy_lgbm_precision": buy_lgbm_m['precision'] if buy_lgbm_m else None,
        "buy_lgbm_auc":       buy_lgbm_m['roc_auc']  if buy_lgbm_m else None,
        "avg_buy_precision":  avg_buy_prec,
        "avg_buy_auc":        avg_buy_auc,
        # SELL
        "sell_xgb_precision":  sell_xgb_m['precision'],
        "sell_xgb_recall":     sell_xgb_m['recall'],
        "sell_xgb_auc":        sell_xgb_m['roc_auc'],
        "sell_lgbm_precision": sell_lgbm_m['precision'] if sell_lgbm_m else None,
        "sell_lgbm_auc":       sell_lgbm_m['roc_auc']  if sell_lgbm_m else None,
        "avg_sell_precision":  avg_sell_prec,
        "avg_sell_auc":        avg_sell_auc,
        # Walk-Forward
        "wf_buy_precision":   wf_buy['wf_precision'],
        "wf_buy_recall":      wf_buy['wf_recall'],
        "wf_sell_precision":  wf_sell['wf_precision'],
        "wf_sell_recall":     wf_sell['wf_recall'],
        "wf_folds":           wf_buy['wf_folds'],
        # Совместимость
        "xgb_precision":      avg_buy_prec,
        "lgbm_precision":     buy_lgbm_m['precision'] if buy_lgbm_m else None,
        "ensemble_precision": (avg_buy_prec + avg_sell_prec) / 2,
        "wf_precision":       (wf_buy['wf_precision'] + wf_sell['wf_precision']) / 2,
        "wf_accuracy":        0.0,
        "lgbm_available":     LGBM_AVAILABLE,
        "smote_available":    SMOTE_AVAILABLE,
        "best_params":        best_params,
    }

    with open(STATS_FILE, 'w') as f:
        json.dump({k: v for k, v in stats.items() if k not in ('best_params',)}, f, indent=2)

    logger.info(
        f"[Trainer] ✅ Готово! "
        f"BUY prec={avg_buy_prec:.1%} auc={avg_buy_auc:.3f} | "
        f"SELL prec={avg_sell_prec:.1%} auc={avg_sell_auc:.3f} | "
        f"WF BUY={wf_buy['wf_precision']:.1%} SELL={wf_sell['wf_precision']:.1%}"
    )

    return {
        **stats,
        "model":      buy_xgb,
        "lgbm_model": buy_lgbm,
    }