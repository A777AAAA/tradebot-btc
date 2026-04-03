"""
auto_trainer.py v6.0 — Triple Barrier Labeling + Meta-Labeling
ИЗМЕНЕНИЯ v6.0:
  - Triple Barrier Method (по Лопесу де Прадо):
      · Таргет = 1 если TP сработал раньше SL (а не просто "цена выросла")
      · Таргет = 0 если SL сработал раньше TP
      · Исключаем свечи где ни TP ни SL не сработали за горизонт
    → Разметка теперь соответствует реальной торговле бота
  - Meta-labeling фильтр:
      · Первая модель (side model) генерирует направление BUY/SELL
      · Вторая модель (meta model) фильтрует: входить или нет
      → Снижает ложные срабатывания без потери хороших сигналов
  - Правильный Sharpe в walk-forward (на почасовых доходностях)
  - SMOTE + Optuna сохранены
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
    WF_TRAIN_DAYS, WF_TEST_DAYS, WF_STEP_DAYS,
    ATR_SL_MULT, ATR_TP_MULT,
)

logger = logging.getLogger(__name__)

META_MODEL_BUY_PATH  = "meta_model_buy.pkl"
META_MODEL_SELL_PATH = "meta_model_sell.pkl"


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
# TRIPLE BARRIER LABELING (главное улучшение)
# ─────────────────────────────────────────────
def triple_barrier_labels(df: pd.DataFrame,
                           horizon: int = None,
                           tp_mult: float = None,
                           sl_mult: float = None) -> pd.DataFrame:
    """
    Triple Barrier Method по Лопесу де Прадо.

    Для каждой свечи i смотрим вперёд на horizon свечей:
      - Если цена достигла TP (entry + atr*tp_mult) раньше SL → target_buy = 1
      - Если цена достигла SL (entry - atr*sl_mult) раньше TP → target_buy = 0
      - Если ни то ни другое за horizon → метка = NaN (исключаем из обучения)

    Аналогично для SELL (инвертированные барьеры).

    Это ключевое отличие от простого "цена через N часов":
    простой метод помечает как WIN даже если по пути цена сначала
    выбила стоп (что в реальной торговле = убыток).
    """
    if horizon is None:
        horizon = TARGET_HORIZON
    if tp_mult is None:
        tp_mult = ATR_TP_MULT
    if sl_mult is None:
        sl_mult = ATR_SL_MULT

    close  = df['Close'].values
    atr    = df['ATR'].values
    high   = df['High'].values
    low    = df['Low'].values
    n      = len(df)

    target_buy  = np.full(n, np.nan)
    target_sell = np.full(n, np.nan)

    for i in range(n - horizon):
        entry  = close[i]
        atr_i  = atr[i]

        tp_buy = entry + atr_i * tp_mult
        sl_buy = entry - atr_i * sl_mult

        tp_sell = entry - atr_i * tp_mult
        sl_sell = entry + atr_i * sl_mult

        buy_result  = np.nan
        sell_result = np.nan

        for j in range(i + 1, min(i + horizon + 1, n)):
            h = high[j]
            l = low[j]

            # BUY барьеры
            if np.isnan(buy_result):
                if h >= tp_buy and l <= sl_buy:
                    # обе зоны в одной свече — смотрим на открытие
                    buy_result = 1 if close[j-1] < entry + atr_i * 0.5 else 0
                elif h >= tp_buy:
                    buy_result = 1
                elif l <= sl_buy:
                    buy_result = 0

            # SELL барьеры
            if np.isnan(sell_result):
                if l <= tp_sell and h >= sl_sell:
                    sell_result = 1 if close[j-1] > entry - atr_i * 0.5 else 0
                elif l <= tp_sell:
                    sell_result = 1
                elif h >= sl_sell:
                    sell_result = 0

            if not np.isnan(buy_result) and not np.isnan(sell_result):
                break

        target_buy[i]  = buy_result
        target_sell[i] = sell_result

    df = df.copy()
    df['Target_BUY']  = target_buy
    df['Target_SELL'] = target_sell

    total     = n - horizon
    buy_valid = int(np.sum(~np.isnan(target_buy[:total])))
    buy_pos   = int(np.nansum(target_buy[:total]))
    sel_valid = int(np.sum(~np.isnan(target_sell[:total])))
    sel_pos   = int(np.nansum(target_sell[:total]))

    logger.info(
        f"[Trainer] Triple Barrier: "
        f"BUY valid={buy_valid} pos={buy_pos} ({buy_pos/(buy_valid+1e-9):.1%}) | "
        f"SELL valid={sel_valid} pos={sel_pos} ({sel_pos/(sel_valid+1e-9):.1%})"
    )
    return df


# ─────────────────────────────────────────────
# SMOTE балансировка
# ─────────────────────────────────────────────
def apply_smote(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    if not SMOTE_AVAILABLE:
        return X_train, y_train

    pos = y_train.sum()
    neg = len(y_train) - pos
    ratio = pos / (neg + 1e-9)

    if ratio > 0.4:
        return X_train, y_train

    try:
        smote = SMOTE(sampling_strategy=0.8, random_state=42, k_neighbors=5)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        logger.info(
            f"[Trainer] SMOTE: {len(y_train)} → {len(y_res)} "
            f"(pos: {int(y_train.sum())} → {int(y_res.sum())})"
        )
        return X_res, y_res
    except Exception as e:
        logger.warning(f"[Trainer] SMOTE ошибка: {e}")
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
            **params, eval_metric='logloss',
            use_label_encoder=False, verbosity=0,
        )
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = m.predict(X_val)
        return precision_score(y_val, y_pred, zero_division=0)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"[Trainer] Optuna best precision: {study.best_value:.3f}")
    return study.best_params


# ─────────────────────────────────────────────
# Обучение XGBoost
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
        **best_params, eval_metric='logloss',
        use_label_encoder=False, verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

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
# Обучение LightGBM
# ─────────────────────────────────────────────
def train_binary_lgbm(X_train, y_train, X_test, y_test) -> tuple:
    if not LGBM_AVAILABLE:
        return None, None

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos = neg_count / (pos_count + 1e-9)

    model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.75, colsample_bytree=0.70, min_child_samples=20,
        reg_alpha=0.3, reg_lambda=2.0,
        scale_pos_weight=min(scale_pos, 5.0),
        objective='binary', verbosity=-1, n_jobs=-1,
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
# META-LABELING (фильтрующая модель)
# ─────────────────────────────────────────────
def train_meta_model(X_train: np.ndarray, y_side_train: np.ndarray,
                     y_true_train: np.ndarray,
                     X_test: np.ndarray, y_side_test: np.ndarray,
                     y_true_test: np.ndarray,
                     side_model) -> tuple:
    """
    Meta-labeling: обучаем вторую модель которая решает
    'стоит ли входить по сигналу первой модели?'

    Вход для мета-модели = оригинальные фичи + вероятность от side_model
    Таргет = 1 если side_model был прав (и сделка выиграла)
           = 0 если side_model ошибся

    Это позволяет бету быть более избирательным:
    торговать только когда оба уровня согласны.
    """
    # Получаем вероятности от side_model
    p_train = side_model.predict_proba(X_train)[:, 1].reshape(-1, 1)
    p_test  = side_model.predict_proba(X_test)[:, 1].reshape(-1, 1)

    # Добавляем вероятность как дополнительную фичу
    X_meta_train = np.hstack([X_train, p_train])
    X_meta_test  = np.hstack([X_test,  p_test])

    # Таргет мета-модели: правильно ли предсказала side_model?
    side_pred_train = (p_train.flatten() >= 0.5).astype(int)
    side_pred_test  = (p_test.flatten()  >= 0.5).astype(int)

    y_meta_train = ((side_pred_train == 1) & (y_true_train == 1)).astype(int)
    y_meta_test  = ((side_pred_test  == 1) & (y_true_test  == 1)).astype(int)

    if y_meta_train.sum() < 10:
        logger.warning("[Trainer] Meta-model: мало позитивных примеров, пропускаем")
        return None, None

    X_meta_sm, y_meta_sm = apply_smote(X_meta_train, y_meta_train)

    meta_model = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.75, colsample_bytree=0.7,
        min_child_weight=10, gamma=0.3,
        eval_metric='logloss', use_label_encoder=False, verbosity=0,
    )
    meta_model.fit(
        X_meta_sm, y_meta_sm,
        eval_set=[(X_meta_test, y_meta_test)],
        verbose=False,
    )

    y_pred  = meta_model.predict(X_meta_test)
    y_proba = meta_model.predict_proba(X_meta_test)[:, 1]

    metrics = {
        'precision': float(precision_score(y_meta_test, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_meta_test, y_pred, zero_division=0)),
        'roc_auc':   float(roc_auc_score(y_meta_test, y_proba)) if y_meta_test.sum() > 0 else 0.0,
    }
    logger.info(
        f"[Trainer] Meta-model: prec={metrics['precision']:.1%} "
        f"rec={metrics['recall']:.1%} auc={metrics['roc_auc']:.3f}"
    )
    return meta_model, metrics


# ─────────────────────────────────────────────
# Walk-Forward с правильным Sharpe
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

        if y_tr.sum() < 5 or y_te.sum() < 3:
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

        y_pred   = m.predict(X_te)
        y_proba  = m.predict_proba(X_te)[:, 1]
        prec     = precision_score(y_te, y_pred, zero_division=0)
        rec      = recall_score(y_te, y_pred, zero_division=0)

        # Правильный Sharpe: симулируем почасовые доходности
        # +TP_pct если предсказали 1 и было 1 (win)
        # -SL_pct если предсказали 1 и было 0 (loss)
        # 0 если предсказали 0 (не входим)
        tp_pct = ATR_TP_MULT * 0.015  # примерный % от ATR
        sl_pct = ATR_SL_MULT * 0.015
        hourly_returns = []
        for pred, true in zip(y_pred, y_te):
            if pred == 1:
                hourly_returns.append(tp_pct if true == 1 else -sl_pct)
            else:
                hourly_returns.append(0.0)

        r = np.array(hourly_returns)
        sharpe = float(r.mean() / (r.std() + 1e-9) * np.sqrt(8760)) if r.std() > 0 else 0.0

        results.append({'precision': prec, 'recall': rec, 'sharpe': sharpe})
        start += step

    if not results:
        return {'wf_precision': 0.0, 'wf_recall': 0.0, 'wf_sharpe': 0.0, 'wf_folds': 0}

    return {
        'wf_precision': round(float(np.mean([r['precision'] for r in results])), 4),
        'wf_recall':    round(float(np.mean([r['recall']    for r in results])), 4),
        'wf_sharpe':    round(float(np.mean([r['sharpe']    for r in results])), 3),
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
# ГЛАВНАЯ: обучение с Triple Barrier + Meta-Label
# ─────────────────────────────────────────────
def train_model() -> dict:
    logger.info("[Trainer] 🚀 v6.0: Triple Barrier + Meta-Labeling + Правильный Sharpe")

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

    # 3. TRIPLE BARRIER разметка
    logger.info("[Trainer] 🎯 Triple Barrier разметка...")
    df = triple_barrier_labels(df)

    # 4. Убираем строки без метки (горизонт не достигнут)
    df_buy  = df[~df['Target_BUY'].isna()].copy()
    df_sell = df[~df['Target_SELL'].isna()].copy()

    # 5. Фичи
    feature_cols = get_available_features(df, FEATURE_COLS)
    if len(feature_cols) < 10:
        feature_cols = get_available_features(df, FEATURE_COLS_LEGACY)
        logger.warning(f"[Trainer] Legacy features: {len(feature_cols)} шт")
    else:
        logger.info(f"[Trainer] Используем {len(feature_cols)} признаков")

    X_buy  = df_buy[feature_cols].values.astype(np.float32)
    y_buy  = df_buy['Target_BUY'].values.astype(int)
    X_sell = df_sell[feature_cols].values.astype(np.float32)
    y_sell = df_sell['Target_SELL'].values.astype(int)

    # 6. Train/test split
    split_buy  = int(len(X_buy)  * 0.8)
    split_sell = int(len(X_sell) * 0.8)

    X_buy_train,  X_buy_test  = X_buy[:split_buy],   X_buy[split_buy:]
    y_buy_train,  y_buy_test  = y_buy[:split_buy],   y_buy[split_buy:]
    X_sell_train, X_sell_test = X_sell[:split_sell], X_sell[split_sell:]
    y_sell_train, y_sell_test = y_sell[:split_sell], y_sell[split_sell:]

    logger.info(
        f"[Trainer] BUY train={len(X_buy_train)} pos={y_buy_train.sum()} | "
        f"SELL train={len(X_sell_train)} pos={y_sell_train.sum()}"
    )

    # 7. SMOTE
    logger.info("[Trainer] 🔄 SMOTE BUY...")
    X_buy_sm,  y_buy_sm  = apply_smote(X_buy_train,  y_buy_train)
    logger.info("[Trainer] 🔄 SMOTE SELL...")
    X_sell_sm, y_sell_sm = apply_smote(X_sell_train, y_sell_train)

    # 8. Optuna
    logger.info("[Trainer] 🔬 Optuna тюнинг BUY (30 trials)...")
    val_split   = int(len(X_buy_sm) * 0.85)
    best_params = tune_xgboost(
        X_buy_sm[:val_split], y_buy_sm[:val_split],
        X_buy_sm[val_split:], y_buy_sm[val_split:],
        n_trials=30
    )

    # 9. BUY модели
    logger.info("[Trainer] 🔧 XGBoost BUY...")
    buy_xgb, buy_xgb_m = train_binary_xgb(X_buy_sm, y_buy_sm, X_buy_test, y_buy_test, best_params)
    logger.info(f"[Trainer] BUY XGB: prec={buy_xgb_m['precision']:.1%} auc={buy_xgb_m['roc_auc']:.3f}")

    logger.info("[Trainer] 🔧 LightGBM BUY...")
    buy_lgbm, buy_lgbm_m = train_binary_lgbm(X_buy_sm, y_buy_sm, X_buy_test, y_buy_test)
    if buy_lgbm_m:
        logger.info(f"[Trainer] BUY LGBM: prec={buy_lgbm_m['precision']:.1%} auc={buy_lgbm_m['roc_auc']:.3f}")

    # 10. SELL модели
    logger.info("[Trainer] 🔧 XGBoost SELL...")
    sell_xgb, sell_xgb_m = train_binary_xgb(X_sell_sm, y_sell_sm, X_sell_test, y_sell_test, best_params)
    logger.info(f"[Trainer] SELL XGB: prec={sell_xgb_m['precision']:.1%} auc={sell_xgb_m['roc_auc']:.3f}")

    logger.info("[Trainer] 🔧 LightGBM SELL...")
    sell_lgbm, sell_lgbm_m = train_binary_lgbm(X_sell_sm, y_sell_sm, X_sell_test, y_sell_test)
    if sell_lgbm_m:
        logger.info(f"[Trainer] SELL LGBM: prec={sell_lgbm_m['precision']:.1%} auc={sell_lgbm_m['roc_auc']:.3f}")

    # 11. META-LABELING
    logger.info("[Trainer] 🧩 Meta-model BUY...")
    meta_buy, meta_buy_m = train_meta_model(
        X_buy_train, y_buy_train, y_buy_train,
        X_buy_test,  y_buy_test,  y_buy_test,
        buy_xgb
    )

    logger.info("[Trainer] 🧩 Meta-model SELL...")
    meta_sell, meta_sell_m = train_meta_model(
        X_sell_train, y_sell_train, y_sell_train,
        X_sell_test,  y_sell_test,  y_sell_test,
        sell_xgb
    )

    # 12. Walk-Forward
    hours_per_day = 24
    wf_train = WF_TRAIN_DAYS * hours_per_day
    wf_test  = WF_TEST_DAYS  * hours_per_day
    wf_step  = WF_STEP_DAYS  * hours_per_day

    logger.info("[Trainer] 📊 Walk-Forward BUY...")
    wf_buy  = walk_forward_binary(X_buy,  y_buy,  wf_train, wf_test, wf_step)
    logger.info("[Trainer] 📊 Walk-Forward SELL...")
    wf_sell = walk_forward_binary(X_sell, y_sell, wf_train, wf_test, wf_step)
    logger.info(
        f"[Trainer] WF BUY:  prec={wf_buy['wf_precision']:.1%} "
        f"sharpe={wf_buy['wf_sharpe']:.2f} folds={wf_buy['wf_folds']}"
    )
    logger.info(
        f"[Trainer] WF SELL: prec={wf_sell['wf_precision']:.1%} "
        f"sharpe={wf_sell['wf_sharpe']:.2f} folds={wf_sell['wf_folds']}"
    )

    # 13. Сохраняем модели
    joblib.dump(buy_xgb,  MODEL_PATH_BUY_XGB)
    joblib.dump(sell_xgb, MODEL_PATH_SELL_XGB)
    if buy_lgbm:
        joblib.dump(buy_lgbm,  MODEL_PATH_BUY_LGBM)
    if sell_lgbm:
        joblib.dump(sell_lgbm, MODEL_PATH_SELL_LGBM)
    if meta_buy:
        joblib.dump(meta_buy,  META_MODEL_BUY_PATH)
        logger.info(f"[Trainer] ✅ Meta-model BUY сохранена: {META_MODEL_BUY_PATH}")
    if meta_sell:
        joblib.dump(meta_sell, META_MODEL_SELL_PATH)
        logger.info(f"[Trainer] ✅ Meta-model SELL сохранена: {META_MODEL_SELL_PATH}")

    with open(MODEL_FEATURES_PATH, 'w') as f:
        json.dump(feature_cols, f)

    # 14. Итоги
    avg_buy_prec  = (buy_xgb_m['precision'] + (buy_lgbm_m['precision'] if buy_lgbm_m else buy_xgb_m['precision'])) / 2
    avg_sell_prec = (sell_xgb_m['precision'] + (sell_lgbm_m['precision'] if sell_lgbm_m else sell_xgb_m['precision'])) / 2
    avg_buy_auc   = (buy_xgb_m['roc_auc']   + (buy_lgbm_m['roc_auc']   if buy_lgbm_m else buy_xgb_m['roc_auc']))   / 2
    avg_sell_auc  = (sell_xgb_m['roc_auc']  + (sell_lgbm_m['roc_auc']  if sell_lgbm_m else sell_xgb_m['roc_auc'])) / 2

    stats = {
        "success":            True,
        "labeling":           "triple_barrier",
        "n_features":         len(feature_cols),
        "n_samples_buy":      len(df_buy),
        "n_samples_sell":     len(df_sell),
        "n_samples":          len(df_buy),
        "n_train":            split_buy,
        "n_test":             len(X_buy_test),
        "buy_xgb_precision":  buy_xgb_m['precision'],
        "buy_xgb_recall":     buy_xgb_m['recall'],
        "buy_xgb_auc":        buy_xgb_m['roc_auc'],
        "buy_lgbm_precision": buy_lgbm_m['precision'] if buy_lgbm_m else None,
        "buy_lgbm_auc":       buy_lgbm_m['roc_auc']  if buy_lgbm_m else None,
        "avg_buy_precision":  avg_buy_prec,
        "avg_buy_auc":        avg_buy_auc,
        "sell_xgb_precision":  sell_xgb_m['precision'],
        "sell_xgb_recall":     sell_xgb_m['recall'],
        "sell_xgb_auc":        sell_xgb_m['roc_auc'],
        "sell_lgbm_precision": sell_lgbm_m['precision'] if sell_lgbm_m else None,
        "sell_lgbm_auc":       sell_lgbm_m['roc_auc']  if sell_lgbm_m else None,
        "avg_sell_precision":  avg_sell_prec,
        "avg_sell_auc":        avg_sell_auc,
        "meta_buy_precision":  meta_buy_m['precision'] if meta_buy_m else None,
        "meta_sell_precision": meta_sell_m['precision'] if meta_sell_m else None,
        "wf_buy_precision":    wf_buy['wf_precision'],
        "wf_sell_precision":   wf_sell['wf_precision'],
        "wf_buy_sharpe":       wf_buy['wf_sharpe'],
        "wf_sell_sharpe":      wf_sell['wf_sharpe'],
        "wf_folds":            wf_buy['wf_folds'],
        "xgb_precision":       avg_buy_prec,
        "lgbm_precision":      buy_lgbm_m['precision'] if buy_lgbm_m else None,
        "ensemble_precision":  (avg_buy_prec + avg_sell_prec) / 2,
        "wf_precision":        (wf_buy['wf_precision'] + wf_sell['wf_precision']) / 2,
        "wf_accuracy":         0.0,
        "lgbm_available":      LGBM_AVAILABLE,
        "smote_available":     SMOTE_AVAILABLE,
        "meta_labeling":       meta_buy is not None,
    }

    with open(STATS_FILE, 'w') as f:
        json.dump({k: v for k, v in stats.items()}, f, indent=2)

    logger.info(
        f"[Trainer] ✅ Готово! "
        f"BUY prec={avg_buy_prec:.1%} auc={avg_buy_auc:.3f} | "
        f"SELL prec={avg_sell_prec:.1%} auc={avg_sell_auc:.3f} | "
        f"WF Sharpe BUY={wf_buy['wf_sharpe']:.2f} SELL={wf_sell['wf_sharpe']:.2f}"
    )

    return {
        **stats,
        "model":      buy_xgb,
        "lgbm_model": buy_lgbm,
    }