"""
auto_trainer.py — Переобучение модели каждые 6 часов.
"""

import os
import json
import joblib
import logging
import requests
import time
import numpy as np
import pandas as pd

from xgboost          import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics  import accuracy_score, precision_score, recall_score
from datetime         import datetime, timezone

MODEL_FILE  = "ai_brain.pkl"
PAPER_FILE  = "paper_trades.json"
STATS_FILE  = "training_stats.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ─────────────────────────────────────────────
# 1. Загрузка данных — несколько запросов по 300 свечей
# ─────────────────────────────────────────────
def fetch_ohlcv(symbol="TON-USDT", timeframe="1H", limit=2000) -> pd.DataFrame:
    """OKX отдаёт max 300 свечей за раз — делаем несколько запросов"""
    try:
        all_data = []
        after = None
        fetched = 0
        max_requests = 10

        for i in range(max_requests):
            url = f"https://www.okx.com/api/v5/market/history-candles?instId={symbol}&bar={timeframe}&limit=300"
            if after:
                url += f"&after={after}"

            r    = requests.get(url, timeout=15)
            data = r.json().get("data", [])

            if not data:
                break

            all_data.extend(data)
            fetched += len(data)

            if fetched >= limit:
                break

            # after = timestamp самой старой свечи для следующего запроса
            after = data[-1][0]
            time.sleep(0.3)

        if not all_data:
            logging.error("[Trainer] ❌ Пустой ответ от OKX")
            return pd.DataFrame()

        df = pd.DataFrame(all_data, columns=[
            'ts','Open','High','Low','Close','Volume','VolCcy','VolCcyQuote','Confirm'
        ])
        df = df[['ts','Open','High','Low','Close','Volume']].copy()
        df['ts'] = pd.to_datetime(df['ts'].astype(float), unit='ms')
        df.set_index('ts', inplace=True)
        for col in ['Open','High','Low','Close','Volume']:
            df[col] = df[col].astype(float)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]

        logging.info(f"[Trainer] ✅ Загружено {len(df)} свечей с OKX")
        return df

    except Exception as e:
        logging.error(f"[Trainer] ❌ Ошибка загрузки OKX: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# 2. Индикаторы
# ─────────────────────────────────────────────
def calc_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast    = series.ewm(span=fast,   adjust=False).mean()
    ema_slow    = series.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_atr(df, period=14):
    high_low   = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close  = (df['Low']  - df['Close'].shift()).abs()
    tr         = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def calc_adx(df, period=14):
    up   = df['High'].diff()
    down = -df['Low'].diff()
    plus_dm  = up.where((up > down) & (up > 0),    0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr      = calc_atr(df, period)
    plus_di  = 100 * (plus_dm.ewm(com=period-1, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(com=period-1, min_periods=period).mean() / atr)
    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(com=period - 1, min_periods=period).mean()
    return adx


def calc_bollinger(series, period=20, std=2):
    sma    = series.rolling(period).mean()
    stddev = series.rolling(period).std()
    upper  = sma + std * stddev
    lower  = sma - std * stddev
    width  = upper - lower
    pos    = (series - lower) / width.replace(0, np.nan)
    return pos.fillna(0.5)


def calc_volume_ratio(df, period=20):
    avg_vol = df['Volume'].rolling(period).mean()
    return (df['Volume'] / avg_vol.replace(0, np.nan)).fillna(1.0)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['RSI_14'] = calc_rsi(df['Close'], 14)
    df['RSI_7']  = calc_rsi(df['Close'], 7)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = calc_macd(df['Close'])
    atr           = calc_atr(df, 14)
    df['ATR_pct'] = atr / df['Close'] * 100
    df['ADX']     = calc_adx(df, 14)
    df['BB_pos']  = calc_bollinger(df['Close'])
    df['EMA20']     = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50']     = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_ratio'] = df['EMA20'] / df['EMA50']
    df['Vol_ratio'] = calc_volume_ratio(df)
    df['Body_pct']   = (df['Close'] - df['Open']).abs() / df['Open'] * 100
    df['Upper_wick'] = (df['High'] - df[['Close','Open']].max(axis=1)) / df['Open'] * 100
    df['Lower_wick'] = (df[['Close','Open']].min(axis=1) - df['Low']) / df['Open'] * 100
    df['Return_1h']  = df['Close'].pct_change(1)  * 100
    df['Return_4h']  = df['Close'].pct_change(4)  * 100
    df['Return_12h'] = df['Close'].pct_change(12) * 100
    df['Return_24h'] = df['Close'].pct_change(24) * 100

    # ✅ Цель: рост на 1.5% за 8 часов (снижено с 1.5% для большего числа сигналов)
    df['Target'] = (df['Close'].shift(-8) > df['Close'] * 1.005).astype(int)
    return df.dropna()


# ─────────────────────────────────────────────
# 3. Paper Trading результаты
# ─────────────────────────────────────────────
def load_paper_results() -> pd.DataFrame:
    if not os.path.exists(PAPER_FILE):
        logging.info("[Trainer] 📝 Paper trades файл не найден — пропускаем")
        return pd.DataFrame()
    with open(PAPER_FILE) as f:
        trades = json.load(f)
    closed = [t for t in trades if t["status"] == "CLOSED"]
    if len(closed) < 5:
        logging.info(f"[Trainer] 📝 Мало сделок ({len(closed)}) — пропускаем")
        return pd.DataFrame()
    rows = []
    for t in closed:
        rows.append({
            "confidence": t.get("confidence", 50) / 100,
            "signal_buy": 1 if t["signal"] == "BUY" else 0,
            "pnl_pct":    t.get("pnl_pct", 0),
            "Target":     1 if t["result"] == "WIN" else 0,
        })
    df   = pd.DataFrame(rows)
    wins = df['Target'].sum()
    logging.info(f"[Trainer] 📊 Paper: {len(df)} сделок | Winrate: {wins/len(df)*100:.1f}%")
    return df


# ─────────────────────────────────────────────
# 4. Обучение модели
# ─────────────────────────────────────────────
FEATURE_COLS = [
    'RSI_14', 'RSI_7',
    'MACD', 'MACD_signal', 'MACD_hist',
    'ATR_pct', 'ADX',
    'BB_pos', 'EMA_ratio',
    'Vol_ratio',
    'Body_pct', 'Upper_wick', 'Lower_wick',
    'Return_1h', 'Return_4h', 'Return_12h', 'Return_24h',
]


def _json_safe(obj):
    if hasattr(obj, 'item'):
        return obj.item()
    return str(obj)


def train_model(symbol="TON-USDT") -> dict:
    logging.info("[Trainer] 🔄 Начало переобучения...")

    df_raw = fetch_ohlcv(symbol, "1H", 2000)
    if df_raw.empty:
        return {"success": False, "error": "Нет данных OKX"}

    df = add_features(df_raw)
    if len(df) < 100:
        return {"success": False, "error": "Мало данных после обработки"}

    # Логируем баланс классов
    pos = df['Target'].sum()
    neg = len(df) - pos
    logging.info(f"[Trainer] 📊 Баланс: BUY={pos} ({pos/len(df)*100:.1f}%) / HOLD={neg}")

    X = df[FEATURE_COLS].values
    y = df['Target'].values

    paper_df       = load_paper_results()
    sample_weights = np.ones(len(X))

    if not paper_df.empty:
        paper_winrate = paper_df['Target'].mean()
        if paper_winrate < 0.45:
            sample_weights[y == 0] *= 1.5

    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42, shuffle=False
    )

    # scale_pos_weight балансирует классы автоматически
    scale = neg / pos if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators     = 300,
        max_depth        = 4,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 2,
        gamma            = 0.05,
        scale_pos_weight = scale,
        use_label_encoder= False,
        eval_metric      = 'logloss',
        random_state     = 42,
        n_jobs           = -1,
    )

    model.fit(
        X_train, y_train,
        sample_weight = w_train,
        eval_set      = [(X_test, y_test)],
        verbose       = False,
    )

    y_pred    = model.predict(X_test)
    accuracy  = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall    = float(recall_score(y_test, y_pred,    zero_division=0))

    importances  = dict(zip(FEATURE_COLS, [float(x) for x in model.feature_importances_]))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]

    joblib.dump(model, MODEL_FILE)

    stats = {
        "trained_at":   datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_samples":    int(len(X_train)),
        "accuracy":     round(accuracy,  4),
        "precision":    round(precision, 4),
        "recall":       round(recall,    4),
        "paper_trades": int(len(paper_df)) if not paper_df.empty else 0,
        "top_features": [[n, round(float(v), 4)] for n, v in top_features],
        "success":      True,
    }

    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2, default=_json_safe)

    logging.info(
        f"[Trainer] ✅ Модель обучена: "
        f"Accuracy={accuracy:.1%} | Precision={precision:.1%} | Recall={recall:.1%}"
    )
    return stats


def load_model():
    if not os.path.exists(MODEL_FILE):
        logging.info("[Trainer] 🆕 Модели нет — начинаем обучение...")
        train_model()
    model = joblib.load(MODEL_FILE)
    logging.info("[Trainer] ✅ Модель загружена")
    return model


if __name__ == "__main__":
    result = train_model()
    if result.get("success"):
        print(f"Accuracy={result['accuracy']:.1%} | Precision={result['precision']:.1%} | Recall={result['recall']:.1%}")
        print(f"Примеров: {result['n_samples']}")
    else:
        print(f"❌ {result.get('error')}")