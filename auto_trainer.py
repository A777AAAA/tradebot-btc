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
from sklearn.metrics import accuracy_score, precision_score, recall_score
from datetime import datetime, timezone

MODEL_FILE = "ai_brain.pkl"
PAPER_FILE = "paper_trades.json"
STATS_FILE = "training_stats.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 17 признаков (СТРОГО ОДИНАКОВО В ОБОИХ ФАЙЛАХ)
FEATURE_COLS = [
    'RSI_14', 'RSI_7', 'MACD', 'MACD_signal', 'MACD_hist',
    'ATR_pct', 'ADX', 'BB_pos', 'EMA_ratio', 'Vol_ratio',
    'Body_pct', 'Upper_wick', 'Lower_wick',
    'Return_1h', 'Return_4h', 'Return_12h', 'Return_24h'
]

def fetch_ohlcv(symbol="TON-USDT", timeframe="1H", limit=2000):
    all_data = []
    after = None
    try:
        for _ in range(7): # 300 * 7 = 2100 свечей
            url = f"https://www.okx.com/api/v5/market/history-candles?instId={symbol}&bar={timeframe}&limit=300"
            if after: url += f"&after={after}"
            r = requests.get(url, timeout=15).json()
            data = r.get("data", [])
            if not data: break
            all_data.extend(data)
            after = data[-1][0]
            time.sleep(0.2)
        df = pd.DataFrame(all_data, columns=['ts','Open','High','Low','Close','Volume','VolCcy','VolCcyQuote','Confirm'])
        df[['Open','High','Low','Close','Volume']] = df[['Open','High','Low','Close','Volume']].astype(float)
        df['ts'] = pd.to_datetime(df['ts'].astype(float), unit='ms')
        df.set_index('ts', inplace=True)
        return df.sort_index()
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def calc_indicators(df):
    d = df.copy()
    # RSI
    for p in [7, 14]:
        diff = d['Close'].diff()
        gain = diff.clip(lower=0); loss = -diff.clip(upper=0)
        avg_g = gain.ewm(com=p-1, min_periods=p).mean()
        avg_l = loss.ewm(com=p-1, min_periods=p).mean()
        d[f'RSI_{p}'] = 100 - (100 / (1 + avg_g/avg_l))
    # MACD
    ema12 = d['Close'].ewm(span=12, adjust=False).mean()
    ema26 = d['Close'].ewm(span=26, adjust=False).mean()
    d['MACD'] = ema12 - ema26
    d['MACD_signal'] = d['MACD'].ewm(span=9, adjust=False).mean()
    d['MACD_hist'] = d['MACD'] - d['MACD_signal']
    # ATR & ADX
    tr = pd.concat([d['High']-d['Low'], (d['High']-d['Close'].shift()).abs(), (d['Low']-d['Close'].shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(com=13, min_periods=14).mean()
    d['ATR_pct'] = (atr / d['Close']) * 100
    # Bollinger
    sma20 = d['Close'].rolling(20).mean(); std20 = d['Close'].rolling(20).std()
    d['BB_pos'] = (d['Close'] - (sma20 - 2*std20)) / (4*std20)
    # EMA Ratio & Volume
    d['EMA_ratio'] = d['Close'].ewm(span=20).mean() / d['Close'].ewm(span=50).mean()
    d['Vol_ratio'] = d['Volume'] / d['Volume'].rolling(20).mean()
    # Wicks & Body
    d['Body_pct'] = (d['Close'] - d['Open']).abs() / d['Open'] * 100
    d['Upper_wick'] = (d['High'] - d[['Close','Open']].max(axis=1)) / d['Open'] * 100
    d['Lower_wick'] = (d[['Close','Open']].min(axis=1) - d['Low']) / d['Open'] * 100
    # Returns
    for h in [1, 4, 12, 24]: d[f'Return_{h}h'] = d['Close'].pct_change(h) * 100
    # ADX (Simple version)
    up = d['High'].diff(); down = -d['Low'].diff()
    pdm = up.where((up>down)&(up>0), 0); mdm = down.where((down>up)&(down>0), 0)
    pdi = 100 * (pdm.ewm(alpha=1/14).mean() / atr)
    mdi = 100 * (mdm.ewm(alpha=1/14).mean() / atr)
    dx = 100 * (pdi-mdi).abs() / (pdi+mdi)
    d['ADX'] = dx.ewm(alpha=1/14).mean()
    return d.dropna()

def train_model():
    df_raw = fetch_ohlcv()
    if df_raw.empty: return {"success": False, "error": "No data"}
    df = calc_indicators(df_raw)
    
    # ЦЕЛЬ: Рост на 1.5% за 8 часов (Повышаем планку!)
    df['Target'] = (df['Close'].shift(-8) > df['Close'] * 1.015).astype(int)
    
    X = df[FEATURE_COLS].values
    y = df['Target'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Балансировка
    pos = np.sum(y_train == 1); neg = np.sum(y_train == 0)
    scale = neg / pos if pos > 0 else 1.0

    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, scale_pos_weight=scale, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    joblib.dump(model, MODEL_FILE)
    
    stats = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "n_samples": len(df),
        "success": True,
        "paper_trades": 0,
        "top_features": []
    }
    with open(STATS_FILE, "w") as f: json.dump(stats, f)
    return stats

if __name__ == "__main__":
    print(train_model())