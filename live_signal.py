import ccxt
import pandas as pd
import logging
import numpy as np
import joblib
import os
from datetime import datetime
from telegram_notify import send_message

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_PATH = "ai_brain.pkl"
FEATURE_COLS = [
    'RSI_14', 'RSI_7', 'MACD', 'MACD_signal', 'MACD_hist',
    'ATR_pct', 'ADX', 'BB_pos', 'EMA_ratio', 'Vol_ratio',
    'Body_pct', 'Upper_wick', 'Lower_wick',
    'Return_1h', 'Return_4h', 'Return_12h', 'Return_24h'
]

OKX_CONFIG = {'options': {'defaultType': 'spot'}, 'timeout': 30000}

def calc_live_indicators(df):
    # Копия логики из auto_trainer.py
    d = df.copy()
    for p in [7, 14]:
        diff = d['Close'].diff()
        gain = diff.clip(lower=0); loss = -diff.clip(upper=0)
        avg_g = gain.ewm(com=p-1, min_periods=p).mean()
        avg_l = loss.ewm(com=p-1, min_periods=p).mean()
        d[f'RSI_{p}'] = 100 - (100 / (1 + avg_g/avg_l))
    ema12 = d['Close'].ewm(span=12, adjust=False).mean()
    ema26 = d['Close'].ewm(span=26, adjust=False).mean()
    d['MACD'] = ema12 - ema26
    d['MACD_signal'] = d['MACD'].ewm(span=9, adjust=False).mean()
    d['MACD_hist'] = d['MACD'] - d['MACD_signal']
    tr = pd.concat([d['High']-d['Low'], (d['High']-d['Close'].shift()).abs(), (d['Low']-d['Close'].shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(com=13, min_periods=14).mean()
    d['ATR_pct'] = (atr / d['Close']) * 100
    sma20 = d['Close'].rolling(20).mean(); std20 = d['Close'].rolling(20).std()
    d['BB_pos'] = (d['Close'] - (sma20 - 2*std20)) / (4*std20)
    d['EMA_ratio'] = d['Close'].ewm(span=20).mean() / d['Close'].ewm(span=50).mean()
    d['Vol_ratio'] = d['Volume'] / d['Volume'].rolling(20).mean()
    d['Body_pct'] = (d['Close'] - d['Open']).abs() / d['Open'] * 100
    d['Upper_wick'] = (d['High'] - d[['Close','Open']].max(axis=1)) / d['Open'] * 100
    d['Lower_wick'] = (d[['Close','Open']].min(axis=1) - d['Low']) / d['Open'] * 100
    for h in [1, 4, 12, 24]: d[f'Return_{h}h'] = d['Close'].pct_change(h) * 100
    up = d['High'].diff(); down = -d['Low'].diff()
    pdm = up.where((up>down)&(up>0), 0); mdm = down.where((down>up)&(down>0), 0)
    pdi = 100 * (pdm.ewm(alpha=1/14).mean() / atr)
    mdi = 100 * (mdm.ewm(alpha=1/14).mean() / atr)
    dx = 100 * (pdi-mdi).abs() / (pdi+mdi); d['ADX'] = dx.ewm(alpha=1/14).mean()
    return d.dropna()

def get_live_signal():
    try:
        if not os.path.exists(MODEL_PATH): return None
        model = joblib.load(MODEL_PATH)
        
        exchange = ccxt.okx(OKX_CONFIG)
        ohlcv = exchange.fetch_ohlcv('TON/USDT', timeframe='1h', limit=100)
        df = pd.DataFrame(ohlcv, columns=['ts','Open','High','Low','Close','Volume'])
        df[['Open','High','Low','Close','Volume']] = df[['Open','High','Low','Close','Volume']].astype(float)
        
        df_feats = calc_live_indicators(df)
        X = df_feats[FEATURE_COLS].iloc[-1:].values
        
        prob = float(model.predict_proba(X)[0][1])
        cur_price = df['Close'].iloc[-1]
        
        # Данные для анализа настроения
        change_24h = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
        
        return {
            "signal": "BUY" if prob > 0.5 else "HOLD",
            "confidence": prob,
            "price": cur_price,
            "change_24h": change_24h,
            "volume": float(df['Volume'].iloc[-1])
        }
    except Exception as e:
        logging.error(f"Error in get_live_signal: {e}")
        return None