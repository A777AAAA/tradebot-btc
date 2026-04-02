import ccxt
import pandas as pd
import logging
import joblib
import os
from datetime import datetime

MODEL_PATH = "ai_brain.pkl"
FEATURE_COLS = [
    'RSI_14', 'RSI_7', 'MACD', 'MACD_signal', 'MACD_hist',
    'ATR_pct', 'ADX', 'BB_pos', 'EMA_ratio', 'Vol_ratio',
    'Body_pct', 'Upper_wick', 'Lower_wick',
    'Return_1h', 'Return_4h', 'Return_12h', 'Return_24h',
    'Hour'
]

def calc_live_indicators(df):
    d = df.copy()
    d['Hour'] = d.index.hour
    for p in [7, 14]:
        diff = d['Close'].diff()
        g = diff.clip(lower=0); l = -diff.clip(upper=0)
        avg_g = g.ewm(com=p-1, min_periods=p).mean()
        avg_l = l.ewm(com=p-1, min_periods=p).mean()
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
    d['BB_pos'] = (d['Close'] - (sma20 - 2*std20)) / (4*std20 + 1e-6)
    d['EMA_ratio'] = d['Close'].ewm(span=20).mean() / d['Close'].ewm(span=50).mean()
    d['Vol_ratio'] = d['Volume'] / d['Volume'].rolling(20).mean()
    d['Body_pct'] = (d['Close'] - d['Open']).abs() / d['Open'] * 100
    d['Upper_wick'] = (d['High'] - d[['Close','Open']].max(axis=1)) / d['Open'] * 100
    d['Lower_wick'] = (d[['Close','Open']].min(axis=1) - d['Low']) / d['Open'] * 100
    for h in [1, 4, 12, 24]: d[f'Return_{h}h'] = d['Close'].pct_change(h) * 100
    up = d['High'].diff(); down = -d['Low'].diff()
    pdm = up.where((up>down)&(up>0), 0); mdm = down.where((down>up)&(down>0), 0)
    pdi = 100 * (pdm.ewm(alpha=1/14).mean() / atr); mdi = 100 * (mdm.ewm(alpha=1/14).mean() / atr)
    dx = 100 * (pdi-mdi).abs() / (pdi+mdi); d['ADX'] = dx.ewm(alpha=1/14).mean()
    return d.dropna()

def get_live_signal():
    try:
        if not os.path.exists(MODEL_PATH): return None
        model = joblib.load(MODEL_PATH)
        
        exchange = ccxt.okx({'options': {'defaultType': 'spot'}})
        ohlcv = exchange.fetch_ohlcv('TON/USDT', timeframe='1h', limit=100)
        df = pd.DataFrame(ohlcv, columns=['ts','Open','High','Low','Close','Volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('ts', inplace=True)
        df[['Open','High','Low','Close','Volume']] = df[['Open','High','Low','Close','Volume']].astype(float)
        
        df_feats = calc_live_indicators(df)
        X = df_feats[FEATURE_COLS].iloc[-1:].values
        
        pred = model.predict(X)[0] # 0=HOLD, 1=BUY, 2=SELL
        probs = model.predict_proba(X)[0]
        
        confidence = float(max(probs))
        cur_price = df['Close'].iloc[-1]
        
        res_signal = "HOLD"
        if pred == 1: res_signal = "BUY"
        if pred == 2: res_signal = "SELL"
        
        return {
            "signal": res_signal,
            "confidence": confidence,
            "price": cur_price,
            "change_24h": 0.0, # Упростили
            "volume": float(df['Volume'].iloc[-1])
        }
    except Exception as e:
        logging.error(f"Signal Error: {e}")
        return None