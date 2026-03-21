import ccxt
import pandas as pd
import logging
import numpy as np
from datetime import datetime
from hf_storage import load_model_from_hub
from telegram_notify import send_telegram_message

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------------------------------------------------
# Индикаторы вручную (без pandas_ta)
# ----------------------------------------------------------------------
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_macd_hist(series, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    return macd_line - signal_line

def calc_bb_dist_lower(series, period=20, std=2):
    sma = series.rolling(window=period).mean()
    stddev = series.rolling(window=period).std()
    lower = sma - std * stddev
    return (lower - series) / series * 100


# ----------------------------------------------------------------------
# 4h признаки
# ----------------------------------------------------------------------
def get_4h_features(symbol='TON/USDT', limit=100):
    logging.info("Начинаем получение 4h данных...")
    try:
        exchange = ccxt.okx({'timeout': 30000})
        ohlcv_4h = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=limit)
        df = pd.DataFrame(ohlcv_4h, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df.set_index('Timestamp', inplace=True)

        df['EMA50_4h'] = calc_ema(df['Close'], 50)
        df['RSI_4h'] = calc_rsi(df['Close'], 14)
        df['ATR_4h'] = calc_atr(df['High'], df['Low'], df['Close'], 14)
        df['MACD_Hist_4h'] = calc_macd_hist(df['Close'])

        df.dropna(inplace=True)
        return df[['EMA50_4h', 'RSI_4h', 'ATR_4h', 'MACD_Hist_4h']]
    except Exception as e:
        logging.error(f"Ошибка получения 4h данных: {e}")
        return pd.DataFrame(columns=['EMA50_4h', 'RSI_4h', 'ATR_4h', 'MACD_Hist_4h'])


# ----------------------------------------------------------------------
# BTC контекст
# ----------------------------------------------------------------------
def get_btc_context_live(limit=100):
    try:
        exchange = ccxt.okx()
        btc_ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=limit)
        df_btc = pd.DataFrame(btc_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df_btc['Timestamp'] = pd.to_datetime(df_btc['Timestamp'], unit='ms')
        df_btc.set_index('Timestamp', inplace=True)
        df_btc['BTC_pct_1h'] = df_btc['Close'].pct_change() * 100
        df_btc['BTC_pct_4h'] = df_btc['Close'].pct_change(4) * 100
        return df_btc[['BTC_pct_1h', 'BTC_pct_4h']].dropna()
    except Exception as e:
        logging.error(f"Ошибка получения BTC данных: {e}")
        return pd.DataFrame()


# ----------------------------------------------------------------------
# Подготовка признаков
# ----------------------------------------------------------------------
def prepare_realtime_features(df_1h_raw, df_4h_features):
    df = df_1h_raw.copy()

    df['RSI'] = calc_rsi(df['Close'], 14)
    df['ATR'] = calc_atr(df['High'], df['Low'], df['Close'], 14)
    df['BB_Dist_Lower'] = calc_bb_dist_lower(df['Close'], 20, 2)
    df['MACD_Hist'] = calc_macd_hist(df['Close'])
    df['Vol_Change'] = df['Volume'].pct_change() * 100
    df['Price_Change_3h'] = df['Close'].pct_change(3) * 100
    df['EMA20'] = calc_ema(df['Close'], 20)
    df['EMA50'] = calc_ema(df['Close'], 50)
    df['RSI7'] = calc_rsi(df['Close'], 7)
    df['Volume_SMA5'] = df['Volume'].rolling(window=5).mean()
    df['High_Low_pct'] = (df['High'] - df['Low']) / df['Close'] * 100
    df['Close_shift_1'] = df['Close'].shift(1)

    df_btc = get_btc_context_live(limit=len(df) + 10)

    df.index = pd.to_datetime(df.index).as_unit('ns')
    df_4h_features.index = pd.to_datetime(df_4h_features.index).as_unit('ns')

    df_merged = pd.merge_asof(
        df.sort_index(),
        df_4h_features.sort_index(),
        left_index=True,
        right_index=True,
        direction='backward'
    )

    if not df_btc.empty:
        df_btc.index = pd.to_datetime(df_btc.index).as_unit('ns')
        df_merged = pd.merge_asof(
            df_merged.sort_index(),
            df_btc.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )
    else:
        df_merged['BTC_pct_1h'] = 0.0
        df_merged['BTC_pct_4h'] = 0.0

    df_clean = df_merged.dropna().copy()

    feature_names = [
        'RSI', 'ATR', 'BB_Dist_Lower', 'MACD_Hist', 'Vol_Change', 'Price_Change_3h',
        'EMA50_4h', 'RSI_4h', 'ATR_4h', 'MACD_Hist_4h',
        'EMA20', 'EMA50', 'RSI7', 'Volume_SMA5', 'High_Low_pct', 'Close_shift_1',
        'BTC_pct_1h', 'BTC_pct_4h'
    ]

    X = df_clean[feature_names].iloc[-1:]
    logging.info(f"✅ Признаки сформированы, всего признаков: {len(feature_names)}")
    return X


# ----------------------------------------------------------------------
# Основная функция
# ----------------------------------------------------------------------
def get_signal():
    logging.info("=" * 50)
    logging.info("Начало работы get_signal()")

    try:
        model, metadata = load_model_from_hub()
        logging.info(f"✅ Модель загружена. Точность: {metadata.get('accuracy', 'N/A')}")
        atr_mean = metadata.get('atr_mean')
    except Exception as e:
        logging.error(f"❌ Не удалось загрузить модель: {e}")
        return None, None

    try:
        exchange = ccxt.okx()
        ohlcv = exchange.fetch_ohlcv('TON/USDT', timeframe='1h', limit=150)
        df_1h = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df_1h['Timestamp'] = pd.to_datetime(df_1h['Timestamp'], unit='ms')
        df_1h.set_index('Timestamp', inplace=True)
        logging.info(f"✅ 1h данные получены, {len(df_1h)} свечей")
    except Exception as e:
        logging.error(f"❌ Ошибка получения 1h данных: {e}")
        return None, None

    df_4h = get_4h_features()
    X = prepare_realtime_features(df_1h, df_4h)

    try:
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]
        logging.info(f"📊 Предсказание: {pred}, вероятность: {prob:.4f}")
    except Exception as e:
        logging.error(f"❌ Ошибка при предсказании: {e}")
        return None, None

    if pred == 1 and prob > 0.6:
        current_price = df_1h['Close'].iloc[-1]
        msg = (f"🚀 <b>СИГНАЛ НА ПОКУПКУ TON/USDT</b>\n"
               f"Цена входа: {current_price:.4f}\n"
               f"Вероятность: {prob:.2%}\n"
               f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        if atr_mean:
            sl = current_price - 2 * atr_mean
            tp = current_price + 3 * atr_mean
            msg += f"\n\n📉 Стоп-лосс: {sl:.4f}\n📈 Тейк-профит: {tp:.4f}"
        send_telegram_message(msg)
        logging.info("✅ Сигнал отправлен в Telegram")
    else:
        logging.info("⏺️ Нет сигнала")

    logging.info("🏁 Завершение get_signal()")
    logging.info("=" * 50)
    return pred, prob


if __name__ == "__main__":
    p, pr = get_signal()
    if p is not None:
        print(f"{'📈 ПОКУПКА' if p == 1 else '📉 НЕТ СИГНАЛА'} | Вероятность: {pr:.2%}")

def get_live_signal():
    """Обёртка для app.py — возвращает словарь вместо tuple"""
    try:
        exchange = ccxt.okx()
        ohlcv = exchange.fetch_ohlcv('TON/USDT', timeframe='1h', limit=5)
        current_price = ohlcv[-1][4]  # Close последней свечи
        change_24h_ohlcv = exchange.fetch_ohlcv('TON/USDT', timeframe='1d', limit=2)
        change_24h = (change_24h_ohlcv[-1][4] - change_24h_ohlcv[-2][4]) / change_24h_ohlcv[-2][4] * 100
        volume = change_24h_ohlcv[-1][5]
    except Exception as e:
        logging.error(f"Ошибка получения цены: {e}")
        current_price = 0.0
        change_24h = 0.0
        volume = 0.0

    pred, prob = get_signal()

    if pred is None:
        return None

    signal = "BUY" if pred == 1 else "HOLD"

    return {
        "signal":     signal,
        "confidence": float(prob),
        "price":      current_price,
        "change_24h": change_24h,
        "volume":     volume
    }