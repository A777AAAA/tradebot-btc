import ccxt
import pandas as pd
import pandas_ta as ta
import joblib
import os
import time
import logging
import numpy as np
from datetime import datetime
from hf_storage import load_model_from_hub
from telegram_notify import send_telegram_message

# Настраиваем логирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_4h_features(symbol='TON/USDT', limit=100):
    """
    Скачивает 4-часовые свечи, рассчитывает индикаторы и возвращает DataFrame
    с индексом timestamp и колонками признаков.
    """
    try:
        exchange = ccxt.okx()
        ohlcv_4h = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=limit)
        df_4h = pd.DataFrame(ohlcv_4h, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df_4h['Timestamp'] = pd.to_datetime(df_4h['Timestamp'], unit='ms')
        df_4h.set_index('Timestamp', inplace=True)

        df_4h['EMA50_4h'] = ta.ema(df_4h['Close'], length=50)
        df_4h['RSI_4h'] = ta.rsi(df_4h['Close'], length=14)
        df_4h['ATR_4h'] = ta.atr(df_4h['High'], df_4h['Low'], df_4h['Close'], length=14)

        macd_4h = ta.macd(df_4h['Close'])
        macdh_col = None
        for col in macd_4h.columns:
            if 'MACDh' in col.upper():
                macdh_col = col
                break
        if macdh_col is None:
            df_4h['MACD_Hist_4h'] = 0.0
        else:
            df_4h['MACD_Hist_4h'] = macd_4h[macdh_col]

        df_4h.dropna(inplace=True)
        return df_4h[['EMA50_4h', 'RSI_4h', 'ATR_4h', 'MACD_Hist_4h']]
    except Exception as e:
        logging.error(f"Ошибка получения 4h данных: {e}")
        return pd.DataFrame(columns=['EMA50_4h', 'RSI_4h', 'ATR_4h', 'MACD_Hist_4h'])

def get_signal():
    """
    Получает последние данные с биржи, вычисляет признаки (включая 4h контекст)
    и возвращает предсказание модели (0 или 1) и вероятность.
    При сильном сигнале отправляет уведомление в Telegram.
    """
    logging.info("Начало get_signal()")
    try:
        model, metadata = load_model_from_hub()
        logging.info(f"✅ Модель загружена. Точность: {metadata.get('accuracy', 'N/A')}")
    except Exception as e:
        logging.error(f"❌ Не удалось загрузить модель: {e}")
        return None, None

    logging.info("Получаем 1h данные...")
    try:
        exchange = ccxt.okx()
        ohlcv = exchange.fetch_ohlcv('TON/USDT', timeframe='1h', limit=100)
        df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df.set_index('Timestamp', inplace=True)
        logging.info(f"1h данные получены, {len(df)} свечей")
    except Exception as e:
        logging.error(f"Ошибка получения 1h данных: {e}")
        return None, None

    logging.info("Расчёт индикаторов 1h...")
    try:
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

        bb = ta.bbands(df['Close'], length=20, std=2)
        bbl_col = [col for col in bb.columns if 'BBL' in col.upper()][0]
        df['BB_Dist_Lower'] = (bb[bbl_col] - df['Close']) / df['Close'] * 100

        macd = ta.macd(df['Close'])
        macd_col = [col for col in macd.columns if 'MACDH' in col.upper()][0]
        df['MACD_Hist'] = macd[macd_col]

        df['Vol_Change'] = df['Volume'].pct_change() * 100
        df['Price_Change_3h'] = df['Close'].pct_change(3) * 100
        logging.info("Индикаторы 1h рассчитаны")
    except Exception as e:
        logging.error(f"Ошибка расчёта индикаторов 1h: {e}")
        return None, None

    logging.info("Получаем 4h признаки...")
    df_4h = get_4h_features()
    if not df_4h.empty:
        last_1h_time = df.index[-1]
        df_4h_relevant = df_4h[df_4h.index <= last_1h_time]
        if not df_4h_relevant.empty:
            last_4h_features = df_4h_relevant.iloc[-1]
        else:
            last_4h_features = df_4h.iloc[-1]
    else:
        last_4h_features = pd.Series([0.0, 0.0, 0.0, 0.0],
                                      index=['EMA50_4h', 'RSI_4h', 'ATR_4h', 'MACD_Hist_4h'])
    logging.info("4h признаки получены")

    latest = df.iloc[-1]
    features = [
        latest['RSI'],
        latest['ATR'],
        latest['BB_Dist_Lower'],
        latest['MACD_Hist'],
        latest['Vol_Change'],
        latest['Price_Change_3h'],
        last_4h_features['EMA50_4h'],
        last_4h_features['RSI_4h'],
        last_4h_features['ATR_4h'],
        last_4h_features['MACD_Hist_4h']
    ]
    logging.info(f"Признаки сформированы, длина: {len(features)}")

    X = np.array(features).reshape(1, -1)

    try:
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
        logging.info(f"Предсказание: {prediction}, вероятность: {probability:.4f}")
    except Exception as e:
        logging.error(f"Ошибка при предсказании: {e}")
        return None, None

    # ПРИНУДИТЕЛЬНЫЙ ТЕСТ: отправляем сообщение в Telegram при каждом запуске (для проверки)
    try:
        send_telegram_message(f"🔔 Тестовое сообщение от бота\nВероятность сигнала: {probability:.2%}")
        logging.info("Принудительный тест отправлен")
    except Exception as e:
        logging.error(f"Ошибка при вызове send_telegram_message: {e}")

    # Отправка уведомления, если сигнал сильный
    if prediction == 1 and probability > 0.6:
        current_price = latest['Close']
        message = (f"🚀 <b>СИГНАЛ НА ПОКУПКУ TON/USDT</b>\n"
                   f"Цена входа: {current_price:.4f}\n"
                   f"Вероятность: {probability:.2%}\n"
                   f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        send_telegram_message(message)

    logging.info("Завершение get_signal()")
    return prediction, probability

if __name__ == "__main__":
    print("🔍 Live Signal Checker (с поддержкой 4h контекста)")
    pred, prob = get_signal()
    if pred is not None:
        signal = "📈 ПОКУПКА" if pred == 1 else "📉 НЕТ СИГНАЛА"
        print(f"{signal} | Вероятность: {prob:.2%}")