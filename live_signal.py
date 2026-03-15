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

# Настраиваем логирование – все сообщения будут видны в логах Render
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_4h_features(symbol='TON/USDT', limit=100):
    """
    Скачивает 4-часовые свечи, рассчитывает индикаторы и возвращает DataFrame
    с индексом timestamp и колонками признаков.
    """
    logging.info("Начинаем получение 4h данных...")
    try:
        exchange = ccxt.okx({'timeout': 30000})  # таймаут 30 секунд
        logging.info("Создан объект exchange, отправляем запрос к OKX...")
        ohlcv_4h = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=limit)
        logging.info(f"Получено {len(ohlcv_4h)} свечей 4h")
        
        df_4h = pd.DataFrame(ohlcv_4h, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df_4h['Timestamp'] = pd.to_datetime(df_4h['Timestamp'], unit='ms')
        df_4h.set_index('Timestamp', inplace=True)
        logging.info("DataFrame 4h создан")

        logging.info("Расчёт индикаторов на 4h...")
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
            logging.warning("Колонка MACDh не найдена, используется 0")
        else:
            df_4h['MACD_Hist_4h'] = macd_4h[macdh_col]

        df_4h.dropna(inplace=True)
        logging.info(f"Индикаторы рассчитаны, осталось {len(df_4h)} записей после dropna")
        return df_4h[['EMA50_4h', 'RSI_4h', 'ATR_4h', 'MACD_Hist_4h']]
    except ccxt.RequestTimeout as e:
        logging.error(f"Таймаут при запросе к OKX: {e}")
        return pd.DataFrame(columns=['EMA50_4h', 'RSI_4h', 'ATR_4h', 'MACD_Hist_4h'])
    except ccxt.NetworkError as e:
        logging.error(f"Сетевая ошибка при запросе к OKX: {e}")
        return pd.DataFrame(columns=['EMA50_4h', 'RSI_4h', 'ATR_4h', 'MACD_Hist_4h'])
    except Exception as e:
        logging.error(f"Неизвестная ошибка получения 4h данных: {e}")
        return pd.DataFrame(columns=['EMA50_4h', 'RSI_4h', 'ATR_4h', 'MACD_Hist_4h'])

def get_signal():
    """
    Получает последние данные с биржи, вычисляет признаки (включая 4h контекст)
    и возвращает предсказание модели (0 или 1) и вероятность.
    При сильном сигнале отправляет уведомление в Telegram.
    """
    logging.info("=" * 50)
    logging.info("Начало работы функции get_signal()")
    
    # 1. Загрузка модели
    try:
        model, metadata = load_model_from_hub()
        logging.info(f"✅ Модель успешно загружена. Точность: {metadata.get('accuracy', 'N/A')}")
    except Exception as e:
        logging.error(f"❌ Критическая ошибка: не удалось загрузить модель: {e}")
        return None, None

    # 2. Получение 1h данных
    logging.info("Получаем 1h данные с OKX...")
    try:
        exchange = ccxt.okx()
        ohlcv = exchange.fetch_ohlcv('TON/USDT', timeframe='1h', limit=100)
        df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df.set_index('Timestamp', inplace=True)
        logging.info(f"✅ 1h данные получены, {len(df)} свечей")
    except Exception as e:
        logging.error(f"❌ Ошибка получения 1h данных: {e}")
        return None, None

    # 3. Расчёт индикаторов на 1h
    logging.info("Расчёт индикаторов на 1h графике...")
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
        logging.info("✅ Индикаторы 1h рассчитаны")
    except Exception as e:
        logging.error(f"❌ Ошибка расчёта индикаторов 1h: {e}")
        return None, None

    # 4. Получение 4h признаков
    logging.info("Получаем 4h признаки...")
    df_4h = get_4h_features()
    if not df_4h.empty:
        last_1h_time = df.index[-1]
        df_4h_relevant = df_4h[df_4h.index <= last_1h_time]
        if not df_4h_relevant.empty:
            last_4h_features = df_4h_relevant.iloc[-1]
            logging.info("✅ 4h признаки получены и сопоставлены")
        else:
            # Если нет 4h свечи до последней 1h, берём самую последнюю 4h свечу (она может быть будущей, но это fallback)
            last_4h_features = df_4h.iloc[-1]
            logging.warning("⚠️ Нет 4h свечи до последней 1h, используем последнюю доступную 4h свечу")
    else:
        last_4h_features = pd.Series([0.0, 0.0, 0.0, 0.0],
                                      index=['EMA50_4h', 'RSI_4h', 'ATR_4h', 'MACD_Hist_4h'])
        logging.warning("⚠️ 4h данные недоступны, используются нулевые значения")

    # 5. Формирование вектора признаков
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
    logging.info(f"✅ Признаки сформированы, всего признаков: {len(features)}")

    X = np.array(features).reshape(1, -1)

    # 6. Получение предсказания
    try:
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
        logging.info(f"📊 Предсказание: {prediction} (0 – нет сигнала, 1 – сигнал), вероятность: {probability:.4f}")
    except Exception as e:
        logging.error(f"❌ Ошибка при предсказании: {e}")
        return None, None

    # 7. Принудительная тестовая отправка в Telegram (для проверки)
    try:
        test_message = f"🔔 Тестовое сообщение от бота\nВероятность сигнала: {probability:.2%}"
        send_result = send_telegram_message(test_message)
        if send_result:
            logging.info("✅ Принудительный тест: сообщение успешно отправлено в Telegram")
        else:
            logging.error("❌ Принудительный тест: не удалось отправить сообщение в Telegram")
    except Exception as e:
        logging.error(f"❌ Исключение при отправке тестового сообщения: {e}")

    # 8. Отправка сигнала, если условие выполнено
    if prediction == 1 and probability > 0.6:
        current_price = latest['Close']
        signal_message = (f"🚀 <b>СИГНАЛ НА ПОКУПКУ TON/USDT</b>\n"
                          f"Цена входа: {current_price:.4f}\n"
                          f"Вероятность: {probability:.2%}\n"
                          f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        try:
            signal_result = send_telegram_message(signal_message)
            if signal_result:
                logging.info("✅ Сигнал отправлен в Telegram")
            else:
                logging.error("❌ Не удалось отправить сигнал в Telegram")
        except Exception as e:
            logging.error(f"❌ Исключение при отправке сигнала: {e}")
    else:
        logging.info("⏺️ Условия для сигнала не выполнены (нет покупки)")

    logging.info("🏁 Завершение работы функции get_signal()")
    logging.info("=" * 50)
    return prediction, probability

if __name__ == "__main__":
    print("🔍 Live Signal Checker (с поддержкой 4h контекста)")
    pred, prob = get_signal()
    if pred is not None:
        signal = "📈 ПОКУПКА" if pred == 1 else "📉 НЕТ СИГНАЛА"
        print(f"{signal} | Вероятность: {prob:.2%}")