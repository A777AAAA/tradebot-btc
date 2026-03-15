import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta

def get_okx_data_massive():
    exchange = ccxt.okx()
    symbol = 'TON/USDT'
    timeframe = '1h'
    
    # Запрашиваем данные за последние 180 дней
    days_back = 180
    since = exchange.parse8601((datetime.now() - timedelta(days=days_back)).isoformat())
    now = exchange.milliseconds()
    
    all_ohlcv = []
    print(f"Начинаю масштабную загрузку истории {symbol} за {days_back} дней...")

    # Цикл "перелистывания страниц" истории
    while since < now:
        try:
            # Запрашиваем порцию данных
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=100)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            # Сдвигаем время "since" на последнюю полученную свечу + 1 миллисекунда
            since = ohlcv[-1][0] + 1
            
            print(f"Загружено свечей: {len(all_ohlcv)}...")
            time.sleep(0.2) # Пауза, чтобы биржа не заблокировала нас за спам
            
        except Exception as e:
            print(f"Ошибка при загрузке: {e}")
            break

    df = pd.DataFrame(all_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)
    
    # Удаляем дубликаты, если они случайно появились
    df = df[~df.index.duplicated(keep='first')]

    print("--- Успешно! Массив данных собран ---")
    print(f"Итого загружено уникальных свечей: {len(df)}")
    
    df.to_csv("okx_ton_data.csv")
    print("Данные сохранены в файл: okx_ton_data.csv")

if __name__ == "__main__":
    get_okx_data_massive()