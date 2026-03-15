import pandas as pd
import pandas_ta as ta

def enhance_data():
    print("Чтение данных из okx_ton_data.csv...")
    
    # 1. Загружаем сохраненные данные
    try:
        df = pd.read_csv("okx_ton_data.csv", index_col='Timestamp', parse_dates=True)
    except FileNotFoundError:
        print("Ошибка: файл с данными не найден. Сначала запустите fetch_data.py")
        return

    # 2. Считаем RSI (Индекс относительной силы)
    # Это поможет нам понять, когда TON "перекуплен" или "перепродан"
    df['RSI'] = ta.rsi(df['Close'], length=14)

    # 3. Считаем ATR (Средний истинный диапазон)
    # Это нужно для правильного выставления Стоп-Лосса (защиты)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # 4. Сохраняем результат в новый файл
    output_file = "enhanced_ton_data.csv"
    df.to_csv(output_file)
    
    print(f"--- Успешно! ---")
    print(f"Добавлены индикаторы RSI и ATR.")
    print(f"Последние показатели:")
    print(f"Цена: {df['Close'].iloc[-1]:.4f}")
    print(f"RSI: {df['RSI'].iloc[-1]:.2f}")
    print(f"ATR (волатильность): {df['ATR'].iloc[-1]:.4f}")
    print(f"\nДанные с индикаторами сохранены в: {output_file}")

if __name__ == "__main__":
    enhance_data()