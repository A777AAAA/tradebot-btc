import pandas as pd
import pandas_ta as ta

def create_ml_features_v2():
    print("🚀 Запуск Feature Engineering v2.2 (Бронебойный)...")
    
    try:
        df = pd.read_csv("okx_ton_data.csv", index_col='Timestamp', parse_dates=True)
    except FileNotFoundError:
        print("❌ Ошибка: файл okx_ton_data.csv не найден!")
        return

    # --- ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ ---
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # Считаем Боллинджера
    bbands = ta.bbands(df['Close'], length=20, std=2)
    # Ищем нижнюю полосу (BBL), не обращая внимания на регистр
    bbl_search = [col for col in bbands.columns if 'BBL' in col.upper()]
    if bbl_search:
        df['BB_Dist_Lower'] = (bbands[bbl_search[0]] - df['Close']) / df['Close'] * 100
    else:
        print(f"⚠️ Не нашел колонку BBL. Доступные колонки BB: {list(bbands.columns)}")

    # Считаем MACD
    macd = ta.macd(df['Close'])
    # Ищем гистограмму (MACDH), не обращая внимания на регистр
    macdh_search = [col for col in macd.columns if 'MACDH' in col.upper()]
    if macdh_search:
        df['MACD_Hist'] = macd[macdh_search[0]]
    else:
        print(f"⚠️ Не нашел колонку MACDH. Доступные колонки MACD: {list(macd.columns)}")

    # --- ДИНАМИКА РЫНКА ---
    df['Vol_Change'] = df['Volume'].pct_change() * 100
    df['Price_Change_3h'] = df['Close'].pct_change(3) * 100

    # --- РАЗМЕТКА ЦЕЛИ ---
    horizon = 8
    threshold = 0.008 
    df['Future_Close'] = df['Close'].shift(-horizon)
    df['Target'] = ((df['Future_Close'] - df['Close']) / df['Close'] > threshold).astype(int)

    df.dropna(inplace=True)

    output_file = "ml_ready_ton_data_v2.csv"
    df.to_csv(output_file)
    
    print(f"✅ Готово! Файл сохранен: {output_file}")
    counts = df['Target'].value_counts()
    print(f"Сигналы: 0 (Ждем) = {counts.get(0, 0)}, 1 (ПОКУПАЕМ) = {counts.get(1, 0)}")

if __name__ == "__main__":
    create_ml_features_v2()