import pandas as pd
import numpy as np
import joblib
from hf_storage import load_model_from_hub
import matplotlib.pyplot as plt

def run_backtest():
    print("📈 Запуск бэктеста: Проверка новой стратегии с BTC...")
    
    # 1. Загрузка данных и модели
    try:
        model, metadata = load_model_from_hub()
        # Важно: загружаем тот же файл, который сохранил auto_trainer.py
        df = pd.read_csv("ml_ready_ton_data_v2.csv", index_col='Timestamp', parse_dates=True)
        print(f"✅ Данные загружены. Записей: {len(df)}")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return

    # 2. Список признаков (ТЕПЕРЬ С BTC)
    features = [
        'RSI', 'ATR', 'BB_Dist_Lower', 'MACD_Hist', 'Vol_Change', 'Price_Change_3h',
        'EMA50_4h', 'RSI_4h', 'ATR_4h', 'MACD_Hist_4h',
        'EMA20', 'EMA50', 'RSI7', 'Volume_SMA5', 'High_Low_pct', 'Close_shift_1',
        'BTC_pct_1h', 'BTC_pct_4h'  # <-- Добавлены новые колонки
    ]
    
    # Проверка: есть ли все нужные колонки в CSV?
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"❌ В данных не хватает колонок: {missing}")
        print("💡 Запусти auto_trainer.py еще раз, чтобы обновить CSV файл.")
        return

    # Берем последние 20% данных для теста
    test_size = int(len(df) * 0.2)
    if test_size < 1: test_size = len(df) # Если данных мало, берем всё
    
    test_df = df.tail(test_size).copy()
    X_test = test_df[features]
    
    # 3. Получаем предсказания
    print("🧠 Модель анализирует рынок...")
    probs = model.predict_proba(X_test)[:, 1]
    test_df['Signal_Prob'] = probs
    # Порог 0.65 — делаем бота чуть более осторожным (фильтр качества)
    test_df['Signal'] = (test_df['Signal_Prob'] > 0.65).astype(int)

    # 4. Параметры симуляции
    initial_balance = 100.0
    balance = initial_balance
    commission = 0.001
    position = 0
    history = [initial_balance]

    print(f"💰 Старт: {initial_balance} USDT | Порог: 65%")

    for i in range(len(test_df)):
        current_price = test_df['Close'].iloc[i]
        signal = test_df['Signal'].iloc[i]
        
        if position == 0 and signal == 1:
            entry_price = current_price
            balance -= balance * commission
            position = 1
            exit_counter = 0
            
        elif position == 1:
            exit_counter += 1
            # Выход через 8 часов или в конце данных
            if exit_counter >= 8 or i == len(test_df) - 1:
                profit_pct = (current_price - entry_price) / entry_price
                balance += balance * profit_pct
                balance -= balance * commission
                position = 0
                history.append(balance)
                print(f"✅ Сделка: {profit_pct*100:.2f}% | Баланс: {balance:.2f} USDT")

    # 5. Итоги
    total_return = ((balance - initial_balance) / initial_balance) * 100
    print(f"\n--- ИТОГИ ---")
    print(f"🏁 Финал: {balance:.2f} USDT ({total_return:.2f}%)")
    
    if len(history) > 1:
        plt.plot(history)
        plt.title("Equity Curve")
        plt.show()
    else:
        print("⚠️ Сделок не было. Попробуй снизить порог Signal_Prob до 0.6")

if __name__ == "__main__":
    run_backtest()