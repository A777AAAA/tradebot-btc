import pandas as pd
import matplotlib.pyplot as plt

def run_backtest():
    print("Запуск симулятора торговли...")
    
    # 1. Загружаем данные с индикаторами
    df = pd.read_csv("enhanced_ton_data.csv", index_col='Timestamp', parse_dates=True)
    
    # Настройки стратегии
    buy_threshold = 40  # Покупаем, если RSI ниже этого
    sell_threshold = 60 # Продаем, если RSI выше этого
    commission = 0.001  # 0.1% комиссия биржи
    
    balance = 1000      # Наш стартовый капитал ($1000)
    position = 0        # Сколько TON у нас в руках
    history = []        # История изменения баланса

    # 2. Идем по истории свеча за свечой
    for i in range(len(df)):
        current_price = df['Close'].iloc[i]
        current_rsi = df['RSI'].iloc[i]
        
        # ЛОГИКА ПОКУПКИ
        if current_rsi < buy_threshold and balance > 0:
            # Покупаем на все деньги (с учетом комиссии)
            position = (balance * (1 - commission)) / current_price
            balance = 0
            print(f"[{df.index[i]}] КУПИЛ по {current_price:.4f}")

        # ЛОГИКА ПРОДАЖИ
        elif current_rsi > sell_threshold and position > 0:
            # Продаем все монеты (с учетом комиссии)
            balance = (position * current_price) * (1 - commission)
            position = 0
            print(f"[{df.index[i]}] ПРОДАЛ по {current_price:.4f} | Баланс: ${balance:.2f}")

        # Записываем текущую стоимость нашего портфеля
        total_value = balance if balance > 0 else position * current_price
        history.append(total_value)

    # 3. Итоги
    final_profit = history[-1] - 1000
    print(f"\n--- ИТОГ ТЕСТА ---")
    print(f"Финальный баланс: ${history[-1]:.2f}")
    print(f"Чистая прибыль: ${final_profit:.2f}")

    # 4. Рисуем график роста капитала
    plt.figure(figsize=(10, 5))
    plt.plot(history, label='Баланс ($)', color='blue')
    plt.axhline(1000, color='red', linestyle='--', label='Старт ($1000)')
    plt.title('Кривая доходности стратегии RSI на TON (OKX)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_backtest()