import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import stumpy
import numpy as np

# --- НАСТРОЙКИ ДЛЯ TON ---
ASSET = "TON11419-USD" # Официальный тикер Toncoin на Yahoo Finance
WINDOW = 21            # Для TON лучше брать чуть меньшее окно (3 недели), он быстрее меняет тренд
FORECAST = 7           # Прогноз на неделю вперед

print(f"\nЗагрузка данных TON...")

try:
    # Загружаем данные (TON активно торгуется последние пару лет, берем этот период)
    df = yf.download(ASSET, period="2y", interval="1d")
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    # Считаем ATR (Средний истинный диапазон) для умного стоп-лосса
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df = df.dropna()

    prices = df['Close'].values.flatten()
    atrs = df['ATR'].values.flatten()
    last_price = prices[-1]
    last_atr = atrs[-1]

    # Поиск паттернов
    current_pat = prices[-WINDOW:]
    history = prices[:-WINDOW-FORECAST]
    distance_profile = stumpy.mass(current_pat, history)
    
    # Берем ТОП-5 совпадений
    best_idxs = np.argpartition(distance_profile, 5)[:5]
    
    print(f"\n{'='*40}")
    print(f" АНАЛИТИКА TON (Toncoin)")
    print(f" Текущая цена: ${last_price:.4f}")
    print(f"{'='*40}")

    all_returns = []
    for idx in best_idxs:
        p_then = prices[idx + WINDOW - 1]
        p_future = prices[idx + WINDOW + FORECAST - 1]
        all_returns.append(((p_future - p_then) / p_then) * 100)

    avg_res = np.mean(all_returns)
    win_rate = (len([r for r in all_returns if r > 0]) / 5) * 100

    # ТОРГОВЫЙ ПЛАН С УЧЕТОМ ВОЛАТИЛЬНОСТИ
    # Стоп-лосс ставим на расстоянии 1.5 средних дневных хода (ATR)
    stop_loss_price = last_price - (last_atr * 1.5)
    take_profit_price = last_price * (1 + avg_res/100)

    print(f"Вероятность роста по истории: {win_rate}%")
    print(f"Средний ожидаемый ход: {avg_res:.2f}%")
    print(f"\n--- ЦИФРЫ ДЛЯ ВХОДА (TON) ---")
    print(f"ВХОД: {last_price:.4f}")
    print(f"ЦЕЛЬ (Take Profit): {take_profit_price:.4f}")
    print(f"ЗАЩИТА (Stop Loss): {stop_loss_price:.4f}")
    print(f"Риск/Прибыль: 1 к {abs(avg_res / ((last_price-stop_loss_price)/last_price*100)):.1f}")
    print(f"{'='*40}\n")

    # График с зонами
    plt.figure(figsize=(10, 6))
    plt.plot(prices[-40:], label="Цена TON", color='cyan', lw=2)
    plt.axhline(take_profit_price, color='lime', linestyle='--', label='Take Profit')
    plt.axhline(stop_loss_price, color='tomato', linestyle='--', label='Stop Loss')
    plt.fill_between(range(40), last_price, take_profit_price, color='green', alpha=0.1)
    plt.fill_between(range(40), last_price, stop_loss_price, color='red', alpha=0.1)
    plt.title(f"Торговый план TON: Ожидание {avg_res:.2f}%")
    plt.legend()
    plt.style.use('dark_background')
    plt.show()

except Exception as e:
    print(f"Ошибка: {e}")