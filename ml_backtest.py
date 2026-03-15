import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

def run_backtest():
    print("📈 Запуск имитации торговли на базе ИИ...")
    
    # 1. Загружаем данные
    df = pd.read_csv("ml_ready_ton_data_v2.csv", index_col='Timestamp', parse_dates=True)
    
    features = ['RSI', 'ATR', 'BB_Dist_Lower', 'MACD_Hist', 'Vol_Change', 'Price_Change_3h']
    X = df[features]
    y = df['Target']

    # 2. Разделяем так же, как при обучении (80% учим, на последних 20% торгуем)
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    test_data = df.iloc[split:].copy()

    # 3. Обучаем модель прямо здесь (с балансировкой)
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    model.fit(X_train_res, y_train_res)

    # 4. Модель делает предсказания для теста
    test_data['Signal'] = model.predict(X_test)

    # 5. Считаем экономику
    initial_balance = 1000
    balance = initial_balance
    commission = 0.001 # 0.1% OKX
    tp_level = 0.008   # Наш профит 0.8%
    
    # Мы будем считать, что если сигнал 1, мы заходим и выходим через 8 часов (как в Target)
    # Или просто считаем результат следующей цены через горизонт
    test_data['Next_Return'] = (test_data['Close'].shift(-8) - test_data['Close']) / test_data['Close']
    
    trades = 0
    wins = 0
    
    for i in range(len(test_data) - 8):
        if test_data['Signal'].iloc[i] == 1:
            trades += 1
            # Считаем чистую прибыль: изменение цены - комиссия за вход - комиссия за выход
            profit_pct = test_data['Next_Return'].iloc[i] - (commission * 2)
            balance *= (1 + profit_pct)
            
            if profit_pct > 0:
                wins += 1

    print("\n" + "="*30)
    print("💰 ИТОГИ ТОРГОВЛИ ИИ")
    print("="*30)
    print(f"Начальный баланс: ${initial_balance}")
    print(f"Конечный баланс: ${balance:.2f}")
    print(f"Всего сделок: {trades}")
    print(f"Прибыльных сделок: {wins}")
    if trades > 0:
        print(f"Процент побед (Winrate): {(wins/trades)*100:.2f}%")
    print(f"Чистый доход: {((balance - initial_balance)/initial_balance)*100:.2f}%")
    print("="*30)

if __name__ == "__main__":
    run_backtest()