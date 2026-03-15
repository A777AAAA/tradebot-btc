import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

def train_brain_v2():
    print("🧠 Загрузка данных для обучения 'Мозга v2'...")
    
    # 1. Загружаем обогащенный файл v2
    try:
        df = pd.read_csv("ml_ready_ton_data_v2.csv", index_col='Timestamp', parse_dates=True)
    except FileNotFoundError:
        print("❌ Ошибка: файл ml_ready_ton_data_v2.csv не найден!")
        return

    # 2. Выбираем признаки (убрали Above_SMA, добавили BB и MACD)
    features = ['RSI', 'ATR', 'BB_Dist_Lower', 'MACD_Hist', 'Vol_Change', 'Price_Change_3h']
    X = df[features]
    y = df['Target']

    # 3. Разделяем на тренировочную и тестовую выборки (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(f"📊 До балансировки: сигналов '1' = {sum(y_train)}")

    # 4. ПРИМЕНЯЕМ SMOTE (Создаем синтетические примеры успеха)
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    print(f"🧬 После SMOTE: сигналов '1' = {sum(y_train_res)} (теперь баланс 1:1)")

    # 5. Обучаем модель Random Forest
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    model.fit(X_train_res, y_train_res)

    # 6. Экзамен модели
    predictions = model.predict(X_test)
    
    print("\n" + "="*30)
    print("🎯 РЕЗУЛЬТАТЫ ЭКЗАМЕНА (v2)")
    print("="*30)
    print(f"Общая точность: {accuracy_score(y_test, predictions)*100:.2f}%")
    print("\nДетальный отчет:")
    print(classification_report(y_test, predictions))

    # Важность параметров
    importances = pd.Series(model.feature_importances_, index=features)
    print("\n💡 На что модель смотрит теперь:")
    print(importances.sort_values(ascending=False))

if __name__ == "__main__":
    train_brain_v2()