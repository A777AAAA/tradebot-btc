from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import datetime
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from hf_storage import save_model_to_hub, update_historical_data, load_model_from_hub
import pandas as pd
import ccxt
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

def tune_hyperparameters(X, y):
    """
    Запускает случайный поиск по сетке гиперпараметров для RandomForest.
    Возвращает лучшую модель и словарь с лучшими параметрами.
    """
    param_dist = {
        'n_estimators': randint(50, 300),
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2']
    }
    
    base_model = RandomForestClassifier(
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print("🔄 Запуск подбора гиперпараметров (может занять 10-20 минут)...")
    random_search.fit(X, y)
    
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    print(f"✅ Лучшие параметры: {best_params}")
    return best_model, best_params

def get_4h_features(symbol='TON/USDT', limit=500):
    """
    Скачивает 4-часовые свечи, рассчитывает индикаторы и возвращает DataFrame
    с индексом timestamp и колонками признаков.
    """
    exchange = ccxt.okx()
    ohlcv_4h = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=limit)
    df_4h = pd.DataFrame(ohlcv_4h, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df_4h['Timestamp'] = pd.to_datetime(df_4h['Timestamp'], unit='ms')
    df_4h.set_index('Timestamp', inplace=True)
    
    # Расчёт индикаторов на 4h
    df_4h['EMA50_4h'] = ta.ema(df_4h['Close'], length=50)
    df_4h['RSI_4h'] = ta.rsi(df_4h['Close'], length=14)
    df_4h['ATR_4h'] = ta.atr(df_4h['High'], df_4h['Low'], df_4h['Close'], length=14)
    
    # Расчёт MACD и поиск колонки гистограммы
    macd_4h = ta.macd(df_4h['Close'])
    macdh_col = None
    for col in macd_4h.columns:
        if 'MACDh' in col.upper():
            macdh_col = col
            break
    if macdh_col is None:
        print("⚠️ Не найдена колонка MACDh в 4h данных, будет заполнена 0.")
        df_4h['MACD_Hist_4h'] = 0.0
    else:
        df_4h['MACD_Hist_4h'] = macd_4h[macdh_col]
    
    df_4h.dropna(inplace=True)
    if df_4h.empty:
        print("⚠️ 4h DataFrame пуст после удаления NaN. Возвращаем пустой DataFrame.")
        return pd.DataFrame(columns=['EMA50_4h', 'RSI_4h', 'ATR_4h', 'MACD_Hist_4h'])
    return df_4h[['EMA50_4h', 'RSI_4h', 'ATR_4h', 'MACD_Hist_4h']]

def update_and_train():
    print("🔄 Запуск модуля самообучения (Data Loop)...")
    file_name = "ml_ready_ton_data_v2.csv"
    
    # 1. Загружаем старую базу знаний
    if os.path.exists(file_name):
        df_old = pd.read_csv(file_name, index_col='Timestamp', parse_dates=True)
        print(f"📊 Найдена база: {len(df_old)} записей.")
        
        # Удаляем старые 4h колонки, если они есть, чтобы избежать дублирования при merge_asof
        cols_to_drop = ['EMA50_4h', 'RSI_4h', 'ATR_4h', 'MACD_Hist_4h']
        existing_cols = [col for col in cols_to_drop if col in df_old.columns]
        if existing_cols:
            df_old = df_old.drop(columns=existing_cols)
            print(f"🧹 Удалены устаревшие 4h колонки: {existing_cols}")
    else:
        print("⚠️ Старая база не найдена, начинаем с нуля.")
        df_old = pd.DataFrame()

    # 2. Скачиваем свежие 1-часовые данные с OKX
    print("🌐 Скачиваем новые свечи с биржи (1h)...")
    exchange = ccxt.okx()
    ohlcv = exchange.fetch_ohlcv('TON/USDT', timeframe='1h', limit=100)
    df_new = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df_new['Timestamp'] = pd.to_datetime(df_new['Timestamp'], unit='ms')
    df_new.set_index('Timestamp', inplace=True)

    # Сохраняем новые данные в облако
    update_historical_data(df_new) 
    print("✅ История цен обновлена в облаке!")

    # 3. Объединяем и удаляем дубликаты
    df_combined = pd.concat([df_old, df_new])
    df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
    df_combined.sort_index(inplace=True)

    # --- MTF: Добавляем 4-часовые признаки ---
    print("📈 Скачиваем 4h данные для расчёта контекста...")
    df_4h_features = get_4h_features()
    if not df_4h_features.empty:
        # Присоединяем к 1h данным: для каждой 1h свечи берём последние доступные 4h индикаторы
        df_combined_sorted = df_combined.sort_index()
        df_4h_sorted = df_4h_features.sort_index()
        
        # Приводим индексы к единому типу (datetime64[ns])
        df_combined_sorted.index = df_combined_sorted.index.astype('datetime64[ns]')
        df_4h_sorted.index = df_4h_sorted.index.astype('datetime64[ns]')
        
        # Слияние asof: для каждого момента времени в df_combined берём ближайшую предыдущую 4h запись
        df_combined_with_4h = pd.merge_asof(
            df_combined_sorted, 
            df_4h_sorted, 
            left_index=True, 
            right_index=True,
            direction='backward'
        )
        print(f"✅ Добавлены 4h признаки. Формат: {df_combined_with_4h.shape}")
    else:
        print("⚠️ Не удалось получить 4h данные, продолжаем без них.")
        df_combined_with_4h = df_combined.copy()
        # Добавим пустые колонки, чтобы не ломать список признаков
        for col in ['EMA50_4h', 'RSI_4h', 'ATR_4h', 'MACD_Hist_4h']:
            df_combined_with_4h[col] = float('nan')
    
    df_combined = df_combined_with_4h

    # 4. Пересчитываем индикаторы (на 1h)
    print("🧮 Пересчет индикаторов и целей...")
    df_combined['RSI'] = ta.rsi(df_combined['Close'], length=14)
    df_combined['ATR'] = ta.atr(df_combined['High'], df_combined['Low'], df_combined['Close'], length=14)
    
    bb = ta.bbands(df_combined['Close'], length=20, std=2)
    bbl_col = [col for col in bb.columns if 'BBL' in col.upper()][0]
    df_combined['BB_Dist_Lower'] = (bb[bbl_col] - df_combined['Close']) / df_combined['Close'] * 100
    
    macd = ta.macd(df_combined['Close'])
    macd_col = [col for col in macd.columns if 'MACDH' in col.upper()][0]
    df_combined['MACD_Hist'] = macd[macd_col]
    
    df_combined['Vol_Change'] = df_combined['Volume'].pct_change() * 100
    df_combined['Price_Change_3h'] = df_combined['Close'].pct_change(3) * 100

    df_combined['Future_Close'] = df_combined['Close'].shift(-8)
    df_combined['Target'] = ((df_combined['Future_Close'] - df_combined['Close']) / df_combined['Close'] > 0.008).astype(int)

    df_clean = df_combined.dropna().copy()
    df_clean.to_csv(file_name)

    # 5. ОБУЧЕНИЕ С АВТОТЮНИНГОМ
    print("🧠 Тренировка нейросети...")
    features = [
        'RSI', 'ATR', 'BB_Dist_Lower', 'MACD_Hist', 'Vol_Change', 'Price_Change_3h',
        'EMA50_4h', 'RSI_4h', 'ATR_4h', 'MACD_Hist_4h'
    ]
    
    # Проверяем наличие всех колонок
    missing_cols = [col for col in features if col not in df_clean.columns]
    if missing_cols:
        print(f"⚠️ Внимание! Отсутствуют колонки: {missing_cols}. Добавляем их с нулями.")
        for col in missing_cols:
            df_clean[col] = 0.0
    else:
        print("✅ Все необходимые колонки присутствуют.")
    
    X = df_clean[features]
    y = df_clean['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Применяем SMOTE для балансировки классов
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    # Загружаем метаданные для проверки даты тюнинга
    try:
        old_model, metadata = load_model_from_hub()
        print("✅ Модель загружена из Hub.")
    except Exception as e:
        print("⚠️ Не удалось загрузить модель из Hub, создаём новую.")
        old_model = None
        metadata = {'last_tuning_date': None, 'best_params': None}

    today = datetime.now()
    need_tuning = False

    if metadata.get('last_tuning_date') is None:
        print("🆕 Никогда не делали тюнинг – запускаем подбор параметров.")
        need_tuning = True
    else:
        last_tune = datetime.fromisoformat(metadata['last_tuning_date'])
        days_since_tune = (today - last_tune).days
        if days_since_tune >= 7:
            print(f"📅 Последний тюнинг был {days_since_tune} дней назад – запускаем повторный подбор.")
            need_tuning = True
        else:
            print(f"⏳ Последний тюнинг был {days_since_tune} дней назад – используем текущие параметры.")

    if need_tuning:
        model, best_params = tune_hyperparameters(X_res, y_res)
        metadata['best_params'] = best_params
        metadata['last_tuning_date'] = today.isoformat()
    else:
        if metadata.get('best_params'):
            params = metadata['best_params'].copy()
            print(f"🔄 Дообучаем модель с параметрами: {params}")
            model = RandomForestClassifier(
                random_state=42,
                class_weight='balanced',
                n_jobs=-1,
                **params
            )
        else:
            print("🔄 Используем стандартные параметры (n_estimators=200).")
            model = RandomForestClassifier(
                n_estimators=200,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        # Обучаем модель на сбалансированных данных
        model.fit(X_res, y_res)

    # Оцениваем точность на тестовых данных
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Точность модели: {acc:.4f}")

    # Обновляем метаданные
    metadata['last_training'] = today.isoformat()
    metadata['accuracy'] = f"{acc:.4f}"
    
    # Сохраняем среднее ATR для риск-менеджмента
    atr_mean_value = float(df_clean['ATR'].mean())
    metadata['atr_mean'] = atr_mean_value
    print(f"📊 Средний ATR сохранен в метаданных: {atr_mean_value:.4f}")

    # Сохраняем модель и метаданные в Hub
    save_model_to_hub(model, metadata)
    print(f"🚀 Модель сохранена в Hub! Точность: {acc:.4f}")

    # Локальное сохранение (опционально)
    joblib.dump(df_clean['ATR'].mean(), 'atr_mean.pkl')
    print("💾 Локальные файлы обновлены. Система готова!")

if __name__ == "__main__":
    update_and_train()