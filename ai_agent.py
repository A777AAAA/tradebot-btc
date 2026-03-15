import csv
import os

def log_trade(asset, prediction, confidence, signal_price):
    file_exists = os.path.isfile('trading_history.csv')
    with open('trading_history.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Asset', 'Prediction', 'Confidence', 'Price'])
        writer.writerow([pd.Timestamp.now(), asset, prediction, confidence, signal_price])
    print(">>> Прогноз сохранен в систему обучения.")