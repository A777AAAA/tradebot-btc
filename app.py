from flask import Flask
import threading
import time
import logging
from live_signal import get_signal
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

def check_market_periodically(interval_seconds=3600):
    """
    Функция, которая будет запущена в отдельном потоке и каждые interval_seconds
    вызывать get_signal() и обрабатывать сигнал.
    """
    while True:
        try:
            logging.info("🔍 Запуск проверки рынка...")
            pred, prob = get_signal()
            if pred is not None:
                if pred == 1 and prob > 0.6:
                    message = f"🚀 СИГНАЛ НА ПОКУПКУ!\nВероятность: {prob:.2%}"
                    logging.info(message)
                    # Здесь можно вызвать отправку в Telegram (добавим позже)
                else:
                    logging.info(f"📉 Нет сигнала. Вероятность: {prob:.2%}")
            else:
                logging.error("❌ Ошибка получения сигнала")
        except Exception as e:
            logging.exception("Ошибка при проверке рынка:")
        time.sleep(interval_seconds)

@app.route('/')
def home():
    return "Trading Bot is running! Check logs for signals."

if __name__ == '__main__':
    # Запускаем фоновый поток для проверки рынка (каждый час)
    threading.Thread(target=check_market_periodically, args=(3600,), daemon=True).start()
    # Запускаем Flask-сервер на порту 7860
    app.run(host='0.0.0.0', port=7860)