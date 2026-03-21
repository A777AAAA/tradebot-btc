"""
Еженедельное переобучение модели
Запускается автоматически по расписанию (воскресенье 02:00 UTC)
"""

import os
import time
import joblib
import schedule
import traceback
from datetime import datetime

from config import RETRAIN_DAY, RETRAIN_HOUR, SYMBOL, TIMEFRAME
from auto_trainer import train_model
from telegram_notify import send_message

# Путь к модели (хранится локально на Render)
MODEL_PATH = "ai_brain.pkl"
SCALER_PATH = "scaler.pkl"


def save_model(model, scaler=None, metadata=None):
    """Сохраняет модель локально"""
    try:
        joblib.dump(model, MODEL_PATH)
        print(f"✅ Модель сохранена: {MODEL_PATH}")

        if scaler is not None:
            joblib.dump(scaler, SCALER_PATH)
            print(f"✅ Scaler сохранён: {SCALER_PATH}")

        return True
    except Exception as e:
        print(f"❌ Ошибка сохранения модели: {e}")
        return False


def load_model():
    """Загружает модель локально"""
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"✅ Модель загружена: {MODEL_PATH}")
            return model
        else:
            print(f"⚠️ Модель не найдена: {MODEL_PATH}")
            return None
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return None


def retrain_job():
    """
    Основная задача переобучения.
    Обучает модель → сохраняет локально
    """
    started_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n[Retrainer] 🔄 Начало переобучения: {started_at}")

    try:
        send_message(
            f"🔄 <b>Начало еженедельного переобучения</b>\n"
            f"📅 {started_at}\n"
            f"📊 {SYMBOL} | {TIMEFRAME}"
        )

        print("[Retrainer] Загрузка данных и обучение...")
        result = train_model()

        if result and result.get("success"):
            accuracy  = result.get("accuracy", 0)
            precision = result.get("precision", 0)
            recall    = result.get("recall", 0)
            n_samples = result.get("n_samples", 0)

            # train_model() уже сохраняет ai_brain.pkl внутри себя
            msg = (
                f"✅ <b>Переобучение завершено!</b>\n\n"
                f"📊 Результаты:\n"
                f"   Точность:  {accuracy:.1%}\n"
                f"   Precision: {precision:.1%}\n"
                f"   Recall:    {recall:.1%}\n"
                f"   Образцов:  {n_samples}\n\n"
                f"💾 Модель сохранена (ai_brain.pkl)"
            )
        else:
            error = result.get("error", "Неизвестная ошибка") if result else "Нет результата"
            msg = (
                f"❌ <b>Ошибка переобучения</b>\n"
                f"Причина: {error}"
            )

        send_message(msg)
        print(f"[Retrainer] {msg}")

    except Exception as e:
        error_msg = f"❌ <b>Критическая ошибка переобучения</b>\n{str(e)}"
        print(f"[Retrainer] ОШИБКА: {e}")
        traceback.print_exc()
        send_message(error_msg)


def schedule_retraining():
    """Настраивает расписание переобучения"""
    time_str = f"{RETRAIN_HOUR:02d}:00"

    if RETRAIN_DAY == "sunday":
        schedule.every().sunday.at(time_str).do(retrain_job)
    elif RETRAIN_DAY == "monday":
        schedule.every().monday.at(time_str).do(retrain_job)
    elif RETRAIN_DAY == "saturday":
        schedule.every().saturday.at(time_str).do(retrain_job)
    else:
        schedule.every().sunday.at(time_str).do(retrain_job)

    print(f"[Retrainer] ✅ Переобучение запланировано: {RETRAIN_DAY} в {time_str} UTC")


def run_retrainer_loop():
    """Запускает бесконечный цикл планировщика"""
    schedule_retraining()

    while True:
        try:
            schedule.run_pending()
            time.sleep(60)
        except Exception as e:
            print(f"[Retrainer] Ошибка в цикле: {e}")
            time.sleep(60)


def force_retrain():
    """Принудительный запуск переобучения"""
    print("[Retrainer] 🚀 Принудительный запуск переобучения...")
    retrain_job()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "force":
        force_retrain()
    else:
        print("[Retrainer] Запуск планировщика...")
        run_retrainer_loop()