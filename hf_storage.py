import os
import joblib
import pandas as pd
import json
from huggingface_hub import HfApi, hf_hub_download, login

# === НАСТРОЙКИ ===
REPO_ID = "LUCK8888/trading-bot-data"
REPO_TYPE = "dataset"

# Автоматический вход по токену из переменной окружения
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("⚠️ HF_TOKEN не найден в переменных окружения. Запись в Hub не будет работать.")

api = HfApi()


def save_model_to_hub(model, metadata, local_path="ai_brain.pkl"):
    """Сохраняет модель и метаданные в датасет на Hugging Face."""
    joblib.dump(model, local_path)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo="ai_brain.pkl",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )
    with open("metadata.json", "w") as f:
        json.dump(metadata, f)
    api.upload_file(
        path_or_fileobj="metadata.json",
        path_in_repo="metadata.json",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )
    os.remove(local_path)
    os.remove("metadata.json")
    print("✅ Модель и метаданные сохранены в Hub.")


def load_model_from_hub(local_path="ai_brain.pkl"):
    """Загружает модель и метаданные из датасета."""
    try:
        hf_hub_download(
            repo_id=REPO_ID,
            filename="ai_brain.pkl",
            repo_type=REPO_TYPE,
            local_dir=".",
            local_dir_use_symlinks=False,
        )
        model = joblib.load(local_path)

        hf_hub_download(
            repo_id=REPO_ID,
            filename="metadata.json",
            repo_type=REPO_TYPE,
            local_dir=".",
            local_dir_use_symlinks=False,
        )
        with open("metadata.json") as f:
            metadata = json.load(f)
        os.remove("metadata.json")
        print("✅ Модель и метаданные загружены из Hub.")
        return model, metadata
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return None, {}


def update_historical_data(new_data: pd.DataFrame):
    """Добавляет новые данные к историческому CSV и сохраняет обратно в Hub."""
    try:
        hf_hub_download(
            repo_id=REPO_ID,
            filename="historical_data.csv",
            repo_type=REPO_TYPE,
            local_dir=".",
            local_dir_use_symlinks=False,
        )
        old_df = pd.read_csv("historical_data.csv", index_col=0, parse_dates=True)
        combined = pd.concat([old_df, new_data]).drop_duplicates().sort_index()
        os.remove("historical_data.csv")
    except Exception:
        print("Файл historical_data.csv не найден, создаём новый.")
        combined = new_data

    combined.to_csv("historical_data.csv")
    api.upload_file(
        path_or_fileobj="historical_data.csv",
        path_in_repo="historical_data.csv",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )
    os.remove("historical_data.csv")
    print(f"✅ Исторические данные обновлены. Всего записей: {len(combined)}")
    return combined


def download_historical_data():
    """Скачивает исторический CSV из Hub и возвращает DataFrame."""
    try:
        hf_hub_download(
            repo_id=REPO_ID,
            filename="historical_data.csv",
            repo_type=REPO_TYPE,
            local_dir=".",
            local_dir_use_symlinks=False,
        )
        df = pd.read_csv("historical_data.csv", index_col=0, parse_dates=True)
        os.remove("historical_data.csv")
        return df
    except Exception:
        print("⚠️ historical_data.csv не найден в Hub, возвращаем пустой DataFrame.")
        return pd.DataFrame()


# === АЛИАСЫ для совместимости с weekly_retrainer.py ===

def save_model(model, scaler=None, metadata=None):
    """Обёртка над save_model_to_hub для совместимости"""
    if metadata is None:
        metadata = {}
    try:
        # Сохраняем scaler если передан
        if scaler is not None:
            metadata["has_scaler"] = True
            joblib.dump(scaler, "scaler.pkl")
            api.upload_file(
                path_or_fileobj="scaler.pkl",
                path_in_repo="scaler.pkl",
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
            )
            os.remove("scaler.pkl")
            print("✅ Scaler сохранён в Hub.")

        save_model_to_hub(model, metadata)
        return True

    except Exception as e:
        print(f"❌ Ошибка save_model: {e}")
        return False


def load_model():
    """Обёртка над load_model_from_hub для совместимости"""
    try:
        model, metadata = load_model_from_hub()
        return model
    except Exception as e:
        print(f"❌ Ошибка load_model: {e}")
        return None