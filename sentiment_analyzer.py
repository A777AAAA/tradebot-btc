"""
sentiment_analyzer.py v6.0 — LunarCrush API (реальный крипто-сентимент)
ИЗМЕНЕНИЯ v6.0:
  - Заменён бесплатный Mistral (который просто пересчитывал цену) на
    LunarCrush API — реальные данные из соцсетей и новостей
  - LunarCrush Free Tier: 10 req/min, не нужна кредитная карта
    Регистрация: https://lunarcrush.com/developers/api/authentication
  - Метрики: Galaxy Score, AltRank, social_volume, social_dominance
  - Fallback: технический сентимент на основе RSI/ATR если API недоступен
  - Кэш 15 минут (не тратим лимиты API)
"""

import os
import json
import time
import logging
import requests
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

LUNARCRUSH_API_KEY = os.getenv("LUNARCRUSH_API_KEY", "")

# Кэш: не запрашиваем API чаще раза в 15 минут
_cache: dict = {}
_cache_ttl   = 15 * 60  # секунды


def _get_cached(key: str) -> dict | None:
    if key in _cache:
        data, ts = _cache[key]
        if time.time() - ts < _cache_ttl:
            return data
    return None


def _set_cache(key: str, data: dict):
    _cache[key] = (data, time.time())


def get_lunarcrush_sentiment(symbol: str = "TON") -> dict:
    """
    Получает сентимент из LunarCrush API.

    Возвращаемые метрики:
      - galaxy_score: 0-100, общий health score монеты
      - alt_rank: ранг по активности (ниже = лучше)
      - social_volume: кол-во упоминаний за 24ч
      - social_dominance: % от общего крипто-трафика
      - sentiment: bullish/bearish/neutral
      - confidence: 0.0-1.0
    """
    cache_key = f"lunar_{symbol}"
    cached = _get_cached(cache_key)
    if cached:
        logger.debug(f"[Sentiment] LunarCrush кэш для {symbol}")
        return cached

    if not LUNARCRUSH_API_KEY:
        logger.debug("[Sentiment] LUNARCRUSH_API_KEY не задан, используем технический fallback")
        return {}

    try:
        # LunarCrush v4 API
        url = f"https://lunarcrush.com/api4/public/coins/{symbol.lower()}/v1"
        headers = {"Authorization": f"Bearer {LUNARCRUSH_API_KEY}"}

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 401:
            logger.warning("[Sentiment] LunarCrush: неверный API ключ")
            return {}

        if response.status_code == 429:
            logger.warning("[Sentiment] LunarCrush: лимит запросов, используем кэш/fallback")
            return {}

        if response.status_code != 200:
            logger.warning(f"[Sentiment] LunarCrush HTTP {response.status_code}")
            return {}

        data = response.json()
        coin = data.get("data", {})

        if not coin:
            return {}

        galaxy_score     = float(coin.get("galaxy_score", 50))
        alt_rank         = int(coin.get("alt_rank", 500))
        social_volume    = float(coin.get("social_volume", 0))
        social_dominance = float(coin.get("social_dominance", 0))
        price_score      = float(coin.get("price_score", 50))
        sentiment_score  = float(coin.get("sentiment", 3))  # 1-5 scale

        # Определяем направление сентимента
        # galaxy_score > 60 и sentiment_score > 3.5 = bullish
        # galaxy_score < 40 или sentiment_score < 2.5 = bearish
        bull_signals = 0
        bear_signals = 0

        if galaxy_score >= 60:
            bull_signals += 2
        elif galaxy_score <= 40:
            bear_signals += 2

        if sentiment_score >= 3.5:
            bull_signals += 2
        elif sentiment_score <= 2.5:
            bear_signals += 2

        if alt_rank <= 50:
            bull_signals += 1
        elif alt_rank >= 200:
            bear_signals += 1

        if social_dominance >= 2.0:
            bull_signals += 1

        total_signals = bull_signals + bear_signals
        if total_signals == 0:
            sentiment = "neutral"
            confidence = 0.5
        elif bull_signals > bear_signals:
            sentiment  = "bullish"
            confidence = min(0.5 + (bull_signals - bear_signals) / total_signals * 0.4, 0.9)
        elif bear_signals > bull_signals:
            sentiment  = "bearish"
            confidence = min(0.5 + (bear_signals - bull_signals) / total_signals * 0.4, 0.9)
        else:
            sentiment  = "neutral"
            confidence = 0.5

        result = {
            "sentiment":        sentiment,
            "confidence":       round(confidence, 2),
            "galaxy_score":     galaxy_score,
            "alt_rank":         alt_rank,
            "social_volume":    social_volume,
            "social_dominance": social_dominance,
            "sentiment_score":  sentiment_score,
            "source":           "lunarcrush",
            "reason": (
                f"Galaxy={galaxy_score:.0f} "
                f"AltRank={alt_rank} "
                f"Sentiment={sentiment_score:.1f}/5 "
                f"SocDom={social_dominance:.2f}%"
            )
        }

        _set_cache(cache_key, result)
        logger.info(
            f"[Sentiment] LunarCrush {symbol}: {sentiment} "
            f"conf={confidence:.2f} galaxy={galaxy_score:.0f} "
            f"rank={alt_rank}"
        )
        return result

    except requests.exceptions.Timeout:
        logger.warning("[Sentiment] LunarCrush timeout")
        return {}
    except Exception as e:
        logger.warning(f"[Sentiment] LunarCrush ошибка: {e}")
        return {}


def _technical_sentiment(price: float, change_24h: float, volume: float,
                          rsi: float = None) -> dict:
    """
    Технический fallback сентимент когда LunarCrush недоступен.
    Использует реальные рыночные данные, а не AI-генерацию.
    """
    bull_score = 0
    bear_score = 0

    # 24h изменение
    if change_24h > 5:
        bull_score += 3
    elif change_24h > 2:
        bull_score += 2
    elif change_24h > 0:
        bull_score += 1
    elif change_24h < -5:
        bear_score += 3
    elif change_24h < -2:
        bear_score += 2
    elif change_24h < 0:
        bear_score += 1

    # RSI если доступен
    if rsi is not None:
        if rsi < 30:
            bull_score += 2  # перепродан = потенциал роста
        elif rsi > 70:
            bear_score += 2  # перекуплен = потенциал падения

    total = bull_score + bear_score
    if total == 0:
        return {"sentiment": "neutral", "confidence": 0.5,
                "source": "technical", "reason": "нейтральные условия"}

    if bull_score > bear_score:
        conf = min(0.5 + (bull_score - bear_score) / total * 0.3, 0.75)
        return {"sentiment": "bullish", "confidence": conf,
                "source": "technical",
                "reason": f"change_24h={change_24h:+.1f}% RSI={rsi or '?'}"}
    elif bear_score > bull_score:
        conf = min(0.5 + (bear_score - bull_score) / total * 0.3, 0.75)
        return {"sentiment": "bearish", "confidence": conf,
                "source": "technical",
                "reason": f"change_24h={change_24h:+.1f}% RSI={rsi or '?'}"}
    else:
        return {"sentiment": "neutral", "confidence": 0.5,
                "source": "technical", "reason": "смешанные сигналы"}


def get_market_sentiment(price: float, change_24h: float, volume: float,
                         rsi: float = None, symbol: str = "TON") -> dict:
    """
    Главная функция сентимента.
    1. Пробует LunarCrush API (реальный сентимент соцсетей)
    2. Fallback: технический сентимент на основе рыночных данных
    """
    # Пробуем LunarCrush
    lunar = get_lunarcrush_sentiment(symbol)
    if lunar:
        return lunar

    # Fallback: технический
    result = _technical_sentiment(price, change_24h, volume, rsi)
    logger.debug(f"[Sentiment] Technical fallback: {result['sentiment']} conf={result['confidence']:.2f}")
    return result


def sentiment_to_signal_boost(sentiment: dict, base_signal: str) -> float:
    """
    Преобразует сентимент в корректировку уверенности сигнала.

    Логика:
      - Совпадение сентимента с сигналом → усиливаем (до 1.25)
      - Противоречие → ослабляем (до 0.75)
      - Нейтральный → без изменений (1.0)

    Коэффициент зависит от уверенности сентимента и источника.
    LunarCrush даёт больший вес чем технический fallback.
    """
    s    = sentiment.get("sentiment", "neutral")
    conf = sentiment.get("confidence", 0.5)
    src  = sentiment.get("source", "technical")

    # Максимальный буст: LunarCrush = ±25%, технический = ±15%
    max_boost = 0.25 if src == "lunarcrush" else 0.15

    if base_signal == "BUY" and s == "bullish":
        return 1.0 + (max_boost * conf)
    elif base_signal == "BUY" and s == "bearish":
        return 1.0 - (max_boost * conf)
    elif base_signal == "SELL" and s == "bearish":
        return 1.0 + (max_boost * conf)
    elif base_signal == "SELL" and s == "bullish":
        return 1.0 - (max_boost * conf)

    return 1.0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Тест LunarCrush API ===")
    if not LUNARCRUSH_API_KEY:
        print("⚠️ LUNARCRUSH_API_KEY не задан — тестируем технический fallback")

    result = get_market_sentiment(
        price=5.234,
        change_24h=2.5,
        volume=1_500_000,
        rsi=55,
        symbol="TON"
    )
    print(f"Сентимент: {result}")

    boost = sentiment_to_signal_boost(result, "BUY")
    print(f"Boost для BUY: {boost:.2f}x")