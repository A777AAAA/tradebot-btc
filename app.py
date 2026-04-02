"""
TradeBot v3.0 — Paper Trading Edition
Главный файл: связывает все модули
"""

import threading
import time
import traceback
import logging
import os
from datetime import datetime, timezone
from flask import Flask

from config import (
    SYMBOL, TIMEFRAME, MIN_CONFIDENCE, STRONG_SIGNAL,
    SIGNAL_INTERVAL_MINUTES, validate_config
)
from live_signal        import get_live_signal
from sentiment_analyzer import get_market_sentiment, sentiment_to_signal_boost
from telegram_notify    import send_message
from trade_archive      import get_statistics
from auto_trainer       import train_model
from paper_trader       import (
    open_trade, monitor_trades,
    get_stats, format_stats_message
)
from backtest_engine    import run_backtest, format_backtest_message

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

PAPER_SYMBOL = SYMBOL  # TON/USDT из config


# ═══════════════════════════════════════════
# HEALTHCHECK СЕРВЕР
# ═══════════════════════════════════════════
health_app = Flask(__name__)

@health_app.route("/health")
def health():
    return {"status": "ok", "bot": "TradeBot v3.0"}, 200

@health_app.route("/")
def index():
    stats = get_statistics()
    paper = get_stats()
    return {
        "bot":           "TradeBot v3.0 — Paper Trading",
        "symbol":        SYMBOL,
        "paper_balance": paper["balance"],
        "paper_winrate": paper["winrate"],
        "stats":         stats
    }, 200

def run_health_server():
    # ✅ ИСПРАВЛЕНО: берём PORT из env (Render требует это)
    port = int(os.environ.get("PORT", 8080))
    health_app.run(
        host        = "0.0.0.0",
        port        = port,
        debug       = False,
        use_reloader= False
    )


# ═══════════════════════════════════════════
# ОСНОВНОЙ ТОРГОВЫЙ ЦИКЛ (каждый час)
# ═══════════════════════════════════════════
def trading_loop():
    logger.info(f"🚀 Торговый цикл запущен | {SYMBOL} | {TIMEFRAME}")

    while True:
        try:
            now = datetime.now(timezone.utc).strftime("%H:%M UTC")
            logger.info(f"⏰ Цикл: {now}")

            # ── Шаг 1: ML сигнал ───────────────────────────
            logger.info("🔍 Шаг 1: get_live_signal()")
            signal_data = get_live_signal()

            if not signal_data:
                logger.warning("⚠️ Сигнал не получен — модель ещё обучается")
                time.sleep(SIGNAL_INTERVAL_MINUTES * 60)
                continue

            signal     = signal_data.get("signal",     "HOLD")
            confidence = signal_data.get("confidence",  0.0)
            price      = signal_data.get("price",       0.0)

            logger.info(
                f"📊 Сигнал: {signal} | "
                f"Уверенность: {confidence:.1%} | "
                f"Цена: {price}"
            )

            # ── Шаг 2: Мониторинг Paper сделок ─────────────
            logger.info("🔍 Шаг 2: monitor_trades()")
            closed = monitor_trades(PAPER_SYMBOL)

            for trade in closed:
                emoji  = "✅" if trade["result"] == "WIN" else "❌"
                result = "ПРИБЫЛЬ" if trade["result"] == "WIN" else "УБЫТОК"
                send_message(
                    f"{emoji} <b>Виртуальная сделка — {result}</b>\n\n"
                    f"📊 {trade['signal']} {trade['symbol']}\n"
                    f"🔵 Вход:    <b>${trade['price_open']:.4f}</b>\n"
                    f"🔴 Выход:   <b>${trade['price_close']:.4f}</b>\n"
                    f"💰 P&L:     <b>{trade['pnl_pct']:+.2f}%"
                    f" (${trade['pnl_usd']:+.2f})</b>\n"
                    f"🏁 Причина: <b>{trade['closed_by']}</b>"
                )
                send_message(format_stats_message(get_stats()))

            # ── Шаг 3: Открытие Paper сделки ───────────────
            logger.info("🔍 Шаг 3: проверка сигнала")
            if signal in ("BUY", "SELL") and confidence >= MIN_CONFIDENCE and signal != "HOLD":

                change_24h    = signal_data.get("change_24h", 0.0)
                volume        = signal_data.get("volume",     0.0)

                # ✅ ИСПРАВЛЕНО: защита если sentiment_analyzer упал
                try:
                    sentiment     = get_market_sentiment(price, change_24h, volume)
                    sentiment_str = sentiment.get("sentiment", "neutral")
                    boost         = sentiment_to_signal_boost(sentiment, signal)
                except Exception as se:
                    logger.warning(f"⚠️ Sentiment error: {se} — используем нейтральный")
                    sentiment_str = "neutral"
                    boost         = 1.0

                adj_conf = min(confidence * boost, 0.99)

                if adj_conf >= MIN_CONFIDENCE:
                    strength = "🔥 СИЛЬНЫЙ" if adj_conf >= STRONG_SIGNAL else "📊 Обычный"
                    emoji    = "🟢" if signal == "BUY" else "🔴"

                    trade = open_trade(signal, price, adj_conf, PAPER_SYMBOL)

                    if trade:
                        send_message(
                            f"{emoji} <b>{signal} {PAPER_SYMBOL}</b> {strength}\n\n"
                            f"💵 Цена входа:   <b>${price:.4f}</b>\n"
                            f"✅ Тейк-профит:  <b>${trade['tp']:.4f}</b>\n"
                            f"🛑 Стоп-лосс:    <b>${trade['sl']:.4f}</b>\n"
                            f"🎯 Уверенность:  <b>{adj_conf:.1%}</b>\n"
                            f"🧠 Настроение:   <b>{sentiment_str}</b>\n"
                            f"📝 Режим:        <b>Paper Trading</b>\n"
                            f"⏰ Время:        <b>{now}</b>"
                        )

            # ── Шаг 4: Пауза ───────────────────────────────
            logger.info(f"💤 Следующий цикл через {SIGNAL_INTERVAL_MINUTES} мин")
            time.sleep(SIGNAL_INTERVAL_MINUTES * 60)

        except Exception as e:
            logger.error(f"❌ Ошибка торгового цикла: {e}")
            traceback.print_exc()
            time.sleep(60)


# ═══════════════════════════════════════════
# ПЕРЕОБУЧЕНИЕ (каждые 6 часов)
# ═══════════════════════════════════════════
def retrainer_loop():
    logger.info("🧠 Retrainer запущен (каждые 6 часов)")

    # Первое обучение через 30 сек после старта
    time.sleep(30)
    _do_retrain()

    while True:
        time.sleep(6 * 60 * 60)
        _do_retrain()


def _do_retrain():
    try:
        logger.info("🔄 Переобучение модели...")
        result = train_model()

        if result.get("success"):
            top     = result.get("top_features", [])
            top_str = "\n".join(
                f"  {n:<18} {v:.3f}" for n, v in top[:3]
            ) if top else "—"

            send_message(
                f"🧠 <b>Модель переобучена!</b>\n\n"
                f"✅ Accuracy:     <b>{result['accuracy']:.1%}</b>\n"
                f"🎯 Precision:    <b>{result['precision']:.1%}</b>\n"
                f"📊 Recall:       <b>{result['recall']:.1%}</b>\n"
                f"📚 Примеров:     <b>{result['n_samples']}</b>\n"
                f"📝 Paper сделок: <b>{result['paper_trades']}</b>\n\n"
                f"🏆 Топ признаки:\n<code>{top_str}</code>"
            )
        else:
            err = result.get("error", "unknown")
            logger.error(f"Обучение не удалось: {err}")
            send_message(f"⚠️ <b>Обучение не удалось</b>\n\n{err}")

    except Exception as e:
        logger.error(f"❌ Ошибка retrainer: {e}")
        traceback.print_exc()


# ═══════════════════════════════════════════
# БЭКТЕСТ (каждые 12 часов)
# ═══════════════════════════════════════════
def backtest_loop():
    logger.info("🔬 Backtest loop запущен (каждые 12 часов)")

    # Первый бэктест через 10 минут (после обучения)
    time.sleep(10 * 60)
    _do_backtest()

    while True:
        time.sleep(12 * 60 * 60)
        _do_backtest()


def _do_backtest():
    try:
        logger.info("🔬 Запуск бэктеста...")
        result = run_backtest(
            symbol        = PAPER_SYMBOL,
            limit         = 3000,
            tp_pct        = 0.03,
            sl_pct        = 0.015,
            start_balance = 600.0
        )
        send_message(format_backtest_message(result))
    except Exception as e:
        logger.error(f"❌ Ошибка бэктеста: {e}")
        traceback.print_exc()


# ═══════════════════════════════════════════
# ЕЖЕДНЕВНАЯ СТАТИСТИКА
# ═══════════════════════════════════════════
def daily_stats_loop():
    while True:
        try:
            time.sleep(24 * 60 * 60)
            paper = get_stats()
            trade = get_statistics()
            send_message(
                f"🌅 <b>Ежедневный отчёт</b>\n\n"
                f"━━━ Paper Trading ━━━\n"
                + format_stats_message(paper) +
                f"\n\n━━━ Сигнальная статистика ━━━\n"
                f"📈 Всего сигналов: <b>{trade['total']}</b>\n"
                f"✅ Прибыльных:     <b>{trade['wins']}</b>\n"
                f"❌ Убыточных:      <b>{trade['losses']}</b>\n"
                f"🎯 Винрейт:        <b>{trade['winrate']}%</b>"
            )
        except Exception as e:
            logger.error(f"❌ Ошибка daily stats: {e}")
            time.sleep(60 * 60)


# ═══════════════════════════════════════════
# ТОЧКА ВХОДА
# ═══════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("  TradeBot v3.0 — Paper Trading Edition")
    print("=" * 50)

    # Проверка конфига
    missing = validate_config()
    if missing:
        msg = f"⚠️ Отсутствуют переменные:\n" + "\n".join(missing)
        logger.error(msg)
        send_message(msg)
        exit(1)

    logger.info("✅ Конфигурация проверена")

    # Стартовое сообщение
    paper = get_stats()
    send_message(
        f"🤖 <b>TradeBot v3.0 запущен!</b>\n\n"
        f"📊 Пара:         <b>{SYMBOL}</b>\n"
        f"⏱️ Таймфрейм:   <b>{TIMEFRAME}</b>\n"
        f"📝 Режим:        <b>Paper Trading</b>\n\n"
        f"💰 Баланс:       <b>${paper['balance']:.2f}</b>\n"
        f"📋 Сделок:       <b>{paper['total_trades']}</b>\n"
        f"🎯 Winrate:      <b>{paper['winrate']}%</b>\n\n"
        f"⏱️ Анализ:       <b>каждый час</b>\n"
        f"🔄 Обучение:     <b>каждые 6 часов</b>\n"
        f"🔬 Бэктест:      <b>каждые 12 часов</b>\n"
        f"📊 Отчёт:        <b>каждые 24 часа</b>"
    )

    # Запускаем все потоки
    threads = [
        threading.Thread(target=trading_loop,      daemon=True, name="TradingLoop"),
        threading.Thread(target=retrainer_loop,    daemon=True, name="Retrainer"),
        threading.Thread(target=backtest_loop,     daemon=True, name="Backtest"),
        threading.Thread(target=daily_stats_loop,  daemon=True, name="DailyStats"),
        threading.Thread(target=run_health_server, daemon=True, name="HealthCheck"),
    ]

    for t in threads:
        t.start()
        logger.info(f"✅ Поток запущен: {t.name}")

    logger.info("🚀 Все системы запущены!")

    # Держим главный поток
    while True:
        time.sleep(60)