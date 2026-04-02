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

PAPER_SYMBOL = SYMBOL


# ═══════════════════════════════════════════
# HEALTHCHECK СЕРВЕР
# ═══════════════════════════════════════════
health_app = Flask(__name__)

@health_app.route("/health")
def health():
    return {"status": "ok", "bot": "TradeBot v3.0"}, 200

@health_app.route("/")
def index():
    try:
        stats = get_statistics()
        paper = get_stats()
        return {
            "bot":           "TradeBot v3.0 — Paper Trading",
            "symbol":        SYMBOL,
            "paper_balance": paper["balance"],
            "paper_winrate": paper["winrate"],
            "stats":         stats
        }, 200
    except:
        return {"status": "initializing"}, 200

def run_health_server():
    port = int(os.environ.get("PORT", 8080))
    health_app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


# ═══════════════════════════════════════════
# ОСНОВНОЙ ТОРГОВЫЙ ЦИКЛ
# ═══════════════════════════════════════════
def trading_loop():
    logger.info(f"🚀 Торговый цикл запущен | {SYMBOL}")

    while True:
        try:
            # ── Шаг 1: Мониторинг Paper сделок (сначала проверяем старые) ──
            closed = monitor_trades(PAPER_SYMBOL)
            for trade in closed:
                emoji = "✅" if trade["result"] == "WIN" else "❌"
                send_message(
                    f"{emoji} <b>Сделка закрыта — {trade['result']}</b>\n\n"
                    f"📊 {trade['signal']} {trade['symbol']}\n"
                    f"🔵 Вход:  <b>${trade['price_open']:.4f}</b>\n"
                    f"🔴 Выход: <b>${trade['price_close']:.4f}</b>\n"
                    f"💰 P&L:   <b>{trade['pnl_pct']:+.2f}% (${trade['pnl_usd']:+.2f})</b>"
                )

            # ── Шаг 2: Получение сигнала ──
            signal_data = get_live_signal()
            if not signal_data:
                time.sleep(60)
                continue

            signal     = signal_data.get("signal", "HOLD")
            confidence = signal_data.get("confidence", 0.0)
            price      = signal_data.get("price", 0.0)

            logger.info(f"📊 Анализ: {signal} ({confidence:.1%})")

            # ── Шаг 3: Логика открытия ──
            if signal in ("BUY", "SELL") and confidence >= MIN_CONFIDENCE:
                
                # Доп. проверка через AI Sentiment
                try:
                    sent = get_market_sentiment(price, 0, 0)
                    boost = sentiment_to_signal_boost(sent, signal)
                    confidence = min(confidence * boost, 0.99)
                except:
                    boost = 1.0

                if confidence >= MIN_CONFIDENCE:
                    trade = open_trade(signal, price, confidence, PAPER_SYMBOL)
                    if trade:
                        emoji = "🟢" if signal == "BUY" else "🔴"
                        send_message(
                            f"{emoji} <b>Новая сделка: {signal}</b>\n\n"
                            f"💵 Цена: <b>${price:.4f}</b>\n"
                            f"🎯 Уверенность: <b>{confidence:.1%}</b>\n"
                            f"🛑 SL: <b>${trade['sl']:.4f}</b> | ✅ TP: <b>${trade['tp']:.4f}</b>"
                        )

            time.sleep(SIGNAL_INTERVAL_MINUTES * 60)

        except Exception as e:
            logger.error(f"Ошибка цикла: {e}")
            time.sleep(60)

# ... остальной код (retrainer_loop, backtest_loop) остается таким же ...

def retrainer_loop():
    time.sleep(30)
    while True:
        try:
            result = train_model()
            if result.get("success"):
                send_message(f"🧠 <b>Модель переобучена!</b>\n🎯 Precision: <b>{result['precision']:.1%}</b>\n📚 База: <b>{result['n_samples']}</b>")
        except: pass
        time.sleep(6 * 60 * 60)

if __name__ == "__main__":
    if not validate_config():
        # Запускаем потоки
        threading.Thread(target=run_health_server, daemon=True).start()
        threading.Thread(target=retrainer_loop,    daemon=True).start()
        threading.Thread(target=trading_loop,      daemon=True).start()
        
        send_message("🤖 <b>TradeBot v3.0 запущен и готов к работе!</b>")
        
        while True:
            time.sleep(60)