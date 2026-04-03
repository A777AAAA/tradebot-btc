"""
TradeBot v4.0 — Multi-TF Ensemble Edition
v4.0 изменения:
  - Ансамбль XGBoost + LightGBM (консенсус)
  - Multi-timeframe: 1H сигнал + 4H фильтр
  - BTC macro-фильтр
  - Market Regime (ADX) фильтр
  - Trailing Stop-Loss + Breakeven
  - 40+ признаков
  - Walk-Forward переобучение
  - Расширенный Telegram-отчёт с информацией о фильтрах
"""

import threading
import time
import logging
import os
from flask import Flask

from config import (
    SYMBOL, MIN_CONFIDENCE, STRONG_SIGNAL,
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

_last_trade_time       = 0
TRADE_COOLDOWN_SECONDS = 2 * 60 * 60   # 2 часа


# ═══════════════════════════════════════════
# HEALTHCHECK
# ═══════════════════════════════════════════
health_app = Flask(__name__)

@health_app.route("/health")
def health():
    return {"status": "ok", "bot": "TradeBot v4.0"}, 200

@health_app.route("/")
def index():
    try:
        stats = get_statistics()
        paper = get_stats()
        return {
            "bot":           "TradeBot v4.0 — Multi-TF Ensemble",
            "symbol":        SYMBOL,
            "paper_balance": paper["balance"],
            "paper_winrate": paper["winrate"],
            "stats":         stats
        }, 200
    except Exception:
        return {"status": "initializing"}, 200

def run_health_server():
    port = int(os.environ.get("PORT", 8080))
    health_app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


# ═══════════════════════════════════════════
# ОСНОВНОЙ ТОРГОВЫЙ ЦИКЛ
# ═══════════════════════════════════════════
def trading_loop():
    global _last_trade_time
    logger.info(f"🚀 Торговый цикл v4.0 запущен | {SYMBOL}")

    while True:
        try:
            # ── Шаг 1: Мониторинг открытых сделок (+ trailing) ──
            closed = monitor_trades(PAPER_SYMBOL)
            for trade in closed:
                _last_trade_time = time.time()
                emoji = "✅" if trade["result"] == "WIN" else "❌"
                trailing_note = ""
                if trade.get("trailing_active"):
                    trailing_note = "\n🔄 <b>Trailing Stop сработал!</b>"
                if trade.get("breakeven_hit"):
                    trailing_note += "\n🎯 <b>Breakeven был активен</b>"

                send_message(
                    f"{emoji} <b>Сделка закрыта — {trade['result']}</b>\n\n"
                    f"📊 {trade['signal']} {trade['symbol']}\n"
                    f"🔵 Вход:  <b>${trade['price_open']:.4f}</b>\n"
                    f"🔴 Выход: <b>${trade['price_close']:.4f}</b>\n"
                    f"💰 P&L:   <b>{trade['pnl_pct']:+.2f}% "
                    f"(${trade['pnl_usd']:+.2f})</b>\n"
                    f"🔒 Причина: <b>{trade.get('closed_by','—')}</b>"
                    f"{trailing_note}"
                )

            # ── Шаг 2: Cooldown ──
            since_last = time.time() - _last_trade_time
            if since_last < TRADE_COOLDOWN_SECONDS and _last_trade_time > 0:
                remaining = int((TRADE_COOLDOWN_SECONDS - since_last) / 60)
                logger.info(f"⏳ Cooldown: ещё {remaining} мин.")
                time.sleep(SIGNAL_INTERVAL_MINUTES * 60)
                continue

            # ── Шаг 3: Сигнал ансамбля ──
            signal_data = get_live_signal()
            if not signal_data:
                logger.warning("⚠️ Сигнал не получен")
                time.sleep(60)
                continue

            signal     = signal_data.get("signal",     "HOLD")
            confidence = signal_data.get("confidence", 0.0)
            price      = signal_data.get("price",      0.0)
            atr        = signal_data.get("atr",        0.0)
            change_24h = signal_data.get("change_24h", 0.0)
            volume     = signal_data.get("volume",     0.0)
            adx        = signal_data.get("adx",        0.0)
            models     = signal_data.get("models_used", "XGB")
            mtf_ok     = signal_data.get("mtf_confirmed", True)
            btc_ch     = signal_data.get("btc_change_4h", 0.0)
            xgb_sig    = signal_data.get("xgb_signal", "N/A")
            lgbm_sig   = signal_data.get("lgbm_signal","N/A")

            logger.info(
                f"📊 {signal} ({confidence:.1%}) | "
                f"XGB={xgb_sig} LGBM={lgbm_sig} | "
                f"ADX={adx:.1f} | 4H={'✅' if mtf_ok else '❌'} | "
                f"BTC={btc_ch:+.2f}%"
            )

            # ── Шаг 4: Открытие ──
            if signal in ("BUY", "SELL") and confidence >= MIN_CONFIDENCE:

                # Sentiment boost
                try:
                    sent  = get_market_sentiment(price, change_24h, volume)
                    boost = sentiment_to_signal_boost(sent, signal)
                    confidence = min(confidence * boost, 0.99)
                    logger.info(
                        f"🧠 Sentiment: {sent.get('sentiment')} "
                        f"boost={boost:.2f} → conf={confidence:.1%}"
                    )
                except Exception:
                    pass

                if confidence >= MIN_CONFIDENCE:
                    strength_label = "🔥 STRONG" if confidence >= STRONG_SIGNAL else "📶 NORMAL"

                    extra_info = {
                        "xgb_signal":    xgb_sig,
                        "lgbm_signal":   lgbm_sig,
                        "models_used":   models,
                        "mtf_confirmed": mtf_ok,
                        "btc_change_4h": btc_ch,
                        "adx":           adx,
                    }

                    trade = open_trade(
                        signal, price, confidence,
                        PAPER_SYMBOL, atr=atr,
                        extra_info=extra_info
                    )

                    if trade:
                        _last_trade_time = time.time()
                        emoji = "🟢" if signal == "BUY" else "🔴"
                        trade_pct_str = f"{trade['trade_pct']:.0%}"

                        send_message(
                            f"{emoji} <b>Новая сделка: {signal} {strength_label}</b>\n\n"
                            f"💵 Цена:          <b>${price:.4f}</b>\n"
                            f"🎯 Уверенность:   <b>{confidence:.1%}</b>\n"
                            f"📈 24h change:    <b>{change_24h:+.2f}%</b>\n"
                            f"📐 ATR:           <b>{atr:.4f}</b>\n"
                            f"💹 ADX:           <b>{adx:.1f}</b>\n"
                            f"🤖 Модели:        <b>{models}</b>\n"
                            f"XGB={xgb_sig} | LGBM={lgbm_sig}\n"
                            f"📊 BTC 4H:        <b>{btc_ch:+.2f}%</b>\n"
                            f"💼 Размер:        <b>{trade_pct_str} (${trade['amount_usd']:.2f})</b>\n"
                            f"🛑 SL: <b>${trade['sl']:.4f}</b> | "
                            f"✅ TP: <b>${trade['tp']:.4f}</b>\n"
                            f"🔄 Trailing SL:   <b>{'Активен' if trade.get('trailing_active') else 'Ожидание +1%'}</b>"
                        )
                    else:
                        logger.info("ℹ️ Сделка не открыта (уже есть открытая)")
            else:
                logger.info("⏸ HOLD или низкая уверенность")

            time.sleep(SIGNAL_INTERVAL_MINUTES * 60)

        except Exception as e:
            logger.error(f"❌ Ошибка торгового цикла: {e}", exc_info=True)
            time.sleep(60)


# ═══════════════════════════════════════════
# ЦИКЛ ПЕРЕОБУЧЕНИЯ
# ═══════════════════════════════════════════
def retrainer_loop():
    time.sleep(60)
    logger.info("🧠 Retrainer v4.0 запущен (Walk-Forward, 24ч)")

    while True:
        try:
            result = train_model()
            if result.get("success"):
                lgbm_str = ""
                if result.get("lgbm_precision"):
                    lgbm_str = f"\n🌿 LightGBM:     <b>{result['lgbm_precision']:.1%}</b>"

                send_message(
                    f"🧠 <b>Ансамбль переобучен! v4.0</b>\n\n"
                    f"🎯 XGBoost:      <b>{result['xgb_precision']:.1%}</b>"
                    f"{lgbm_str}\n"
                    f"🏆 Ensemble:     <b>{result['ensemble_precision']:.1%}</b>\n\n"
                    f"📊 Walk-Forward:\n"
                    f"   Precision: <b>{result['wf_precision']:.1%}</b>\n"
                    f"   Accuracy:  <b>{result['wf_accuracy']:.1%}</b>\n"
                    f"   Folds:     <b>{result['wf_folds']}</b>\n\n"
                    f"📚 Выборка:      <b>{result['n_samples']}</b> свечей\n"
                    f"🔢 Признаков:    <b>{result.get('n_features', '?')}</b>"
                )
            else:
                logger.warning(f"[Retrainer] Неудача: {result.get('error')}")
        except Exception as e:
            logger.error(f"[Retrainer] Ошибка: {e}", exc_info=True)

        time.sleep(24 * 60 * 60)


# ═══════════════════════════════════════════
# БЭКТЕСТ (каждые 12 часов)
# ═══════════════════════════════════════════
def backtest_loop():
    time.sleep(120)

    while True:
        try:
            result = run_backtest(symbol=SYMBOL)
            msg    = format_backtest_message(result)
            send_message(msg)
        except Exception as e:
            logger.error(f"[Backtest] Ошибка: {e}", exc_info=True)

        time.sleep(12 * 60 * 60)


# ═══════════════════════════════════════════
# ЕЖЕДНЕВНЫЙ ОТЧЁТ
# ═══════════════════════════════════════════
def stats_loop():
    time.sleep(300)

    while True:
        try:
            stats = get_stats()
            send_message(format_stats_message(stats))
        except Exception as e:
            logger.error(f"[Stats] Ошибка: {e}")

        time.sleep(24 * 60 * 60)


# ═══════════════════════════════════════════
# ТОЧКА ВХОДА
# ═══════════════════════════════════════════
if __name__ == "__main__":
    errors = validate_config()
    if errors:
        logger.critical(f"❌ Не заданы переменные окружения: {errors}")
        exit(1)

    logger.info("✅ Конфиг OK, запускаем TradeBot v4.0...")

    threading.Thread(target=run_health_server, daemon=True).start()
    threading.Thread(target=retrainer_loop,    daemon=True).start()
    threading.Thread(target=trading_loop,      daemon=True).start()
    threading.Thread(target=backtest_loop,     daemon=True).start()
    threading.Thread(target=stats_loop,        daemon=True).start()

    send_message(
        "🤖 <b>TradeBot v4.0 запущен!</b>\n\n"
        f"📊 Пара:              <b>{SYMBOL}</b>\n"
        f"⏱ Интервал:          <b>{SIGNAL_INTERVAL_MINUTES} мин</b>\n"
        f"🎯 Мин. уверенность: <b>{MIN_CONFIDENCE:.0%}</b>\n"
        f"🤖 Ансамбль:         <b>XGBoost + LightGBM</b>\n"
        f"📐 Multi-TF:         <b>1H сигнал + 4H фильтр</b>\n"
        f"₿  BTC macro:        <b>Активен</b>\n"
        f"📊 Market Regime:    <b>ADX фильтр активен</b>\n"
        f"🔄 Trailing SL:      <b>+1.0% breakeven / +1.5% trail</b>\n"
        f"🔢 Признаков:        <b>40+</b>"
    )

    while True:
        time.sleep(60)