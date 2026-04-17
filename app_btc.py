"""
TradeBot v8.3 — BTC
Фиксы vs v8.0:
  - retrainer_loop читает RETRAIN_INTERVAL_HRS из config (был жёсткий 24ч)
  - signal_logger: paper trading лог каждого сигнала + авто-проверка через 6ч
"""

import threading
from claude_advisor import run_advisor, add_log
import time
import logging
import os
import json
from flask import Flask

from config import (
    SYMBOL, MIN_CONFIDENCE, STRONG_SIGNAL,
    SIGNAL_INTERVAL_MINUTES, validate_config,
    MODEL_FEATURES_PATH
)
from live_signal        import get_live_signal

import importlib
def get_config(param, default):
    try:
        import config as _cfg
        importlib.reload(_cfg)
        return getattr(_cfg, param, default)
    except Exception:
        return default

from sentiment_analyzer import get_market_sentiment, sentiment_to_signal_boost
from telegram_notify    import send_message
from trade_archive      import get_statistics
from auto_trainer       import train_model
from paper_trader       import (
    open_trade, monitor_trades,
    get_stats, format_stats_message
)
from backtest_engine    import run_backtest, format_backtest_message

try:
    from signal_logger import (
        log_signal, check_outcomes,
        format_outcome_message, get_stats_message, daily_report_message
    )
    SIGNAL_LOGGER_OK = True
except ImportError:
    SIGNAL_LOGGER_OK = False

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

PAPER_SYMBOL = SYMBOL

_last_trade_time       = 0
TRADE_COOLDOWN_SECONDS = 2 * 60 * 60


def _get_feature_count() -> int:
    try:
        if os.path.exists(MODEL_FEATURES_PATH):
            with open(MODEL_FEATURES_PATH) as f:
                return len(json.load(f))
    except Exception:
        pass
    return 55


# ═══════════════════════════════════════════
# HEALTHCHECK
# ═══════════════════════════════════════════
health_app = Flask(__name__)

@health_app.route("/health")
def health():
    return {"status": "ok", "bot": "TradeBot v8.3"}, 200

@health_app.route("/")
def index():
    try:
        stats = get_statistics()
        paper = get_stats()
        dd = paper.get("current_drawdown", 0)
        cl = paper.get("consecutive_losses", 0)
        return {
            "bot":                "TradeBot v8.3 BTC — signal_logger + retrain 6h",
            "symbol":             SYMBOL,
            "paper_balance":      paper["balance"],
            "paper_winrate":      paper["winrate"],
            "paper_drawdown_pct": dd,
            "consecutive_losses": cl,
            "stats":              stats
        }, 200
    except Exception:
        return {"status": "initializing"}, 200


def run_health_server():
    port = int(os.environ.get("PORT", 8080))
    health_app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


# ═══════════════════════════════════════════
# ПРОВЕРКА ИСХОДОВ СИГНАЛОВ (signal_logger)
# ═══════════════════════════════════════════
def _get_btc_price(symbol: str) -> float:
    try:
        sig = get_live_signal()
        return sig.get("price", 0.0) if sig else 0.0
    except Exception:
        return 0.0


def outcome_checker_loop():
    """Каждые 30 минут проверяет результаты сигналов из signal_logger."""
    if not SIGNAL_LOGGER_OK:
        return
    time.sleep(300)
    while True:
        try:
            results = check_outcomes(_get_btc_price)
            for r in results:
                send_message(format_outcome_message(r))
                logger.info(f"[SignalLogger] {r['signal']} {r['outcome']} P&L={r['pnl_pct']:+.2f}%")
        except Exception as e:
            logger.error(f"[OutcomeChecker] {e}")
        time.sleep(30 * 60)


# ═══════════════════════════════════════════
# ТОРГОВЫЙ ЦИКЛ
# ═══════════════════════════════════════════
def trading_loop():
    global _last_trade_time
    logger.info(f"🚀 Торговый цикл v8.3 запущен | {SYMBOL}")

    while True:
        try:
            closed = monitor_trades(PAPER_SYMBOL)
            for trade in closed:
                _last_trade_time = time.time()
                emoji        = "✅" if trade["result"] == "WIN" else "❌"
                trailing_note = ""
                if trade.get("trailing_active"):
                    trailing_note = "\n🔄 <b>Trailing Stop сработал!</b>"
                if trade.get("breakeven_hit"):
                    trailing_note += "\n🎯 <b>Breakeven был активен</b>"
                cl_note = ""
                cl = trade.get("consecutive_loss_at_open", 0)
                if cl >= 2:
                    cl_note = f"\n⚠️ <b>Kelly снижен (серия убытков: {cl})</b>"
                send_message(
                    f"{emoji} <b>Сделка закрыта — {trade['result']}</b>\n\n"
                    f"📊 {trade['signal']} {trade['symbol']}\n"
                    f"🔵 Вход:  <b>${trade['price_open']:.4f}</b>\n"
                    f"🔴 Выход: <b>${trade['price_close']:.4f}</b>\n"
                    f"💰 P&L:   <b>{trade['pnl_pct']:+.2f}% (${trade['pnl_usd']:+.2f})</b>\n"
                    f"🔒 Причина: <b>{trade.get('closed_by','—')}</b>"
                    f"{trailing_note}{cl_note}"
                )
                try:
                    from claude_advisor import add_trade_result
                    add_trade_result(
                        signal=trade.get('signal','?'),
                        regime=trade.get('regime','?'),
                        adx=float(trade.get('adx',0)),
                        hurst=float(trade.get('hurst',0.5)),
                        result=trade['result'],
                        pnl=float(trade.get('pnl_pct',0))
                    )
                except Exception:
                    pass

            since_last = time.time() - _last_trade_time
            if since_last < TRADE_COOLDOWN_SECONDS and _last_trade_time > 0:
                remaining = int((TRADE_COOLDOWN_SECONDS - since_last) / 60)
                logger.info(f"⏳ Cooldown: ещё {remaining} мин.")
                time.sleep(SIGNAL_INTERVAL_MINUTES * 60)
                continue

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
            p_buy      = signal_data.get("p_buy",      0.0)
            p_sell     = signal_data.get("p_sell",     0.0)
            models     = signal_data.get("models_used","XGB")
            mtf_ok     = signal_data.get("mtf_confirmed", True)
            btc_ch     = signal_data.get("btc_change_4h", 0.0)
            rsi        = signal_data.get("rsi14",      50.0)
            p_meta     = signal_data.get("p_meta",     None)
            hurst      = signal_data.get("hurst",      0.5)
            regime     = signal_data.get("regime",     "unknown")

            logger.info(
                f"📊 {signal} | p_buy={p_buy:.1%} p_sell={p_sell:.1%} | "
                f"ADX={adx:.1f} | Hurst={hurst:.3f} | Regime={regime} | "
                f"4H={'✅' if mtf_ok else '❌'} | BTC={btc_ch:+.2f}%"
                + (f" | meta={p_meta:.1%}" if p_meta is not None else "")
            )
            add_log(f"[Signal v8.3] {signal} | p_buy={p_buy:.1%} p_sell={p_sell:.1%} | ADX={adx:.1f} | Hurst={hurst:.3f}")

            # ── Логируем сигнал в paper trading БД ──
            if signal in ("BUY", "SELL") and SIGNAL_LOGGER_OK:
                log_signal(signal, price, SYMBOL, confidence, {
                    "p_buy": p_buy, "p_sell": p_sell, "models_used": models
                })

            if signal in ("BUY", "SELL") and confidence >= get_config("MIN_CONFIDENCE", 0.52):
                sent = {}
                try:
                    sent  = get_market_sentiment(price, change_24h, volume, rsi=rsi, symbol="BTC")
                    boost = sentiment_to_signal_boost(sent, signal)
                    old_conf = confidence
                    confidence = min(confidence * boost, 0.99)
                    logger.info(f"🧠 Sentiment: {sent.get('sentiment')} boost={boost:.2f} conf: {old_conf:.1%}→{confidence:.1%}")
                except Exception:
                    pass

                if confidence >= get_config("MIN_CONFIDENCE", 0.52):
                    strength_label = "🔥 STRONG" if confidence >= get_config("STRONG_SIGNAL", 0.65) else "📶 NORMAL"
                    extra_info = {
                        "p_buy": p_buy, "p_sell": p_sell, "models_used": models,
                        "mtf_confirmed": mtf_ok, "btc_change_4h": btc_ch,
                        "adx": adx, "hurst": hurst, "regime": regime,
                    }
                    trade = open_trade(signal, price, confidence, PAPER_SYMBOL, atr=atr, extra_info=extra_info)

                    if trade:
                        _last_trade_time = time.time()
                        emoji = "🟢" if signal == "BUY" else "🔴"
                        meta_line = f"\n🧩 Meta-filter:   <b>{p_meta:.1%}</b>" if p_meta is not None else ""
                        sent_line = (f"\n🌐 Sentiment:     <b>{sent.get('sentiment','?')} ({sent.get('source','?')})</b>") if sent else ""
                        hurst_label = "Тренд" if hurst > 0.6 else ("Mean-Rev" if hurst < 0.4 else "Случайный")
                        paper_stats = get_stats()
                        cl    = paper_stats.get("consecutive_losses", 0)
                        dd    = paper_stats.get("current_drawdown", 0)
                        cl_note = f"\n⚠️ Серия убытков: <b>{cl}</b>" if cl >= 2 else ""
                        dd_note = f"\n📉 DrawDown: <b>{dd:.1f}%</b>" if dd >= 5 else ""

                        send_message(
                            f"{emoji} <b>Новая сделка: {signal} {strength_label}</b>\n\n"
                            f"💵 Цена:          <b>${price:.4f}</b>\n"
                            f"🎯 p(BUY):        <b>{p_buy:.1%}</b>\n"
                            f"🎯 p(SELL):       <b>{p_sell:.1%}</b>"
                            f"{meta_line}"
                            f"\n📈 24h change:    <b>{change_24h:+.2f}%</b>\n"
                            f"📐 ATR:           <b>{atr:.4f}</b>\n"
                            f"💹 ADX:           <b>{adx:.1f}</b>\n"
                            f"🌊 Hurst:         <b>{hurst:.3f} ({hurst_label})</b>\n"
                            f"📊 Режим:         <b>{regime}</b>\n"
                            f"🤖 Модели:        <b>{models}</b>\n"
                            f"📊 BTC 4H:        <b>{btc_ch:+.2f}%</b>"
                            f"{sent_line}"
                            f"\n💼 Размер:        <b>${trade['amount_usd']:.2f} ({trade['kelly_pct']:.1%})</b>\n"
                            f"🛑 SL: <b>${trade['sl']:.4f}</b> | ✅ TP: <b>${trade['tp']:.4f}</b>\n"
                            f"🔄 Trailing SL:   <b>{'Активен' if trade.get('trailing_active') else 'Ожидание +1%'}</b>"
                            f"{cl_note}{dd_note}"
                        )
                    else:
                        paper_stats = get_stats()
                        dd = paper_stats.get("current_drawdown", 0)
                        if dd >= 20:
                            logger.warning(f"[Trading] 🚫 DrawDown Guard: {dd:.1f}%")
                        else:
                            logger.info("ℹ️ Сделка не открыта (уже есть открытая)")
            else:
                logger.info(f"⏸ {signal} | p_buy={p_buy:.1%} p_sell={p_sell:.1%}")

            time.sleep(SIGNAL_INTERVAL_MINUTES * 60)

        except Exception as e:
            logger.error(f"❌ Ошибка торгового цикла: {e}", exc_info=True)
            time.sleep(60)


# ═══════════════════════════════════════════
# ПЕРЕОБУЧЕНИЕ — ЧИТАЕТ RETRAIN_INTERVAL_HRS
# ═══════════════════════════════════════════
def retrainer_loop():
    time.sleep(60)
    interval_hrs = get_config("RETRAIN_INTERVAL_HRS", 6)
    logger.info(f"🧠 Retrainer v8.3 запущен | интервал: {interval_hrs}ч")

    while True:
        try:
            result = train_model()
            if result.get("success"):
                buy_prec       = result.get("avg_buy_precision", 0)
                sell_prec      = result.get("avg_sell_precision", 0)
                buy_auc        = result.get("avg_buy_auc", 0)
                sell_auc       = result.get("avg_sell_auc", 0)
                wf_buy         = result.get("wf_buy_precision", 0)
                wf_sell        = result.get("wf_sell_precision", 0)
                wf_sharpe_buy  = result.get("wf_buy_sharpe", 0)
                wf_sharpe_sell = result.get("wf_sell_sharpe", 0)
                meta_buy_p     = result.get("meta_buy_precision")
                meta_sell_p    = result.get("meta_sell_precision")
                stack_buy_p    = result.get("stack_buy_precision")
                kelly_f        = result.get("kelly_fraction", 0)
                n_features     = result.get("n_features", 0)
                labeling       = result.get("labeling", "simple")

                meta_line = ""
                if meta_buy_p is not None:
                    meta_line = (
                        f"\n\n🧩 Meta-filter:\n   BUY:  <b>{meta_buy_p:.1%}</b>\n   SELL: <b>{meta_sell_p:.1%}</b>"
                        if meta_sell_p else
                        f"\n\n🧩 Meta BUY: <b>{meta_buy_p:.1%}</b>"
                    )
                stack_line = f"\n🏗️ Stack BUY prec: <b>{stack_buy_p:.1%}</b>" if stack_buy_p else ""

                send_message(
                    f"🧠 <b>Ансамбль v8.3 переобучен!</b>\n\n"
                    f"🏷 Разметка: <b>{'Triple Barrier ✅' if labeling == 'triple_barrier' else 'Simple'}</b>\n\n"
                    f"🟢 BUY:\n   Precision: <b>{buy_prec:.1%}</b> | AUC: <b>{buy_auc:.3f}</b>\n"
                    f"   WF prec: <b>{wf_buy:.1%}</b> | WF Sharpe: <b>{wf_sharpe_buy:.2f}</b>\n\n"
                    f"🔴 SELL:\n   Precision: <b>{sell_prec:.1%}</b> | AUC: <b>{sell_auc:.3f}</b>\n"
                    f"   WF prec: <b>{wf_sell:.1%}</b> | WF Sharpe: <b>{wf_sharpe_sell:.2f}</b>"
                    f"{stack_line}{meta_line}\n\n"
                    f"📐 Kelly: <b>{kelly_f:.1%}</b> | Признаков: <b>{n_features}</b>\n"
                    f"⏱ Следующее через: <b>{get_config('RETRAIN_INTERVAL_HRS', 6)}ч</b>"
                )
                add_log(f"BUY prec={buy_prec:.1%} | sharpe={wf_sharpe_buy:.2f} | Kelly={kelly_f:.1%}")
                try:
                    from auto_trainer import _git_push_if_better
                    _git_push_if_better(result)
                except Exception:
                    pass
            else:
                add_log(f"[Retrainer] Неудача: {result.get('error')}")
                logger.warning(f"[Retrainer] Неудача: {result.get('error')}")
        except Exception as e:
            logger.error(f"[Retrainer] Ошибка: {e}", exc_info=True)
            time.sleep(2 * 60 * 60)
            continue

        interval_hrs = get_config("RETRAIN_INTERVAL_HRS", 6)
        logger.info(f"[Retrainer] Следующее переобучение через {interval_hrs}ч")
        time.sleep(interval_hrs * 3600)


# ═══════════════════════════════════════════
# БЭКТЕСТ (12ч)
# ═══════════════════════════════════════════
def backtest_loop():
    time.sleep(120)
    while True:
        try:
            result = run_backtest(symbol=SYMBOL)
            send_message(format_backtest_message(result))
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
            if SIGNAL_LOGGER_OK:
                send_message(daily_report_message())
        except Exception as e:
            logger.error(f"[Stats] Ошибка: {e}")
        time.sleep(24 * 60 * 60)


def advisor_loop():
    time.sleep(600)
    while True:
        try:
            run_advisor(symbol=SYMBOL)
        except Exception as e:
            logger.error(f"[Advisor] Ошибка: {e}")
        time.sleep(6 * 60 * 60)


# ═══════════════════════════════════════════
# ТОЧКА ВХОДА
# ═══════════════════════════════════════════
if __name__ == "__main__":
    errors = validate_config()
    if errors:
        logger.critical(f"❌ Не заданы переменные окружения: {errors}")
        exit(1)

    n_features = _get_feature_count()
    logger.info(f"✅ Конфиг OK | Признаков: {n_features} | Запускаем TradeBot v8.3 BTC...")

    threading.Thread(target=run_health_server,    daemon=True).start()
    threading.Thread(target=retrainer_loop,       daemon=True).start()
    threading.Thread(target=trading_loop,         daemon=True).start()
    threading.Thread(target=backtest_loop,        daemon=True).start()
    threading.Thread(target=stats_loop,           daemon=True).start()
    threading.Thread(target=advisor_loop,         daemon=True).start()
    if SIGNAL_LOGGER_OK:
        threading.Thread(target=outcome_checker_loop, daemon=True).start()

    lunarcrush_status = "✅ Активен" if os.getenv("LUNARCRUSH_API_KEY") else "⚠️ Нет ключа"

    send_message(
        f"🤖 <b>TradeBot v8.3 BTC запущен!</b>\n\n"
        f"📊 Пара: <b>{SYMBOL}</b>\n"
        f"⏱ Интервал: <b>{SIGNAL_INTERVAL_MINUTES} мин</b>\n"
        f"🎯 Мин. уверенность: <b>{MIN_CONFIDENCE:.0%}</b>\n"
        f"🔁 Переобучение: <b>каждые {get_config('RETRAIN_INTERVAL_HRS',6)}ч</b>\n"
        f"📝 Signal Logger: <b>{'✅ Активен' if SIGNAL_LOGGER_OK else '❌ Не загружен'}</b>\n"
        f"🌐 Sentiment: <b>{lunarcrush_status}</b>\n"
        f"🔢 Признаков: <b>{n_features}</b>"
    )

    while True:
        time.sleep(60)
