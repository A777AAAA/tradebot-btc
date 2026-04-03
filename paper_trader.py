"""
paper_trader.py v5.0 — Kelly Criterion + исправлен TP/SL + Dynamic Position Sizing
v5.0 изменения:
  - Kelly Criterion: читает из training_stats.json и применяет Half-Kelly
    для динамического размера позиции вместо фиксированных 10%/15%
  - Исправлена синхронизация TP/SL: используются ATR_TP_MULT и ATR_SL_MULT из config
    (было несоответствие между paper_trader и backtest_engine)
  - Добавлен Risk-of-Ruin check: если Kelly даёт > 25% — ограничиваем
  - Logging R:R ratio для каждой сделки
  - Trailing Stop-Loss v4 логика сохранена полностью
"""

import json
import os
import logging
import ccxt
from datetime import datetime, timezone

from config import (
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    ATR_SL_MULT, ATR_TP_MULT, SL_FLOOR_PCT, SL_CAP_PCT,
    TRAILING_ENABLED, TRAILING_ACTIVATION_PCT,
    TRAILING_DISTANCE_PCT, BREAKEVEN_ACTIVATION,
    STRONG_SIGNAL,
)

PAPER_FILE   = "paper_trades.json"
BALANCE_FILE = "paper_balance.json"
STATS_FILE   = "training_stats.json"

INITIAL_BALANCE  = 600.0

# Дефолтные размеры позиции (используются если Kelly недоступен)
TRADE_PCT_DEFAULT        = 0.10   # 10% — обычный сигнал
TRADE_PCT_STRONG_DEFAULT = 0.15   # 15% — STRONG_SIGNAL
TRADE_PCT_MAX            = 0.25   # 25% — абсолютный максимум (Risk-of-Ruin защита)
TRADE_PCT_MIN            = 0.05   # 5%  — минимальный размер

OKX_CONFIG = {'options': {'defaultType': 'spot'}, 'timeout': 30000}

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Kelly Criterion Position Sizing
# ─────────────────────────────────────────────
def get_kelly_trade_pct(confidence: float) -> float:
    """
    Читает Kelly Fraction из последнего обучения и возвращает
    рекомендуемый размер позиции.

    Логика:
    - Если confidence высокий (>= STRONG_SIGNAL) → Kelly × 1.25
    - Обычный сигнал → Kelly × 1.0
    - Если Kelly файл недоступен → дефолтные значения

    Half-Kelly уже применён при обучении (в auto_trainer.py).
    Здесь применяем дополнительную корректировку по уверенности.
    """
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE) as f:
                stats = json.load(f)

            kelly_f = float(stats.get("kelly_fraction", 0.0))

            if kelly_f > 0.03:  # Kelly посчитан и разумен
                # Корректировка по уверенности сигнала
                if confidence >= STRONG_SIGNAL:
                    trade_pct = kelly_f * 1.25
                elif confidence >= 0.65:
                    trade_pct = kelly_f * 1.0
                else:
                    trade_pct = kelly_f * 0.75

                # Ограничения
                trade_pct = max(TRADE_PCT_MIN, min(trade_pct, TRADE_PCT_MAX))

                logger.debug(
                    f"[Paper] Kelly={kelly_f:.1%} conf={confidence:.1%} "
                    f"→ size={trade_pct:.1%}"
                )
                return round(trade_pct, 3)

    except Exception as e:
        logger.debug(f"[Paper] Kelly чтение ошибка: {e}")

    # Fallback к фиксированным значениям
    if confidence >= STRONG_SIGNAL:
        return TRADE_PCT_STRONG_DEFAULT
    return TRADE_PCT_DEFAULT


# ─────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_balance() -> dict:
    if not os.path.exists(BALANCE_FILE):
        data = {
            "balance":    INITIAL_BALANCE,
            "total_pnl":  0.0,
            "trades":     0,
            "wins":       0,
            "losses":     0,
            "created_at": _now()
        }
        save_balance(data)
        return data
    with open(BALANCE_FILE) as f:
        return json.load(f)


def save_balance(data: dict):
    with open(BALANCE_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_trades() -> list:
    if not os.path.exists(PAPER_FILE):
        return []
    with open(PAPER_FILE) as f:
        return json.load(f)


def save_trades(trades: list):
    with open(PAPER_FILE, 'w') as f:
        json.dump(trades, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────
# Текущая цена
# ─────────────────────────────────────────────
def get_current_price(symbol: str = "TON/USDT") -> float:
    try:
        exchange = ccxt.okx(OKX_CONFIG)
        ticker   = exchange.fetch_ticker(symbol)
        return float(ticker['last'])
    except Exception as e:
        logger.error(f"[Paper] ❌ Ошибка цены: {e}")
        return 0.0


# ─────────────────────────────────────────────
# Расчёт SL/TP — ИСПРАВЛЕНО v5.0
# ─────────────────────────────────────────────
def _calc_sl_tp(signal: str, price: float, atr: float = 0.0) -> tuple:
    """
    ИСПРАВЛЕНИЕ v5.0: теперь использует ATR_TP_MULT и ATR_SL_MULT из config
    для расчёта как SL так и TP.

    В v4.0 была ошибка: tp_pct = sl_pct * (ATR_TP_MULT / ATR_SL_MULT)
    Это давало то же соотношение что и ATR_TP_MULT/ATR_SL_MULT = 3.0/1.5 = 2:1,
    но через двойное вычисление теряло точность когда sl_pct обрезался FLOOR/CAP.

    Теперь SL и TP считаются НЕЗАВИСИМО от ATR × множитель.
    """
    use_atr = atr > 0

    if use_atr:
        # SL
        raw_sl_pct = (atr * ATR_SL_MULT) / price
        sl_pct     = max(SL_FLOOR_PCT, min(raw_sl_pct, SL_CAP_PCT))

        # TP — независимо, не производная от sl_pct
        raw_tp_pct = (atr * ATR_TP_MULT) / price
        tp_pct     = max(SL_FLOOR_PCT * 2, raw_tp_pct)  # TP всегда минимум 2×FLOOR

        rr_ratio = round(raw_tp_pct / (raw_sl_pct + 1e-9), 2)
        mode     = f"ATR×SL={ATR_SL_MULT}/TP={ATR_TP_MULT} R:R≈{rr_ratio}"
    else:
        sl_pct = STOP_LOSS_PCT
        tp_pct = TAKE_PROFIT_PCT
        mode   = "FIXED"

    if signal == "BUY":
        sl = round(price * (1 - sl_pct), 6)
        tp = round(price * (1 + tp_pct), 6)
    else:
        sl = round(price * (1 + sl_pct), 6)
        tp = round(price * (1 - tp_pct), 6)

    return sl, tp, mode


# ─────────────────────────────────────────────
# Открыть виртуальную сделку
# ─────────────────────────────────────────────
def open_trade(
    signal:     str,
    price:      float,
    confidence: float,
    symbol:     str   = "TON/USDT",
    atr:        float = 0.0,
    extra_info: dict  = None,
) -> dict | None:

    if signal not in ("BUY", "SELL"):
        return None

    trades = load_trades()
    if any(t["status"] == "OPEN" for t in trades):
        logger.info("[Paper] ⚠️ Уже есть открытая сделка")
        return None

    balance = load_balance()

    # Kelly Criterion Position Sizing (новое v5.0)
    trade_pct  = get_kelly_trade_pct(confidence)
    amount_usd = round(balance["balance"] * trade_pct, 2)
    qty        = round(amount_usd / price, 4)

    sl, tp, sl_mode = _calc_sl_tp(signal, price, atr)

    # R:R ratio для логирования
    rr = round(abs(tp - price) / (abs(price - sl) + 1e-9), 2)

    logger.info(
        f"[Paper] SL/TP: {sl_mode} | SL=${sl:.4f} TP=${tp:.4f} | "
        f"Kelly размер: {trade_pct:.1%} (${amount_usd:.2f}) | "
        f"R:R = 1:{rr}"
    )

    trade = {
        "id":              len(trades) + 1,
        "symbol":          symbol,
        "signal":          signal,
        "status":          "OPEN",
        "price_open":      price,
        "qty":             qty,
        "amount_usd":      amount_usd,
        "trade_pct":       trade_pct,
        "tp":              tp,
        "sl":              sl,
        "sl_initial":      sl,
        "atr":             round(atr, 6),
        "sl_mode":         sl_mode,
        "rr_ratio":        rr,
        "confidence":      round(confidence * 100, 1),
        "opened_at":       _now(),
        "closed_at":       None,
        "price_close":     None,
        "pnl_usd":         None,
        "pnl_pct":         None,
        "result":          None,
        "closed_by":       None,
        # Trailing state
        "breakeven_hit":   False,
        "trailing_active": False,
        "max_price":       price,
        "min_price":       price,
        # Meta
        "extra_info":      extra_info or {},
        "sl_updates":      [],
    }

    trades.append(trade)
    save_trades(trades)

    logger.info(
        f"[Paper] 📝 Открыта #{trade['id']}: "
        f"{signal} {symbol} @ {price} | TP={tp} SL={sl} | R:R=1:{rr}"
    )
    return trade


# ─────────────────────────────────────────────
# Обновление Trailing Stop
# ─────────────────────────────────────────────
def _update_trailing(trade: dict, price: float) -> dict:
    if not TRAILING_ENABLED:
        return trade

    signal     = trade["signal"]
    price_open = trade["price_open"]
    sl         = trade["sl"]

    if signal == "BUY":
        pnl_pct = (price - price_open) / price_open

        if price > trade["max_price"]:
            trade["max_price"] = price

        # Breakeven
        if (not trade["breakeven_hit"]
                and pnl_pct >= BREAKEVEN_ACTIVATION
                and sl < price_open):
            new_sl = round(price_open * 1.0001, 6)
            if new_sl > sl:
                trade["sl_updates"].append({
                    "time": _now(), "old_sl": sl,
                    "new_sl": new_sl, "reason": "BREAKEVEN"
                })
                trade["sl"]           = new_sl
                trade["breakeven_hit"] = True
                logger.info(f"[Paper] 🔄 #{trade['id']} BREAKEVEN SL: ${sl:.4f} → ${new_sl:.4f}")

        # Trailing
        if pnl_pct >= TRAILING_ACTIVATION_PCT:
            trade["trailing_active"] = True
            trailing_sl = round(price * (1 - TRAILING_DISTANCE_PCT), 6)
            if trailing_sl > trade["sl"]:
                old_sl = trade["sl"]
                trade["sl_updates"].append({
                    "time": _now(), "old_sl": old_sl,
                    "new_sl": trailing_sl, "reason": "TRAILING"
                })
                trade["sl"] = trailing_sl
                logger.info(
                    f"[Paper] 📈 #{trade['id']} TRAILING SL: "
                    f"${old_sl:.4f} → ${trailing_sl:.4f} (цена=${price:.4f})"
                )

    elif signal == "SELL":
        pnl_pct = (price_open - price) / price_open

        if price < trade["min_price"]:
            trade["min_price"] = price

        if (not trade["breakeven_hit"]
                and pnl_pct >= BREAKEVEN_ACTIVATION
                and sl > price_open):
            new_sl = round(price_open * 0.9999, 6)
            if new_sl < sl:
                trade["sl_updates"].append({
                    "time": _now(), "old_sl": sl,
                    "new_sl": new_sl, "reason": "BREAKEVEN"
                })
                trade["sl"]           = new_sl
                trade["breakeven_hit"] = True

        if pnl_pct >= TRAILING_ACTIVATION_PCT:
            trade["trailing_active"] = True
            trailing_sl = round(price * (1 + TRAILING_DISTANCE_PCT), 6)
            if trailing_sl < trade["sl"]:
                old_sl = trade["sl"]
                trade["sl_updates"].append({
                    "time": _now(), "old_sl": old_sl,
                    "new_sl": trailing_sl, "reason": "TRAILING"
                })
                trade["sl"] = trailing_sl

    return trade


# ─────────────────────────────────────────────
# Мониторинг открытых сделок
# ─────────────────────────────────────────────
def monitor_trades(symbol: str = "TON/USDT") -> list:
    """Проверяет TP/SL + обновляет Trailing. Возвращает закрытые сделки."""
    trades = load_trades()
    if not any(t["status"] == "OPEN" for t in trades):
        return []

    price = get_current_price(symbol)
    if price == 0.0:
        return []

    balance    = load_balance()
    closed_now = []

    for i, trade in enumerate(trades):
        if trade["status"] != "OPEN":
            continue

        trade = _update_trailing(trade, price)

        signal = trade["signal"]
        tp     = trade["tp"]
        sl     = trade["sl"]
        hit    = None

        if signal == "BUY":
            if price >= tp:
                hit = "TP"
            elif price <= sl:
                hit = "SL_TRAILING" if trade["trailing_active"] else "SL"
        else:
            if price <= tp:
                hit = "TP"
            elif price >= sl:
                hit = "SL_TRAILING" if trade["trailing_active"] else "SL"

        if not hit:
            if signal == "BUY":
                float_pnl = (price - trade["price_open"]) / trade["price_open"] * 100
            else:
                float_pnl = (trade["price_open"] - price) / trade["price_open"] * 100

            trailing_str = "🔄 TRAIL" if trade.get("trailing_active") else ""
            be_str       = "🎯 BE"   if trade.get("breakeven_hit")   else ""
            logger.info(
                f"[Paper] ⏳ #{trade['id']} {signal} | "
                f"P&L: {float_pnl:+.2f}% | "
                f"Price={price:.4f} TP={tp:.4f} SL={sl:.4f} "
                f"{trailing_str}{be_str}"
            )
            trades[i] = trade
            continue

        close_price = tp if hit == "TP" else sl

        if signal == "BUY":
            pnl_pct = (close_price - trade["price_open"]) / trade["price_open"] * 100
        else:
            pnl_pct = (trade["price_open"] - close_price) / trade["price_open"] * 100

        pnl_usd = round(trade["amount_usd"] * pnl_pct / 100, 2)
        result  = "WIN" if hit == "TP" else "LOSS"

        if hit == "SL_TRAILING" and pnl_pct >= 0:
            result = "WIN"

        trades[i].update({
            **trade,
            "status":      "CLOSED",
            "closed_at":   _now(),
            "price_close": close_price,
            "pnl_usd":     pnl_usd,
            "pnl_pct":     round(pnl_pct, 2),
            "result":      result,
            "closed_by":   hit,
        })

        balance["balance"]   = round(balance["balance"] + pnl_usd, 2)
        balance["total_pnl"] = round(balance["total_pnl"] + pnl_usd, 2)
        balance["trades"]   += 1
        balance["wins"]     += (1 if result == "WIN" else 0)
        balance["losses"]   += (0 if result == "WIN" else 1)

        closed_now.append(trades[i])
        logger.info(
            f"[Paper] {'✅' if result == 'WIN' else '❌'} "
            f"#{trade['id']} закрыта по {hit}: "
            f"{pnl_pct:+.2f}% | ${pnl_usd:+.2f} | "
            f"Баланс: ${balance['balance']}"
        )

    save_trades(trades)
    save_balance(balance)
    return closed_now


# ─────────────────────────────────────────────
# Статистика
# ─────────────────────────────────────────────
def get_stats() -> dict:
    balance = load_balance()
    trades  = load_trades()
    closed  = [t for t in trades if t["status"] == "CLOSED"]

    total   = balance["trades"]
    wins    = balance["wins"]
    winrate = round(wins / total * 100, 1) if total > 0 else 0

    pnl_list    = [t["pnl_pct"] for t in closed if t.get("pnl_pct") is not None]
    avg_pnl     = round(sum(pnl_list) / len(pnl_list), 2) if pnl_list else 0
    best_trade  = round(max(pnl_list), 2) if pnl_list else 0
    worst_trade = round(min(pnl_list), 2) if pnl_list else 0
    growth_pct  = round(
        (balance["balance"] - INITIAL_BALANCE) / INITIAL_BALANCE * 100, 2
    )

    trailing_wins  = sum(1 for t in closed if t.get("closed_by") == "SL_TRAILING" and t.get("result") == "WIN")
    breakeven_hits = sum(1 for t in closed if t.get("breakeven_hit"))

    # Текущий Kelly fraction
    kelly_f = 0.0
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE) as f:
                s = json.load(f)
            kelly_f = float(s.get("kelly_fraction", 0.0))
    except Exception:
        pass

    return {
        "balance":        balance["balance"],
        "start_balance":  INITIAL_BALANCE,
        "growth_pct":     growth_pct,
        "total_pnl":      balance["total_pnl"],
        "total_trades":   total,
        "wins":           wins,
        "losses":         balance["losses"],
        "winrate":        winrate,
        "avg_pnl":        avg_pnl,
        "best_trade":     best_trade,
        "worst_trade":    worst_trade,
        "open_trades":    len([t for t in trades if t["status"] == "OPEN"]),
        "trailing_wins":  trailing_wins,
        "breakeven_hits": breakeven_hits,
        "kelly_fraction": kelly_f,
    }


def format_stats_message(stats: dict) -> str:
    emoji  = "📈" if stats["growth_pct"] >= 0 else "📉"
    kelly  = stats.get("kelly_fraction", 0)
    return (
        f"📊 <b>Paper Trading v5.0 — Статистика</b>\n\n"
        f"💰 Баланс:       <b>${stats['balance']:.2f}</b> {emoji}\n"
        f"📈 Рост:         <b>{stats['growth_pct']:+.2f}%</b>\n"
        f"💵 P&L всего:    <b>${stats['total_pnl']:+.2f}</b>\n\n"
        f"📋 Сделок:       <b>{stats['total_trades']}</b>\n"
        f"✅ Побед:        <b>{stats['wins']}</b>\n"
        f"❌ Поражений:    <b>{stats['losses']}</b>\n"
        f"🎯 Winrate:      <b>{stats['winrate']}%</b>\n\n"
        f"📊 Средний P&L:  <b>{stats['avg_pnl']:+.2f}%</b>\n"
        f"🏆 Лучшая:       <b>{stats['best_trade']:+.2f}%</b>\n"
        f"💀 Худшая:       <b>{stats['worst_trade']:+.2f}%</b>\n"
        f"🔄 Trailing WIN: <b>{stats['trailing_wins']}</b>\n"
        f"🎯 Breakeven:    <b>{stats['breakeven_hits']}</b>\n"
        f"📐 Kelly (Half): <b>{kelly:.1%}</b>\n"
        f"⏳ Открыто:      <b>{stats['open_trades']}</b>"
    )