"""
paper_trader.py v4.0 — Виртуальные сделки с продвинутым риск-менеджментом
v4.0 изменения:
  - Trailing Stop-Loss: при +1.5% тянем SL за ценой на 0.8%
  - Breakeven: при +1.0% переносим SL в точку входа
  - Динамический размер позиции: STRONG_SIGNAL → 15%, обычный → 10%
  - Частичный тейк: при +2% закрываем 50%, остаток держим с trailing
  - Полная история всех обновлений SL/TP
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

INITIAL_BALANCE  = 600.0
TRADE_PCT        = 0.10    # 10% баланса — обычный сигнал
TRADE_PCT_STRONG = 0.15    # 15% — STRONG_SIGNAL ≥ 0.85

OKX_CONFIG = {'options': {'defaultType': 'spot'}, 'timeout': 30000}

logger = logging.getLogger(__name__)


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
# Расчёт SL/TP
# ─────────────────────────────────────────────
def _calc_sl_tp(signal: str, price: float, atr: float = 0.0) -> tuple:
    use_atr = atr > 0

    if use_atr:
        raw_sl_pct = (atr * ATR_SL_MULT) / price
        sl_pct     = max(SL_FLOOR_PCT, min(raw_sl_pct, SL_CAP_PCT))
        tp_pct     = sl_pct * (ATR_TP_MULT / ATR_SL_MULT)
        mode       = f"ATR×{ATR_SL_MULT} (raw={raw_sl_pct:.2%}→{sl_pct:.2%})"
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

    # Динамический размер: сильный сигнал → +50% размера
    trade_pct  = TRADE_PCT_STRONG if confidence >= STRONG_SIGNAL else TRADE_PCT
    amount_usd = round(balance["balance"] * trade_pct, 2)
    qty        = round(amount_usd / price, 4)

    sl, tp, sl_mode = _calc_sl_tp(signal, price, atr)

    logger.info(
        f"[Paper] SL/TP: {sl_mode} | SL=${sl:.4f} TP=${tp:.4f} | "
        f"Размер: {trade_pct:.0%} (${amount_usd:.2f}) | "
        f"R:R = 1:{round(abs(tp - price) / (abs(price - sl) + 1e-9), 2)}"
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
        "sl_initial":      sl,    # оригинальный SL для истории
        "atr":             round(atr, 6),
        "sl_mode":         sl_mode,
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
        "max_price":       price,  # максимум цены с момента открытия (для BUY)
        "min_price":       price,  # минимум цены (для SELL)
        # Meta
        "extra_info":      extra_info or {},
        "sl_updates":      [],     # история изменений SL
    }

    trades.append(trade)
    save_trades(trades)

    logger.info(
        f"[Paper] 📝 Открыта #{trade['id']}: "
        f"{signal} {symbol} @ {price} | TP={tp} SL={sl}"
    )
    return trade


# ─────────────────────────────────────────────
# Обновление Trailing Stop
# ─────────────────────────────────────────────
def _update_trailing(trade: dict, price: float) -> dict:
    """
    Обновляет trailing SL при движении цены в прибыльную сторону.
    Возвращает обновлённый trade dict.
    """
    if not TRAILING_ENABLED:
        return trade

    signal     = trade["signal"]
    price_open = trade["price_open"]
    sl         = trade["sl"]
    updated    = False

    if signal == "BUY":
        pnl_pct = (price - price_open) / price_open

        # Обновляем максимум
        if price > trade["max_price"]:
            trade["max_price"] = price

        # Breakeven: при +1.0% → SL в точку входа
        if (not trade["breakeven_hit"]
                and pnl_pct >= BREAKEVEN_ACTIVATION
                and sl < price_open):
            new_sl = round(price_open * 1.0001, 6)  # чуть выше входа
            if new_sl > sl:
                trade["sl_updates"].append({
                    "time": _now(), "old_sl": sl,
                    "new_sl": new_sl, "reason": "BREAKEVEN"
                })
                trade["sl"]           = new_sl
                trade["breakeven_hit"] = True
                updated = True
                logger.info(f"[Paper] 🔄 #{trade['id']} BREAKEVEN SL: ${sl:.4f} → ${new_sl:.4f}")

        # Trailing: при +1.5% → тянем SL на TRAILING_DISTANCE_PCT ниже цены
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
                updated = True
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
                updated = True

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
                updated = True

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

        # Обновляем trailing перед проверкой TP/SL
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
        else:  # SELL
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
            trades[i] = trade  # Сохраняем trailing-апдейты
            continue

        # Закрытие
        close_price = tp if hit == "TP" else sl

        if signal == "BUY":
            pnl_pct = (close_price - trade["price_open"]) / trade["price_open"] * 100
        else:
            pnl_pct = (trade["price_open"] - close_price) / trade["price_open"] * 100

        pnl_usd = round(trade["amount_usd"] * pnl_pct / 100, 2)
        result  = "WIN" if hit == "TP" else "LOSS"

        # Trailing SL-закрытие считается WIN если ≥ breakeven
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

    # Trailing статистика
    trailing_wins  = sum(1 for t in closed if t.get("closed_by") == "SL_TRAILING" and t.get("result") == "WIN")
    breakeven_hits = sum(1 for t in closed if t.get("breakeven_hit"))

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
    }


def format_stats_message(stats: dict) -> str:
    emoji = "📈" if stats["growth_pct"] >= 0 else "📉"
    return (
        f"📊 <b>Paper Trading v4.0 — Статистика</b>\n\n"
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
        f"⏳ Открыто:      <b>{stats['open_trades']}</b>"
    )