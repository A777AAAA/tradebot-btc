"""
paper_trader.py v5.1 — Защита от серии убытков + Drawdown Guard
ИЗМЕНЕНИЯ v5.1:
  - Consecutive Loss Penalty: после 3+ стопов подряд Kelly снижается на 15%/стоп
  - Drawdown Guard: если баланс упал > 20% от пика — сделки не открываются
    (торговля возобновляется когда просадка < 20%)
  - Счётчик consecutive_losses сохраняется в paper_balance.json
  - Все остальные механизмы v5.0 сохранены полностью:
    Kelly Criterion, ATR-based SL/TP, Trailing Stop, Breakeven
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
TRADE_PCT_DEFAULT        = 0.10
TRADE_PCT_STRONG_DEFAULT = 0.15
TRADE_PCT_MAX            = 0.25   # Risk-of-Ruin защита
TRADE_PCT_MIN            = 0.05

# v5.1: параметры защиты
MAX_DRAWDOWN_GUARD_PCT   = 20.0   # Блокируем торговлю при просадке > 20%
CONSEC_LOSS_PENALTY      = 0.15   # -15% Kelly на каждый стоп-лосс подряд

OKX_CONFIG = {'options': {'defaultType': 'spot'}, 'timeout': 30000}

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Kelly Criterion Position Sizing
# ─────────────────────────────────────────────
def get_kelly_trade_pct(confidence: float, consecutive_losses: int = 0) -> float:
    """
    Читает Kelly Fraction из последнего обучения.
    v5.1: учитывает consecutive_losses для снижения позиции после серии убытков.
    """
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE) as f:
                stats = json.load(f)

            kelly_f = float(stats.get("kelly_fraction", 0.0))

            if kelly_f > 0.03:
                # Корректировка по уверенности
                if confidence >= STRONG_SIGNAL:
                    trade_pct = kelly_f * 1.25
                elif confidence >= 0.65:
                    trade_pct = kelly_f * 1.0
                else:
                    trade_pct = kelly_f * 0.75

                # v5.1: Consecutive Loss Penalty
                if consecutive_losses >= 2:
                    penalty   = max(0.40, 1.0 - consecutive_losses * CONSEC_LOSS_PENALTY)
                    trade_pct = trade_pct * penalty
                    logger.info(
                        f"[Paper] ⚠️ Consecutive losses={consecutive_losses} "
                        f"→ Kelly penalty={penalty:.0%} → size={trade_pct:.1%}"
                    )

                trade_pct = max(TRADE_PCT_MIN, min(trade_pct, TRADE_PCT_MAX))
                logger.debug(
                    f"[Paper] Kelly={kelly_f:.1%} conf={confidence:.1%} "
                    f"consec_loss={consecutive_losses} → size={trade_pct:.1%}"
                )
                return round(trade_pct, 3)

    except Exception as e:
        logger.debug(f"[Paper] Kelly чтение ошибка: {e}")

    # Fallback
    base = TRADE_PCT_STRONG_DEFAULT if confidence >= STRONG_SIGNAL else TRADE_PCT_DEFAULT
    if consecutive_losses >= 2:
        penalty = max(0.40, 1.0 - consecutive_losses * CONSEC_LOSS_PENALTY)
        base    = max(TRADE_PCT_MIN, base * penalty)
    return base


# ─────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_balance() -> dict:
    if not os.path.exists(BALANCE_FILE):
        data = {
            "balance":           INITIAL_BALANCE,
            "peak_balance":      INITIAL_BALANCE,
            "total_pnl":         0.0,
            "trades":            0,
            "wins":              0,
            "losses":            0,
            "consecutive_losses": 0,
            "created_at":        _now()
        }
        save_balance(data)
        return data
    with open(BALANCE_FILE) as f:
        data = json.load(f)
    # Обратная совместимость: добавляем новые поля если их нет
    if "consecutive_losses" not in data:
        data["consecutive_losses"] = 0
    if "peak_balance" not in data:
        data["peak_balance"] = max(data.get("balance", INITIAL_BALANCE), INITIAL_BALANCE)
    return data


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
# Проверка Drawdown Guard (v5.1)
# ─────────────────────────────────────────────
def check_drawdown_guard(balance_data: dict) -> tuple:
    """
    Возвращает (is_blocked, current_drawdown_pct).
    Если просадка > MAX_DRAWDOWN_GUARD_PCT — торговля заблокирована.
    """
    balance      = balance_data.get("balance", INITIAL_BALANCE)
    peak_balance = balance_data.get("peak_balance", balance)

    # Обновляем пик
    if balance > peak_balance:
        peak_balance = balance
        balance_data["peak_balance"] = peak_balance

    drawdown_pct = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0.0
    is_blocked   = drawdown_pct >= MAX_DRAWDOWN_GUARD_PCT

    return is_blocked, round(drawdown_pct, 2)


# ─────────────────────────────────────────────
# Расчёт SL/TP
# ─────────────────────────────────────────────
def _calc_sl_tp(signal: str, price: float, atr: float = 0.0) -> tuple:
    use_atr = atr > 0

    if use_atr:
        raw_sl_pct = (atr * ATR_SL_MULT) / price
        sl_pct     = max(SL_FLOOR_PCT, min(raw_sl_pct, SL_CAP_PCT))
        raw_tp_pct = (atr * ATR_TP_MULT) / price
        tp_pct     = max(SL_FLOOR_PCT * 2, raw_tp_pct)
        rr_ratio   = round(raw_tp_pct / (raw_sl_pct + 1e-9), 2)
        mode       = f"ATR×SL={ATR_SL_MULT}/TP={ATR_TP_MULT} R:R≈{rr_ratio}"
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
# Trailing Stop
# ─────────────────────────────────────────────
def _update_trailing(trade: dict, price: float) -> dict:
    if not TRAILING_ENABLED:
        return trade

    signal     = trade["signal"]
    price_open = trade["price_open"]
    sl         = trade["sl"]

    if signal == "BUY":
        pnl_pct = (price - price_open) / price_open

        if price > trade.get("max_price", price_open):
            trade["max_price"] = price

        if (not trade.get("breakeven_hit", False)
                and pnl_pct >= BREAKEVEN_ACTIVATION
                and sl < price_open):
            new_sl = round(price_open * 1.0001, 6)
            if new_sl > sl:
                trade.setdefault("sl_updates", []).append({
                    "time": _now(), "old_sl": sl,
                    "new_sl": new_sl, "reason": "BREAKEVEN"
                })
                trade["sl"]            = new_sl
                trade["breakeven_hit"] = True
                logger.info(f"[Paper] 🔄 #{trade['id']} BREAKEVEN SL: ${sl:.4f} → ${new_sl:.4f}")

        if pnl_pct >= TRAILING_ACTIVATION_PCT:
            trade["trailing_active"] = True
            trailing_sl = round(price * (1 - TRAILING_DISTANCE_PCT), 6)
            if trailing_sl > trade["sl"]:
                old_sl = trade["sl"]
                trade.setdefault("sl_updates", []).append({
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

        if price < trade.get("min_price", price_open):
            trade["min_price"] = price

        if (not trade.get("breakeven_hit", False)
                and pnl_pct >= BREAKEVEN_ACTIVATION
                and sl > price_open):
            new_sl = round(price_open * 0.9999, 6)
            if new_sl < sl:
                trade.setdefault("sl_updates", []).append({
                    "time": _now(), "old_sl": sl,
                    "new_sl": new_sl, "reason": "BREAKEVEN"
                })
                trade["sl"]            = new_sl
                trade["breakeven_hit"] = True

        if pnl_pct >= TRAILING_ACTIVATION_PCT:
            trade["trailing_active"] = True
            trailing_sl = round(price * (1 + TRAILING_DISTANCE_PCT), 6)
            if trailing_sl < trade["sl"]:
                old_sl = trade["sl"]
                trade.setdefault("sl_updates", []).append({
                    "time": _now(), "old_sl": old_sl,
                    "new_sl": trailing_sl, "reason": "TRAILING"
                })
                trade["sl"] = trailing_sl

    return trade


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

    balance_data = load_balance()

    # v5.1: Drawdown Guard
    is_blocked, drawdown_pct = check_drawdown_guard(balance_data)
    if is_blocked:
        logger.warning(
            f"[Paper] 🚫 DRAWDOWN GUARD — просадка {drawdown_pct:.1f}% ≥ {MAX_DRAWDOWN_GUARD_PCT}%! "
            f"Сделки заблокированы до восстановления."
        )
        return None

    # v5.1: Kelly с учётом серии убытков
    consecutive_losses = balance_data.get("consecutive_losses", 0)
    trade_pct  = get_kelly_trade_pct(confidence, consecutive_losses)
    amount_usd = round(balance_data["balance"] * trade_pct, 2)
    qty        = round(amount_usd / price, 4)

    sl, tp, sl_mode = _calc_sl_tp(signal, price, atr)
    rr = round(abs(tp - price) / (abs(price - sl) + 1e-9), 2)

    logger.info(
        f"[Paper] {signal} | SL/TP: {sl_mode} | SL=${sl:.4f} TP=${tp:.4f} | "
        f"Kelly: {trade_pct:.1%} (${amount_usd:.2f}) | R:R=1:{rr} | "
        f"DrawDown={drawdown_pct:.1f}% | ConLoss={consecutive_losses}"
    )

    trade = {
        "id":               len(trades) + 1,
        "status":           "OPEN",
        "signal":           signal,
        "symbol":           symbol,
        "price_open":       price,
        "sl":               sl,
        "tp":               tp,
        "sl_mode":          sl_mode,
        "amount_usd":       amount_usd,
        "qty":              qty,
        "confidence":       round(confidence, 4),
        "kelly_pct":        trade_pct,
        "consecutive_loss_at_open": consecutive_losses,
        "opened_at":        _now(),
        "price_close":      None,
        "pnl_pct":          None,
        "pnl_usd":          None,
        "result":           None,
        "closed_at":        None,
        "closed_by":        None,
        "trailing_active":  False,
        "breakeven_hit":    False,
        "max_price":        price,
        "min_price":        price,
        "sl_updates":       [],
        **(extra_info or {}),
    }

    trades.append(trade)
    save_trades(trades)

    # Обновляем пик баланса
    balance_data["peak_balance"] = max(
        balance_data.get("peak_balance", INITIAL_BALANCE),
        balance_data["balance"]
    )
    save_balance(balance_data)

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

    balance_data = load_balance()
    closed_now   = []

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
                hit = "SL_TRAILING" if trade.get("trailing_active") else "SL"
        else:
            if price <= tp:
                hit = "TP"
            elif price >= sl:
                hit = "SL_TRAILING" if trade.get("trailing_active") else "SL"

        if not hit:
            if signal == "BUY":
                float_pnl = (price - trade["price_open"]) / trade["price_open"] * 100
            else:
                float_pnl = (trade["price_open"] - price) / trade["price_open"] * 100

            trailing_str = "🔄 TRAIL" if trade.get("trailing_active") else ""
            be_str       = "🎯 BE"    if trade.get("breakeven_hit")   else ""
            dd_blocked, dd_pct = check_drawdown_guard(balance_data)
            dd_warn = f" ⚠️ DD={dd_pct:.1f}%" if dd_pct >= 10 else ""

            logger.info(
                f"[Paper] ⏳ #{trade['id']} {signal} | "
                f"P&L: {float_pnl:+.2f}% | "
                f"Price={price:.4f} TP={tp:.4f} SL={sl:.4f} "
                f"{trailing_str}{be_str}{dd_warn}"
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

        balance_data["balance"]   = round(balance_data["balance"] + pnl_usd, 2)
        balance_data["total_pnl"] = round(balance_data.get("total_pnl", 0) + pnl_usd, 2)
        balance_data["trades"]    = balance_data.get("trades", 0) + 1
        balance_data["wins"]      = balance_data.get("wins", 0) + (1 if result == "WIN" else 0)
        balance_data["losses"]    = balance_data.get("losses", 0) + (0 if result == "WIN" else 1)

        # v5.1: обновляем consecutive_losses
        if result == "WIN":
            balance_data["consecutive_losses"] = 0
        else:
            balance_data["consecutive_losses"] = balance_data.get("consecutive_losses", 0) + 1

        # Обновляем пик
        if balance_data["balance"] > balance_data.get("peak_balance", INITIAL_BALANCE):
            balance_data["peak_balance"] = balance_data["balance"]

        closed_now.append(trades[i])
        cl = balance_data["consecutive_losses"]
        logger.info(
            f"[Paper] {'✅' if result == 'WIN' else '❌'} "
            f"#{trade['id']} закрыта по {hit}: "
            f"{pnl_pct:+.2f}% | ${pnl_usd:+.2f} | "
            f"Баланс: ${balance_data['balance']:.2f} | "
            f"ConLoss: {cl}"
        )

    save_trades(trades)
    save_balance(balance_data)
    return closed_now


# ─────────────────────────────────────────────
# Статистика
# ─────────────────────────────────────────────
def get_stats() -> dict:
    balance_data = load_balance()
    trades       = load_trades()
    closed       = [t for t in trades if t["status"] == "CLOSED"]

    total   = balance_data.get("trades", 0)
    wins    = balance_data.get("wins", 0)
    winrate = round(wins / total * 100, 1) if total > 0 else 0

    pnl_list    = [t["pnl_pct"] for t in closed if t.get("pnl_pct") is not None]
    avg_pnl     = round(sum(pnl_list) / len(pnl_list), 2) if pnl_list else 0
    best_trade  = round(max(pnl_list), 2) if pnl_list else 0
    worst_trade = round(min(pnl_list), 2) if pnl_list else 0
    growth_pct  = round(
        (balance_data["balance"] - INITIAL_BALANCE) / INITIAL_BALANCE * 100, 2
    )

    # Drawdown
    _, current_dd = check_drawdown_guard(balance_data)

    trailing_wins  = sum(1 for t in closed if t.get("closed_by") == "SL_TRAILING" and t.get("result") == "WIN")
    breakeven_hits = sum(1 for t in closed if t.get("breakeven_hit"))

    kelly_f = 0.0
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE) as f:
                s = json.load(f)
            kelly_f = float(s.get("kelly_fraction", 0.0))
    except Exception:
        pass

    return {
        "balance":            balance_data["balance"],
        "start_balance":      INITIAL_BALANCE,
        "growth_pct":         growth_pct,
        "total_pnl":          balance_data.get("total_pnl", 0),
        "total_trades":       total,
        "wins":               wins,
        "losses":             balance_data.get("losses", 0),
        "winrate":            winrate,
        "avg_pnl":            avg_pnl,
        "best_trade":         best_trade,
        "worst_trade":        worst_trade,
        "open_trades":        len([t for t in trades if t["status"] == "OPEN"]),
        "trailing_wins":      trailing_wins,
        "breakeven_hits":     breakeven_hits,
        "kelly_fraction":     kelly_f,
        "consecutive_losses": balance_data.get("consecutive_losses", 0),
        "current_drawdown":   current_dd,
        "peak_balance":       balance_data.get("peak_balance", INITIAL_BALANCE),
    }


def format_stats_message(stats: dict) -> str:
    emoji   = "📈" if stats["growth_pct"] >= 0 else "📉"
    kelly   = stats.get("kelly_fraction", 0)
    cl      = stats.get("consecutive_losses", 0)
    dd      = stats.get("current_drawdown", 0)
    dd_warn = f" ⚠️ DD={dd:.1f}%" if dd >= 10 else f" ({dd:.1f}%)"

    cl_line = ""
    if cl >= 2:
        cl_line = f"\n⚠️ Серия убытков:  <b>{cl}</b> (Kelly снижен!)"

    return (
        f"📊 <b>Paper Trading v5.1 — Статистика</b>\n\n"
        f"💰 Баланс:       <b>${stats['balance']:.2f}</b> {emoji}\n"
        f"📈 Рост:         <b>{stats['growth_pct']:+.2f}%</b>\n"
        f"💵 P&L всего:    <b>${stats['total_pnl']:+.2f}</b>\n"
        f"📉 DrawDown:     <b>{dd:.1f}%</b>{dd_warn}\n\n"
        f"📋 Сделок:       <b>{stats['total_trades']}</b>\n"
        f"✅ Побед:        <b>{stats['wins']}</b>\n"
        f"❌ Поражений:    <b>{stats['losses']}</b>\n"
        f"🎯 Winrate:      <b>{stats['winrate']}%</b>\n\n"
        f"📊 Средний P&L:  <b>{stats['avg_pnl']:+.2f}%</b>\n"
        f"🏆 Лучшая:       <b>{stats['best_trade']:+.2f}%</b>\n"
        f"💀 Худшая:       <b>{stats['worst_trade']:+.2f}%</b>\n"
        f"🔄 Trailing WIN: <b>{stats['trailing_wins']}</b>\n"
        f"🎯 Breakeven:    <b>{stats['breakeven_hits']}</b>\n"
        f"📐 Kelly (Half): <b>{kelly:.1%}</b>"
        f"{cl_line}\n"
        f"⏳ Открыто:      <b>{stats['open_trades']}</b>"
    )