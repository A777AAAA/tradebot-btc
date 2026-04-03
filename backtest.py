"""
backtest.py v8.0 — Реальный Slippage + Честный Out-of-Sample + Калиброванные модели
ИЗМЕНЕНИЯ v8.0 vs v7.0:
  - РЕАЛЬНЫЙ SLIPPAGE: 0.15% при входе + 0.15% при выходе (итого ~0.40% round-trip)
    Это реалистичная оценка для TON/USDT на OKX при исполнении по рынку.
    Без slippage бэктест всегда будет выглядеть лучше чем live-трейдинг.
  - ЧЕСТНЫЙ OOS ТЕСТ: последние 20% данных (только тест, не 30%)
    + дополнительный разрез: результаты по кварталам
  - Поддержка calibrated_model_buy.pkl (v8.0)
  - Добавлен Profit Factor по периодам (кварталам)
  - Добавлен Average Trade Duration (для оценки частоты)
  - Добавлен Expectancy (математическое ожидание на сделку)
  - Реальные SL/TP уровни с учётом slippage
"""

import os
import json
import joblib
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# ПАРАМЕТРЫ БЭКТЕСТА v8.0
# ─────────────────────────────────────────────

SLIPPAGE_PCT   = 0.0015   # 0.15% slippage при входе и выходе
COMMISSION_PCT = 0.001    # 0.1% OKX taker комиссия


def load_local_model():
    """
    Загружает лучшую доступную модель.
    v8.0: Приоритет у калиброванной модели (точнее вероятности).
    """
    priority = [
        "calibrated_model_buy.pkl",   # v8.0 НОВОЕ — лучшие вероятности
        "stack_model_buy.pkl",
        "model_buy_xgb.pkl",
        "model_buy_lgbm.pkl",
    ]
    for path in priority:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                print(f"✅ Загружена модель: {path}")
                return model, path
            except Exception as e:
                print(f"⚠️ Не удалось загрузить {path}: {e}")
    return None, None


def load_feature_cols():
    """Загружает список фичей из model_features.json или использует дефолтный."""
    if os.path.exists("model_features.json"):
        with open("model_features.json") as f:
            cols = json.load(f)
        print(f"✅ Загружено {len(cols)} фичей из model_features.json")
        return cols

    print("⚠️ model_features.json не найден — использую дефолтный набор")
    return [
        'RSI_14', 'RSI_7', 'RSI_21',
        'MACD', 'MACD_signal', 'MACD_hist',
        'ATR_pct', 'ATR_norm', 'ATR_ratio',
        'ADX', 'BB_pos', 'BB_width',
        'EMA_ratio_20_50', 'EMA_ratio_20_100',
        'Vol_ratio', 'OBV_norm', 'MFI_14',
        'Body_pct', 'Upper_wick', 'Lower_wick', 'Doji',
        'Return_1h', 'Return_4h', 'Return_12h', 'Return_24h',
        'StochRSI_K', 'StochRSI_D',
        'ZScore_20', 'ZScore_50',
        'WilliamsR', 'Hour', 'DayOfWeek',
        'Momentum_10', 'ROC_10',
    ]


def get_kelly_size() -> float:
    """Читает Kelly fraction из training_stats.json."""
    if os.path.exists("training_stats.json"):
        try:
            with open("training_stats.json") as f:
                stats = json.load(f)
            kelly = float(stats.get("kelly_fraction", 0.0))
            if kelly > 0.03:
                print(f"✅ Kelly Fraction из обучения: {kelly:.1%}")
                return kelly
        except Exception:
            pass
    print("⚠️ Kelly не найден — использую 10% по умолчанию")
    return 0.10


def calc_sharpe(returns: list, annualize: bool = True) -> float:
    """Sharpe Ratio по списку % доходностей сделок."""
    if len(returns) < 3:
        return 0.0
    arr = np.array(returns)
    if arr.std() == 0:
        return 0.0
    factor = np.sqrt(4380) if annualize else 1.0
    return float((arr.mean() / arr.std()) * factor)


def calc_max_drawdown(equity_curve: list) -> float:
    """Максимальная просадка от пика."""
    if len(equity_curve) < 2:
        return 0.0
    peak   = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 2)


def calc_expectancy(trade_returns: list) -> float:
    """
    Математическое ожидание на сделку (%).
    Expectancy = WR × avg_win - LR × avg_loss
    Положительная expectancy = система выгодна.
    """
    if not trade_returns:
        return 0.0
    arr   = np.array(trade_returns)
    wins  = arr[arr > 0]
    losses = arr[arr < 0]
    if len(wins) == 0 or len(losses) == 0:
        return float(arr.mean())
    win_rate  = len(wins) / len(arr)
    loss_rate = 1 - win_rate
    return round(float(win_rate * wins.mean() + loss_rate * losses.mean()), 3)


def run_advanced_backtest():
    print("=" * 60)
    print("📈 Бэктест v8.0 — Slippage + Честный OOS + Calibrated модели")
    print("=" * 60)

    # 1. Загрузка модели
    model, model_path = load_local_model()
    if model is None:
        print("❌ Нет доступных моделей. Сначала запусти обучение (auto_trainer.py).")
        return

    # 2. Загрузка данных
    data_files = ["ml_ready_ton_data_v2.csv", "okx_ton_data.csv"]
    df = None
    for fname in data_files:
        if os.path.exists(fname):
            try:
                df = pd.read_csv(fname, index_col='Timestamp', parse_dates=True)
                print(f"✅ Данные: {fname} — {len(df)} строк")
                break
            except Exception as e:
                print(f"⚠️ Ошибка чтения {fname}: {e}")

    if df is None or df.empty:
        print("❌ Файл данных не найден.")
        return

    # 3. Фичи
    feature_cols = load_feature_cols()
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        print(f"⚠️ Отсутствующих колонок: {len(missing)}")
        feature_cols = [f for f in feature_cols if f in df.columns]
        print(f"ℹ️ Продолжаем с {len(feature_cols)} признаками")

    if len(feature_cols) == 0:
        print("❌ Нет ни одного подходящего признака")
        return

    required_ohlcv = ['Close', 'High', 'Low', 'ATR']
    missing_ohlcv  = [c for c in required_ohlcv if c not in df.columns]
    if missing_ohlcv:
        print(f"❌ Не хватает обязательных колонок: {missing_ohlcv}")
        return

    # 4. v8.0: честный OOS — последние 20%
    test_size = max(int(len(df) * 0.20), 100)
    test_df   = df.tail(test_size).copy()
    X_test    = test_df[feature_cols].fillna(0).values

    print(f"\n🔬 Тест на {test_size} свечах (честный OOS — последние 20%)")
    print(f"   Период: {test_df.index[0]} → {test_df.index[-1]}")
    print(f"   Slippage: {SLIPPAGE_PCT*100:.2f}% вход + {SLIPPAGE_PCT*100:.2f}% выход")
    print(f"   Комиссия: {COMMISSION_PCT*100:.2f}% × 2 = {COMMISSION_PCT*200:.2f}% round-trip")

    # 5. Предсказания
    print("🧠 Генерация вероятностей...")
    try:
        probs = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"❌ Ошибка предсказания: {e}")
        return
    test_df = test_df.copy()
    test_df['Signal_Prob'] = probs

    # 6. Торговые параметры
    initial_balance = 1000.0
    balance         = initial_balance

    kelly_base = get_kelly_size()

    position          = 0
    entry_price       = 0.0
    entry_price_real  = 0.0  # v8.0: цена с учётом slippage
    position_size_usd = 0.0
    take_profit       = 0.0
    stop_loss         = 0.0
    trade_open_idx    = 0

    equity_curve   = [initial_balance]
    trade_returns  = []
    winning_trades = 0
    losing_trades  = 0
    consecutive_losses = 0
    trade_durations = []  # v8.0: длительности сделок

    # Для квартального анализа
    quarterly_pnl = {}

    print(f"\n💰 Старт: ${initial_balance:.2f} | Kelly: {kelly_base:.1%}")
    print(f"   Slippage: {SLIPPAGE_PCT*100:.2f}% | Комиссия: {COMMISSION_PCT*100:.2f}%\n")

    for i in range(len(test_df)):
        row          = test_df.iloc[i]
        current_close = float(row['Close'])
        current_high  = float(row['High'])
        current_low   = float(row['Low'])
        current_atr   = float(row.get('ATR', current_close * 0.01))
        prob          = float(row['Signal_Prob'])

        # Квартал для анализа
        try:
            quarter = f"{test_df.index[i].year}Q{(test_df.index[i].month-1)//3+1}"
        except Exception:
            quarter = "unknown"

        # Защита от серии убытков
        loss_penalty = max(0.5, 1.0 - consecutive_losses * 0.15)

        # --- ЛОГИКА ВЫХОДА ---
        if position == 1:
            # v8.0: SL/TP с учётом slippage при выходе
            if current_low <= stop_loss:
                # При SL: выходим по stop_loss + slippage (хуже)
                exit_price = stop_loss * (1 - SLIPPAGE_PCT)  # проскальзываем вниз
                pnl_pct        = (exit_price - entry_price_real) / entry_price_real
                balance_change = position_size_usd * pnl_pct
                commission_out = position_size_usd * COMMISSION_PCT
                balance       += balance_change - commission_out
                balance        = max(balance, 0)
                position       = 0
                trade_returns.append(pnl_pct * 100)
                losing_trades += 1
                consecutive_losses += 1
                duration = i - trade_open_idx
                trade_durations.append(duration)
                equity_curve.append(balance)

                # Квартальный учёт
                quarterly_pnl[quarter] = quarterly_pnl.get(quarter, 0) + (balance_change - commission_out)

                print(
                    f"  🔴 SL | p={prob:.2f} | Вход={entry_price_real:.4f} "
                    f"Выход={exit_price:.4f} | P&L={pnl_pct*100:+.2f}% "
                    f"| Баланс=${balance:.2f} | Длит={duration}h"
                )
                continue

            elif current_high >= take_profit:
                # При TP: выходим по take_profit - slippage (чуть хуже)
                exit_price = take_profit * (1 - SLIPPAGE_PCT * 0.5)  # при TP slippage меньше
                pnl_pct        = (exit_price - entry_price_real) / entry_price_real
                balance_change = position_size_usd * pnl_pct
                commission_out = position_size_usd * COMMISSION_PCT
                balance       += balance_change - commission_out
                position       = 0
                trade_returns.append(pnl_pct * 100)
                winning_trades    += 1
                consecutive_losses = 0
                duration = i - trade_open_idx
                trade_durations.append(duration)
                equity_curve.append(balance)

                quarterly_pnl[quarter] = quarterly_pnl.get(quarter, 0) + (balance_change - commission_out)

                print(
                    f"  🟢 TP | p={prob:.2f} | Вход={entry_price_real:.4f} "
                    f"Выход={exit_price:.4f} | P&L={pnl_pct*100:+.2f}% "
                    f"| Баланс=${balance:.2f} | Длит={duration}h"
                )
                continue

        # --- ЛОГИКА ВХОДА ---
        if position == 0 and balance > 1:

            # Drawdown guard
            peak_balance = max(equity_curve)
            current_dd   = (peak_balance - balance) / peak_balance * 100
            if current_dd > 20.0:
                continue

            if prob >= 0.75:
                risk_allocation = kelly_base * 1.25 * loss_penalty
                risk_allocation = min(risk_allocation, 0.30)
            elif prob >= 0.62:
                risk_allocation = kelly_base * 1.0 * loss_penalty
            elif prob >= 0.58:
                risk_allocation = kelly_base * 0.75 * loss_penalty
            else:
                risk_allocation = 0.0

            if risk_allocation > 0.03:
                # v8.0: реальная цена входа с учётом slippage
                entry_price      = current_close
                entry_price_real = current_close * (1 + SLIPPAGE_PCT)  # покупаем дороже
                position_size_usd = round(balance * risk_allocation, 2)

                # ATR-based SL/TP (считаем от реальной цены входа)
                atr_sl_mult = 1.5
                atr_tp_mult = 3.0
                stop_loss   = entry_price_real - (current_atr * atr_sl_mult)
                take_profit = entry_price_real + (current_atr * atr_tp_mult)

                # Комиссия за вход
                commission_in = position_size_usd * COMMISSION_PCT
                balance      -= commission_in
                position      = 1
                trade_open_idx = i

                rr = (take_profit - entry_price_real) / (entry_price_real - stop_loss + 1e-9)
                slippage_cost = position_size_usd * SLIPPAGE_PCT
                print(
                    f"  ⚡ ВХОД | p={prob:.2f} | size={risk_allocation*100:.0f}% "
                    f"(${position_size_usd:.1f}) | "
                    f"Цена={entry_price_real:.4f} (slip=${slippage_cost:.3f}) | "
                    f"SL={stop_loss:.4f} | TP={take_profit:.4f} | R:R=1:{rr:.1f}"
                )

    # Принудительное закрытие
    if position == 1:
        last_price    = float(test_df['Close'].iloc[-1])
        exit_price    = last_price * (1 - SLIPPAGE_PCT)
        pnl_pct       = (exit_price - entry_price_real) / entry_price_real
        balance_change = position_size_usd * pnl_pct
        commission_out = position_size_usd * COMMISSION_PCT
        balance       += balance_change - commission_out
        balance        = max(balance, 0)
        trade_returns.append(pnl_pct * 100)
        if pnl_pct >= 0:
            winning_trades += 1
        else:
            losing_trades  += 1
        equity_curve.append(balance)
        print(f"  ⏰ ПРИНУД. ЗАКРЫТИЕ | P&L={pnl_pct*100:+.2f}%")

    # 7. Метрики
    total_trades  = winning_trades + losing_trades
    win_rate      = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    total_return  = ((balance - initial_balance) / initial_balance) * 100
    sharpe        = calc_sharpe(trade_returns)
    max_dd        = calc_max_drawdown(equity_curve)
    calmar        = abs(total_return / max_dd) if max_dd > 0 else 0.0
    expectancy    = calc_expectancy(trade_returns)

    wins_list  = [r for r in trade_returns if r > 0]
    losses_list = [r for r in trade_returns if r < 0]
    avg_win    = np.mean(wins_list)  if wins_list  else 0
    avg_loss   = np.mean(losses_list) if losses_list else 0
    profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades + 1e-9))

    avg_duration = np.mean(trade_durations) if trade_durations else 0

    # Общий slippage cost
    total_slippage = total_trades * (initial_balance / total_trades if total_trades > 0 else 0) * SLIPPAGE_PCT * 2
    total_commission = total_trades * (initial_balance / total_trades if total_trades > 0 else 0) * COMMISSION_PCT * 2

    print(f"\n{'='*60}")
    print(f"📊 ИТОГИ БЭКТЕСТА v8.0")
    print(f"{'='*60}")
    print(f"🏁 Финальный баланс:  ${balance:.2f}  ({total_return:+.2f}%)")
    print(f"📋 Всего сделок:      {total_trades}")
    print(f"🏆 Win Rate:          {win_rate:.1f}%")
    print(f"✅ Плюсовых:         {winning_trades}  |  ❌ Минусовых: {losing_trades}")
    print(f"📊 Средний WIN:       {avg_win:+.2f}%  |  Средний LOSS: {avg_loss:+.2f}%")
    print(f"💡 Expectancy:        {expectancy:+.3f}% на сделку")
    print(f"⚡ Profit Factor:     {profit_factor:.2f}")
    print(f"📉 Max Drawdown:      {max_dd:.2f}%")
    print(f"📐 Sharpe Ratio:      {sharpe:.2f}")
    print(f"🏋 Calmar Ratio:      {calmar:.2f}")
    print(f"⏱  Средн. длит.:     {avg_duration:.1f}h (~{avg_duration/24:.1f} дней)")
    print(f"💸 Slip+Comms impact: ~${total_slippage + total_commission:.2f} (оценка)")
    print(f"🔢 Модель:            {model_path}")

    # Квартальный разбор
    if quarterly_pnl:
        print(f"\n📅 P&L по кварталам:")
        for q, pnl in sorted(quarterly_pnl.items()):
            emoji = "✅" if pnl >= 0 else "❌"
            print(f"   {emoji} {q}: ${pnl:+.2f}")

    print(f"{'='*60}")

    # Интерпретация
    print("\n💡 Интерпретация:")
    if sharpe >= 1.5:
        print("   ✅ Sharpe ≥ 1.5 — профессиональный уровень")
    elif sharpe >= 0.8:
        print("   ⚠️ Sharpe 0.8-1.5 — приемлемо, есть резерв")
    else:
        print("   ❌ Sharpe < 0.8 — модель требует доработки")

    if expectancy > 0.1:
        print("   ✅ Положительное математическое ожидание")
    elif expectancy > 0:
        print("   ⚠️ Marginally положительное ожидание — нужно больше данных")
    else:
        print("   ❌ Отрицательное ожидание — система убыточна")

    if max_dd <= 15:
        print("   ✅ Max Drawdown ≤ 15% — контролируемый риск")
    elif max_dd <= 25:
        print("   ⚠️ Max Drawdown 15-25% — допустимо")
    else:
        print("   ❌ Max Drawdown > 25% — снизить размеры позиций")

    if profit_factor >= 1.5:
        print("   ✅ Profit Factor ≥ 1.5 — система прибыльна")
    elif profit_factor >= 1.0:
        print("   ⚠️ Profit Factor 1.0-1.5 — на грани")
    else:
        print("   ❌ Profit Factor < 1.0 — убыточная система")

    return {
        "total_return":   round(total_return, 2),
        "total_trades":   total_trades,
        "win_rate":       round(win_rate, 1),
        "sharpe":         round(sharpe, 2),
        "max_drawdown":   max_dd,
        "calmar":         round(calmar, 2),
        "profit_factor":  round(profit_factor, 2),
        "expectancy":     expectancy,
        "avg_duration_h": round(avg_duration, 1),
        "final_balance":  round(balance, 2),
        "slippage_used":  SLIPPAGE_PCT,
        "quarterly_pnl":  quarterly_pnl,
    }


if __name__ == "__main__":
    run_advanced_backtest()