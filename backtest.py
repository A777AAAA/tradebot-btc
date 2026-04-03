"""
backtest.py v7.0 — Standalone бэктест с локальными моделями
ИСПРАВЛЕНИЯ:
  - Убран hf_storage (HuggingFace) — использует только локальные модели
  - Обновлён список фичей: читается из model_features.json (динамически)
  - Добавлен Sharpe Ratio, Max Drawdown, Calmar Ratio в итоги
  - Kelly-размер позиции из training_stats.json
  - Полный консистентный код без внешних зависимостей
"""

import os
import json
import joblib
import numpy as np
import pandas as pd


def load_local_model():
    """Загружает лучшую доступную модель из локальных файлов."""
    # Приоритет: Stack > XGB > LGBM
    for path in ["stack_model_buy.pkl", "model_buy_xgb.pkl", "model_buy_lgbm.pkl"]:
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

    # Фоллбек — базовый набор v7.0
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
    # Предполагаем ~4380 1h-баров в год
    factor = np.sqrt(4380) if annualize else 1.0
    return float((arr.mean() / arr.std()) * factor)


def calc_max_drawdown(equity_curve: list) -> float:
    """Максимальная просадка от пика."""
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 2)


def run_advanced_backtest():
    print("=" * 55)
    print("📈 Бэктест v7.0 — Динамический риск + Локальные модели")
    print("=" * 55)

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
        print("❌ Файл данных не найден. Нужен ml_ready_ton_data_v2.csv или okx_ton_data.csv")
        return

    # 3. Определяем фичи
    feature_cols = load_feature_cols()
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        print(f"⚠️ Отсутствующих колонок: {len(missing)}: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        feature_cols = [f for f in feature_cols if f in df.columns]
        print(f"ℹ️ Продолжаем с {len(feature_cols)} доступными признаками")

    if len(feature_cols) == 0:
        print("❌ Нет ни одного подходящего признака в данных")
        return

    # Убеждаемся что есть OHLCV колонки
    required_ohlcv = ['Close', 'High', 'Low', 'ATR']
    missing_ohlcv  = [c for c in required_ohlcv if c not in df.columns]
    if missing_ohlcv:
        print(f"❌ Не хватает обязательных колонок: {missing_ohlcv}")
        return

    # 4. Тестовая выборка (последние 30%)
    test_size = max(int(len(df) * 0.30), 100)
    test_df   = df.tail(test_size).copy()
    X_test    = test_df[feature_cols].fillna(0).values

    print(f"\n🔬 Тест на {test_size} свечах (последние 30%)")
    print(f"   Период: {test_df.index[0]} → {test_df.index[-1]}")

    # 5. Предсказания
    print("🧠 Генерация вероятностей...")
    try:
        probs = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"❌ Ошибка предсказания: {e}")
        return
    test_df['Signal_Prob'] = probs

    # 6. Торговые параметры
    initial_balance = 1000.0
    balance         = initial_balance
    commission      = 0.001   # 0.1% OKX taker

    kelly_base   = get_kelly_size()

    position          = 0
    entry_price       = 0.0
    position_size_usd = 0.0
    take_profit       = 0.0
    stop_loss         = 0.0

    equity_curve   = [initial_balance]
    trade_returns  = []
    winning_trades = 0
    losing_trades  = 0
    consecutive_losses = 0   # защита от серии убытков

    print(f"\n💰 Старт: ${initial_balance:.2f} | Kelly: {kelly_base:.1%} | Комиссия: {commission*100:.1f}%\n")

    for i in range(len(test_df)):
        row          = test_df.iloc[i]
        current_close = float(row['Close'])
        current_high  = float(row['High'])
        current_low   = float(row['Low'])
        current_atr   = float(row.get('ATR', current_close * 0.01))
        prob          = float(row['Signal_Prob'])

        # Защита от серии убытков: снижаем Kelly
        loss_penalty = max(0.5, 1.0 - consecutive_losses * 0.15)

        # --- ЛОГИКА ВЫХОДА ---
        if position == 1:
            if current_low <= stop_loss:
                pnl_pct        = (stop_loss - entry_price) / entry_price
                balance_change = position_size_usd * pnl_pct
                balance       += balance_change - (position_size_usd * commission)
                balance        = max(balance, 0)
                position       = 0
                trade_returns.append(pnl_pct * 100)
                losing_trades += 1
                consecutive_losses += 1
                equity_curve.append(balance)
                print(f"  🔴 SL | Цена={current_close:.4f} | P&L={pnl_pct*100:+.2f}% | Баланс=${balance:.2f}")
                continue

            elif current_high >= take_profit:
                pnl_pct        = (take_profit - entry_price) / entry_price
                balance_change = position_size_usd * pnl_pct
                balance       += balance_change - (position_size_usd * commission)
                position       = 0
                trade_returns.append(pnl_pct * 100)
                winning_trades    += 1
                consecutive_losses = 0
                equity_curve.append(balance)
                print(f"  🟢 TP | Цена={current_close:.4f} | P&L={pnl_pct*100:+.2f}% | Баланс=${balance:.2f}")
                continue

        # --- ЛОГИКА ВХОДА ---
        if position == 0 and balance > 1:

            # Drawdown guard: если просадка > 20% — пауза
            peak_balance = max(equity_curve)
            current_dd   = (peak_balance - balance) / peak_balance * 100
            if current_dd > 20.0:
                continue  # Просадка > 20% — не открываем новые сделки

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
                entry_price       = current_close
                position_size_usd = round(balance * risk_allocation, 2)

                # ATR-based SL/TP (синхронизировано с paper_trader и config)
                atr_sl_mult = 1.5
                atr_tp_mult = 3.0
                stop_loss   = entry_price - (current_atr * atr_sl_mult)
                take_profit = entry_price + (current_atr * atr_tp_mult)

                # Комиссия за вход
                balance  -= position_size_usd * commission
                position  = 1

                rr = (take_profit - entry_price) / (entry_price - stop_loss + 1e-9)
                print(
                    f"  ⚡ ВХОД | p={prob:.2f} | size={risk_allocation*100:.0f}% "
                    f"(${position_size_usd:.1f}) | SL={stop_loss:.4f} | TP={take_profit:.4f} | R:R=1:{rr:.1f}"
                )

    # Принудительное закрытие если позиция осталась открытой
    if position == 1:
        last_price     = float(test_df['Close'].iloc[-1])
        pnl_pct        = (last_price - entry_price) / entry_price
        balance_change = position_size_usd * pnl_pct
        balance       += balance_change - (position_size_usd * commission)
        balance        = max(balance, 0)
        trade_returns.append(pnl_pct * 100)
        if pnl_pct >= 0:
            winning_trades += 1
        else:
            losing_trades  += 1
        equity_curve.append(balance)
        print(f"  ⏰ ПРИНУД. ЗАКРЫТИЕ | P&L={pnl_pct*100:+.2f}%")

    # 7. Итоги
    total_trades  = winning_trades + losing_trades
    win_rate      = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    total_return  = ((balance - initial_balance) / initial_balance) * 100
    sharpe        = calc_sharpe(trade_returns)
    max_dd        = calc_max_drawdown(equity_curve)
    calmar        = abs(total_return / max_dd) if max_dd > 0 else 0.0
    avg_win       = np.mean([r for r in trade_returns if r > 0]) if winning_trades > 0 else 0
    avg_loss      = np.mean([r for r in trade_returns if r < 0]) if losing_trades  > 0 else 0
    profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades + 1e-9))

    print(f"\n{'='*55}")
    print(f"📊 ИТОГИ БЭКТЕСТА v7.0")
    print(f"{'='*55}")
    print(f"🏁 Финальный баланс:  ${balance:.2f}  ({total_return:+.2f}%)")
    print(f"📋 Всего сделок:      {total_trades}")
    print(f"🏆 Win Rate:          {win_rate:.1f}%")
    print(f"✅ Плюсовых:         {winning_trades}  |  ❌ Минусовых: {losing_trades}")
    print(f"📊 Средний WIN:       {avg_win:+.2f}%  |  Средний LOSS: {avg_loss:+.2f}%")
    print(f"⚡ Profit Factor:     {profit_factor:.2f}")
    print(f"📉 Max Drawdown:      {max_dd:.2f}%")
    print(f"📐 Sharpe Ratio:      {sharpe:.2f}")
    print(f"🏋 Calmar Ratio:      {calmar:.2f}")
    print(f"🔢 Модель:           {model_path}")
    print(f"{'='*55}")

    # Интерпретация
    print("\n💡 Интерпретация:")
    if sharpe >= 1.5:
        print("   ✅ Sharpe ≥ 1.5 — отличный результат (уровень профессиональных фондов)")
    elif sharpe >= 0.8:
        print("   ⚠️ Sharpe 0.8-1.5 — приемлемо, но есть резерв")
    else:
        print("   ❌ Sharpe < 0.8 — модель требует доработки")

    if max_dd <= 15:
        print("   ✅ Max Drawdown ≤ 15% — контролируемый риск")
    elif max_dd <= 25:
        print("   ⚠️ Max Drawdown 15-25% — допустимо, но стоит ужесточить SL")
    else:
        print("   ❌ Max Drawdown > 25% — необходимо снизить размеры позиций")

    if profit_factor >= 1.5:
        print("   ✅ Profit Factor ≥ 1.5 — система прибыльна на длинных дистанциях")
    elif profit_factor >= 1.0:
        print("   ⚠️ Profit Factor 1.0-1.5 — на грани, нужно больше данных")
    else:
        print("   ❌ Profit Factor < 1.0 — убыточная система")

    return {
        "total_return": round(total_return, 2),
        "total_trades": total_trades,
        "win_rate":     round(win_rate, 1),
        "sharpe":       round(sharpe, 2),
        "max_drawdown": max_dd,
        "calmar":       round(calmar, 2),
        "profit_factor": round(profit_factor, 2),
        "final_balance": round(balance, 2),
    }


if __name__ == "__main__":
    run_advanced_backtest()