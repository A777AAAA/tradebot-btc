"""
backtest_engine.py v5.0 — ML-бэктест на реальных бинарных моделях
ИЗМЕНЕНИЯ:
  - Бэктест теперь использует сами ML-модели для генерации сигналов
  - Если модели не загружены — fallback на RSI+EMA стратегию
  - Пагинация OKX (реальные 3000 свечей)
  - ATR-based динамические SL/TP
  - Расчёт Sharpe ratio
"""

import requests
import joblib
import json
import os
import pandas as pd
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)

MODEL_PATH_BUY_XGB  = "model_buy_xgb.pkl"
MODEL_PATH_SELL_XGB = "model_sell_xgb.pkl"
MODEL_FEATURES_PATH = "model_features.json"


def fetch_history(symbol: str = "TON-USDT",
                  timeframe: str = "1H", limit: int = 3000) -> pd.DataFrame:
    all_data = []
    after    = None
    fetched  = 0
    per_req  = 300

    try:
        while fetched < limit:
            url = (
                f"https://www.okx.com/api/v5/market/history-candles"
                f"?instId={symbol}&bar={timeframe}&limit={per_req}"
            )
            if after:
                url += f"&after={after}"

            r    = requests.get(url, timeout=15)
            data = r.json().get("data", [])
            if not data:
                break

            all_data.extend(data)
            fetched += len(data)
            after    = data[-1][0]
            time.sleep(0.3)

        if not all_data:
            logger.error("[Backtest] Нет данных от OKX")
            return pd.DataFrame()

        df = pd.DataFrame(
            all_data,
            columns=['ts', 'Open', 'High', 'Low', 'Close',
                     'Volume', 'VolCcy', 'VolCcyQuote', 'Confirm']
        )
        df[['Open', 'High', 'Low', 'Close', 'Volume']] = \
            df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        df['ts'] = pd.to_datetime(df['ts'].astype(float), unit='ms')
        df.set_index('ts', inplace=True)
        df = df.sort_index()
        logger.info(f"[Backtest] ✅ Загружено {len(df)} свечей")
        return df

    except Exception as e:
        logger.error(f"[Backtest] ❌ Ошибка загрузки: {e}")
        return pd.DataFrame()


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df['Close']
    high  = df['High']
    low   = df['Low']
    vol   = df['Volume']

    df = df.copy()
    df['Hour']      = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek

    for p in [7, 14, 21]:
        diff  = close.diff()
        g     = diff.clip(lower=0)
        l     = -diff.clip(upper=0)
        avg_g = g.ewm(com=p-1, min_periods=p).mean()
        avg_l = l.ewm(com=p-1, min_periods=p).mean()
        df[f'RSI_{p}'] = 100 - (100 / (1 + avg_g / (avg_l + 1e-9)))

    ema12              = close.ewm(span=12, adjust=False).mean()
    ema26              = close.ewm(span=26, adjust=False).mean()
    df['MACD']         = ema12 - ema26
    df['MACD_signal']  = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist']    = df['MACD'] - df['MACD_signal']

    tr   = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr14         = tr.ewm(com=13, min_periods=14).mean()
    atr50         = tr.ewm(com=49, min_periods=50).mean()
    df['ATR']     = atr14
    df['ATR_pct'] = (atr14 / (close + 1e-9)) * 100
    df['ATR_norm']= atr14 / (close + 1e-9)
    df['ATR_ratio']= atr14 / (atr50 + 1e-9)

    sma20         = close.rolling(20).mean()
    std20         = close.rolling(20).std()
    bb_upper      = sma20 + 2*std20
    bb_lower      = sma20 - 2*std20
    df['BB_pos']  = (close - bb_lower) / (4*std20 + 1e-9)
    df['BB_width']= (bb_upper - bb_lower) / (sma20 + 1e-9)

    ema20  = close.ewm(span=20).mean()
    ema50  = close.ewm(span=50).mean()
    ema100 = close.ewm(span=100).mean()
    df['EMA_ratio_20_50']  = ema20 / (ema50 + 1e-9)
    df['EMA_ratio_20_100'] = ema20 / (ema100 + 1e-9)
    df['EMA_ratio']        = df['EMA_ratio_20_50']

    vol_sma20      = vol.rolling(20).mean()
    df['Vol_ratio']= vol / (vol_sma20 + 1e-9)

    obv            = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    obv_sma20      = obv.rolling(20).mean()
    df['OBV_norm'] = (obv - obv_sma20) / (obv.rolling(20).std() + 1e-9)

    tp             = (high + low + close) / 3
    mf             = tp * vol
    pos_mf         = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_mf         = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
    df['MFI_14']   = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-9)))

    rsi14           = df['RSI_14']
    stoch_min       = rsi14.rolling(14).min()
    stoch_max       = rsi14.rolling(14).max()
    stoch_k         = (rsi14 - stoch_min) / (stoch_max - stoch_min + 1e-9) * 100
    df['StochRSI_K']= stoch_k
    df['StochRSI_D']= stoch_k.rolling(3).mean()

    hw14            = high.rolling(14).max()
    lw14            = low.rolling(14).min()
    df['WilliamsR'] = (hw14 - close) / (hw14 - lw14 + 1e-9) * -100

    df['ZScore_20'] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-9)
    df['ZScore_50'] = (close - close.rolling(50).mean()) / (close.rolling(50).std() + 1e-9)

    up   = high.diff()
    down = -low.diff()
    pdm  = up.where((up > down) & (up > 0), 0)
    mdm  = down.where((down > up) & (down > 0), 0)
    pdi  = 100 * (pdm.ewm(alpha=1/14).mean() / (atr14 + 1e-9))
    mdi  = 100 * (mdm.ewm(alpha=1/14).mean() / (atr14 + 1e-9))
    dx   = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    df['ADX'] = dx.ewm(alpha=1/14).mean()

    df['Body_pct']   = (close - df['Open']).abs() / (df['Open'] + 1e-9) * 100
    df['Upper_wick'] = (high - df[['Close','Open']].max(axis=1)) / (df['Open'] + 1e-9) * 100
    df['Lower_wick'] = (df[['Close','Open']].min(axis=1) - low) / (df['Open'] + 1e-9) * 100
    df['Doji']       = ((df['Body_pct'] / (high - low + 1e-9)) < 0.1).astype(int)

    df['Momentum_10'] = close - close.shift(10)
    df['ROC_10']      = close.pct_change(10) * 100

    for h in [1, 4, 12, 24]:
        df[f'Return_{h}h'] = close.pct_change(h) * 100

    return df.dropna()


def _sharpe(returns: list, periods_per_year: int = 8760) -> float:
    if len(returns) < 2:
        return 0.0
    r = np.array(returns)
    if r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def run_backtest(
    symbol        = "TON/USDT",
    timeframe     = "1h",
    limit         = 3000,
    trade_pct     = 0.10,
    start_balance = 600.0
) -> dict:

    logger.info(f"[Backtest] 🔍 Старт v5.0: {limit} свечей...")

    okx_symbol = symbol.replace("/", "-")
    okx_tf     = timeframe.upper()

    df = fetch_history(okx_symbol, okx_tf, limit)
    if df.empty:
        return {"success": False, "error": "Нет данных"}

    df = _add_indicators(df)

    # Пробуем загрузить ML-модели
    buy_model  = None
    sell_model = None
    feature_cols = []

    if os.path.exists(MODEL_PATH_BUY_XGB) and os.path.exists(MODEL_FEATURES_PATH):
        try:
            buy_model  = joblib.load(MODEL_PATH_BUY_XGB)
            sell_model = joblib.load(MODEL_PATH_SELL_XGB) if os.path.exists(MODEL_PATH_SELL_XGB) else None
            with open(MODEL_FEATURES_PATH) as f:
                feature_cols = json.load(f)
            logger.info(f"[Backtest] ✅ ML-модели загружены ({len(feature_cols)} фичей)")
        except Exception as e:
            logger.warning(f"[Backtest] ML-модели недоступны: {e} → RSI fallback")

    use_ml = buy_model is not None and len(feature_cols) > 0

    balance      = start_balance
    trades       = []
    hourly_ret   = []
    in_trade     = False
    trade_signal = ""
    trade_open   = 0.0
    tp_price     = 0.0
    sl_price     = 0.0
    amount_usd   = 0.0

    # Убираем первые 10% (warmup для индикаторов если ML)
    start_idx = int(len(df) * 0.10) if use_ml else 1

    for i in range(start_idx, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]

        # Закрытие
        if in_trade:
            hit = None
            if trade_signal == "BUY":
                if row['High'] >= tp_price:
                    hit = "TP"
                elif row['Low'] <= sl_price:
                    hit = "SL"
            else:
                if row['Low'] <= tp_price:
                    hit = "TP"
                elif row['High'] >= sl_price:
                    hit = "SL"

            if hit:
                cp = tp_price if hit == "TP" else sl_price
                if trade_signal == "BUY":
                    pnl_pct = (cp - trade_open) / trade_open * 100
                else:
                    pnl_pct = (trade_open - cp) / trade_open * 100

                pnl_usd   = amount_usd * pnl_pct / 100
                balance  += pnl_usd
                hourly_ret.append(pnl_pct)

                trades.append({
                    "signal":  trade_signal,
                    "result":  "WIN" if hit == "TP" else "LOSS",
                    "pnl_pct": round(pnl_pct, 2),
                    "pnl_usd": round(pnl_usd, 2),
                })
                in_trade = False
            else:
                hourly_ret.append(0.0)

        # Открытие
        if not in_trade:
            signal = None
            conf   = 0.0

            if use_ml:
                # ML-сигнал
                row_data = {}
                for col in feature_cols:
                    row_data[col] = float(row[col]) if col in df.columns else 0.0
                X = np.array([[row_data[c] for c in feature_cols]], dtype=np.float32)

                p_buy  = float(buy_model.predict_proba(X)[0][1])
                p_sell = float(sell_model.predict_proba(X)[0][1]) if sell_model else 0.0

                if p_buy >= 0.58 and p_buy > p_sell:
                    signal = "BUY"
                    conf   = p_buy
                elif p_sell >= 0.58 and p_sell > p_buy:
                    signal = "SELL"
                    conf   = p_sell
            else:
                # Fallback: RSI + EMA
                if prev['RSI_14'] < 35 and row['RSI_14'] >= 35 and row['EMA_ratio_20_50'] > 1.0:
                    signal = "BUY"
                    conf   = 0.65
                elif prev['RSI_14'] > 65 and row['RSI_14'] <= 65 and row['EMA_ratio_20_50'] < 1.0:
                    signal = "SELL"
                    conf   = 0.65

            if signal:
                atr_val    = float(row['ATR'])
                raw_sl     = (atr_val * 1.5) / float(row['Close'])
                sl_pct_dyn = max(0.008, min(raw_sl, 0.04))
                tp_pct_dyn = sl_pct_dyn * 2.0

                # Больший размер при высокой уверенности
                size_mult  = 1.5 if conf >= 0.70 else 1.0
                amount_usd = balance * trade_pct * size_mult
                trade_open   = float(row['Close'])
                trade_signal = signal
                in_trade     = True

                if signal == "BUY":
                    tp_price = trade_open * (1 + tp_pct_dyn)
                    sl_price = trade_open * (1 - sl_pct_dyn)
                else:
                    tp_price = trade_open * (1 - tp_pct_dyn)
                    sl_price = trade_open * (1 + sl_pct_dyn)

    # Закрываем остаток
    if in_trade:
        last_close = float(df['Close'].iloc[-1])
        if trade_signal == "BUY":
            pnl_pct = (last_close - trade_open) / trade_open * 100
        else:
            pnl_pct = (trade_open - last_close) / trade_open * 100
        pnl_usd  = amount_usd * pnl_pct / 100
        balance += pnl_usd
        trades.append({
            "signal":  trade_signal,
            "result":  "WIN" if pnl_pct > 0 else "LOSS",
            "pnl_pct": round(pnl_pct, 2),
            "pnl_usd": round(pnl_usd, 2),
        })

    total   = len(trades)
    wins    = sum(1 for t in trades if t["result"] == "WIN")
    losses  = total - wins
    winrate = round(wins / total * 100, 1) if total > 0 else 0

    pnl_list   = [t["pnl_pct"] for t in trades]
    avg_pnl    = round(sum(pnl_list) / len(pnl_list), 2) if pnl_list else 0
    total_pnl  = round(balance - start_balance, 2)
    growth_pct = round(total_pnl / start_balance * 100, 2)

    # Максимальная просадка
    peak, max_dd, running_b = start_balance, 0.0, start_balance
    for t in trades:
        running_b += t["pnl_usd"]
        if running_b > peak:
            peak = running_b
        dd = (peak - running_b) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    sharpe = _sharpe([t["pnl_pct"] for t in trades])

    result = {
        "success":       True,
        "symbol":        symbol,
        "candles":       len(df),
        "mode":          "ML" if use_ml else "RSI_FALLBACK",
        "total_trades":  total,
        "wins":          wins,
        "losses":        losses,
        "winrate":       winrate,
        "avg_pnl":       avg_pnl,
        "total_pnl":     total_pnl,
        "growth_pct":    growth_pct,
        "max_drawdown":  round(max_dd, 2),
        "sharpe_ratio":  round(sharpe, 2),
        "final_balance": round(balance, 2),
        "start_balance": start_balance,
    }

    logger.info(
        f"[Backtest] ✅ [{result['mode']}] {total} сделок | "
        f"Winrate: {winrate}% | Рост: {growth_pct:+.2f}% | "
        f"Sharpe: {sharpe:.2f}"
    )
    return result


def format_backtest_message(r: dict) -> str:
    if not r.get("success"):
        return f"❌ Бэктест не удался: {r.get('error')}"

    emoji = "📈" if r["growth_pct"] >= 0 else "📉"
    mode  = "🤖 ML-модель" if r.get("mode") == "ML" else "📐 RSI fallback"
    return (
        f"🔬 <b>Бэктест v5.0</b> {emoji}\n\n"
        f"⚙️ Режим:         <b>{mode}</b>\n"
        f"📊 Свечей:        <b>{r['candles']}</b>\n"
        f"📋 Сделок:        <b>{r['total_trades']}</b>\n"
        f"✅ Побед:         <b>{r['wins']}</b>\n"
        f"❌ Поражений:     <b>{r['losses']}</b>\n"
        f"🎯 Winrate:       <b>{r['winrate']}%</b>\n\n"
        f"💰 Старт:         <b>${r['start_balance']:.2f}</b>\n"
        f"💰 Финиш:         <b>${r['final_balance']:.2f}</b>\n"
        f"📈 Рост:          <b>{r['growth_pct']:+.2f}%</b>\n"
        f"💵 P&L:           <b>${r['total_pnl']:+.2f}</b>\n\n"
        f"📊 Средний P&L:   <b>{r['avg_pnl']:+.2f}%</b>\n"
        f"📉 Макс просадка: <b>{r['max_drawdown']:.2f}%</b>\n"
        f"⚡ Sharpe ratio:  <b>{r['sharpe_ratio']:.2f}</b>"
    )