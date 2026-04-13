import os, re, time, json, logging, requests, subprocess
from datetime import datetime

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
TELEGRAM_TOKEN      = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "")
OPENROUTER_URL      = "https://openrouter.ai/api/v1/chat/completions"
MODEL               = "anthropic/claude-sonnet-4-6"

# Интервал планового запроса — только после переобучения (24ч)
ADVISOR_INTERVAL_HOURS = 24

# Границы безопасного изменения параметров
PARAM_LIMITS = {
    "MIN_CONFIDENCE":       (0.50, 0.72),
    "STRONG_SIGNAL":        (0.60, 0.85),
    "REGIME_ADX_THRESHOLD": (12.0, 30.0),
    "ATR_SL_MULT":          (1.0,  3.0),
    "ATR_TP_MULT":          (1.5,  5.0),
}

CONFIG_PATH = "/app/config.py"

_log_buffer        = []
_LOG_BUFFER_MAX    = 500
_last_advice_ts    = 0.0
_last_regime       = ""
_last_precision    = 0.0
_last_errors_count = 0
_trade_results     = []   # {"signal","regime","adx","hurst","result","pnl"}
_params_history    = []   # история изменений параметров

# ─────────────────────────────────────────
# Буфер логов и сделок
# ─────────────────────────────────────────

def add_log(line):
    global _log_buffer
    _log_buffer.append(line)
    if len(_log_buffer) > _LOG_BUFFER_MAX:
        _log_buffer = _log_buffer[-_LOG_BUFFER_MAX:]

def add_trade_result(signal, regime, adx, hurst, result, pnl):
    """Вызывать из paper_trader при закрытии сделки."""
    _trade_results.append({
        "signal": signal, "regime": regime,
        "adx": adx, "hurst": hurst,
        "result": result, "pnl": pnl,
        "ts": datetime.now().strftime("%d.%m %H:%M")
    })
    if len(_trade_results) > 200:
        _trade_results.pop(0)

def _get_logs(lines=200):
    return "\n".join(_log_buffer[-lines:])

# ─────────────────────────────────────────
# Парсинг метрик
# ─────────────────────────────────────────

def _parse_metrics(logs):
    m = {
        "signals_buy": 0, "signals_sell": 0, "signals_hold": 0,
        "last_p_buy": 0.0, "last_p_sell": 0.0,
        "last_regime": "UNKNOWN", "last_adx": 0.0,
        "last_hurst": 0.0, "last_threshold": 0.0,
        "errors": [], "last_precision_buy": 0.0,
        "last_sharpe": 0.0, "last_kelly": 0.0,
        "filters_blocked": []
    }
    for line in logs.split("\n"):
        if "Signal v8.0] BUY"  in line: m["signals_buy"]  += 1
        elif "Signal v8.0] SELL" in line: m["signals_sell"] += 1
        elif "Signal v8.0] HOLD" in line: m["signals_hold"] += 1
        if "Signal v8.0]" in line:
            for pat, key in [
                (r"p_buy=([\d.]+)%",   "last_p_buy"),
                (r"p_sell=([\d.]+)%",  "last_p_sell"),
                (r"ADX=([\d.]+)",      "last_adx"),
                (r"Hurst=([\d.]+)",    "last_hurst"),
                (r"thresh=([\d.]+)%",  "last_threshold"),
            ]:
                x = re.search(pat, line)
                if x: m[key] = float(x.group(1))
            x = re.search(r"Regime=(\w+)", line)
            if x: m["last_regime"] = x.group(1)
            x = re.search(r"filters=\[([^\]]+)\]", line)
            if x: m["filters_blocked"].append(x.group(1))
        if "BUY prec=" in line:
            x = re.search(r"BUY prec=([\d.]+)%", line)
            if x: m["last_precision_buy"] = float(x.group(1))
        if "sharpe=" in line:
            x = re.search(r"sharpe=([\d.]+)", line)
            if x: m["last_sharpe"] = float(x.group(1))
        if "Kelly=" in line:
            x = re.search(r"Kelly=([\d.]+)%", line)
            if x: m["last_kelly"] = float(x.group(1))
        if "ERROR" in line and len(m["errors"]) < 5:
            m["errors"].append(line.strip()[-150:])
    return m

def _trade_summary():
    if not _trade_results:
        return "Сделок пока нет"
    wins  = [t for t in _trade_results if t["result"] == "WIN"]
    loses = [t for t in _trade_results if t["result"] == "LOSS"]
    wr    = len(wins) / len(_trade_results) * 100 if _trade_results else 0
    # паттерны потерь
    loss_regimes = {}
    for t in loses:
        loss_regimes[t["regime"]] = loss_regimes.get(t["regime"], 0) + 1
    worst = sorted(loss_regimes.items(), key=lambda x: -x[1])
    lines = [f"Сделок: {len(_trade_results)} | WIN: {len(wins)} ({wr:.0f}%) | LOSS: {len(loses)}"]
    if worst:
        lines.append("Потери по режимам: " + ", ".join(f"{r}={c}" for r, c in worst[:3]))
    last5 = _trade_results[-5:]
    lines.append("Последние 5: " + " | ".join(
        f"{t['signal']} {t['result']} {t['pnl']:+.2f}%" for t in last5
    ))
    return "\n".join(lines)

# ─────────────────────────────────────────
# Промпт — запрашиваем JSON с параметрами
# ─────────────────────────────────────────

def _build_prompt(metrics, symbol="BTC/USDT"):
    total    = metrics["signals_buy"] + metrics["signals_sell"] + metrics["signals_hold"]
    buy_rate = (metrics["signals_buy"] / total * 100) if total > 0 else 0
    filters  = ", ".join(set(metrics["filters_blocked"][-5:])) if metrics["filters_blocked"] else "нет"
    errors   = "\n".join(metrics["errors"]) if metrics["errors"] else "нет"
    trades   = _trade_summary()

    if _params_history:
        history = "\n".join(f"{h['ts']} | {h['param']}={h['value']}" for h in _params_history[-10:])
    else:
        history = "Изменений ещё не было"
    return f"""Ты эксперт по алгоритмической торговле BTC. Анализируй метрики и давай точные рекомендации.

=== МЕТРИКИ БОТА {symbol} ===
Циклов: {total} | BUY: {metrics['signals_buy']} ({buy_rate:.0f}%) | SELL: {metrics['signals_sell']} | HOLD: {metrics['signals_hold']}
p_buy: {metrics['last_p_buy']:.1f}% | p_sell: {metrics['last_p_sell']:.1f}%
Режим: {metrics['last_regime']} | Hurst: {metrics['last_hurst']:.3f} | ADX: {metrics['last_adx']:.1f}
Порог: {metrics['last_threshold']:.1f}% | BUY precision: {metrics['last_precision_buy']:.1f}%
WF Sharpe: {metrics['last_sharpe']:.2f} | Kelly: {metrics['last_kelly']:.1f}%
Фильтры заблокированы: {filters}
Ошибки: {errors}

=== РЕЗУЛЬТАТЫ СДЕЛОК ===
{trades}

=== ИСТОРИЯ ПОСЛЕДНИХ ИЗМЕНЕНИЙ ===
{history}

=== ТЕКУЩИЕ ПАРАМЕТРЫ CONFIG ===
MIN_CONFIDENCE: {_read_config_param('MIN_CONFIDENCE', 0.52)}
STRONG_SIGNAL: {_read_config_param('STRONG_SIGNAL', 0.65)}
REGIME_ADX_THRESHOLD: {_read_config_param('REGIME_ADX_THRESHOLD', 18.0)}
ATR_SL_MULT: {_read_config_param('ATR_SL_MULT', 1.5)}
ATR_TP_MULT: {_read_config_param('ATR_TP_MULT', 3.0)}

{_memory_context()}

=== ЗАДАЧА ===
1. Дай 2-3 текстовых вывода что не так и почему
2. В конце ОБЯЗАТЕЛЬНО верни JSON блок с новыми значениями параметров.
   Меняй только те параметры которые реально нужно изменить исходя из данных.
   Если параметр менять не нужно — не включай его в JSON.

Формат JSON (строго, без пояснений внутри):
```json
{{
  "params": {{
    "MIN_CONFIDENCE": 0.55,
    "REGIME_ADX_THRESHOLD": 20.0
  }},
  "reason": "Краткое обоснование изменений одной строкой"
}}
```

Границы допустимых значений:
MIN_CONFIDENCE: 0.50-0.72
STRONG_SIGNAL: 0.60-0.85
REGIME_ADX_THRESHOLD: 12.0-30.0
ATR_SL_MULT: 1.0-3.0
ATR_TP_MULT: 1.5-5.0"""

def _read_config_param(name, default):
    try:
        with open(CONFIG_PATH) as f:
            for line in f:
                if line.strip().startswith(name):
                    x = re.search(r"=\s*([\d.]+)", line)
                    if x: return float(x.group(1))
    except Exception:
        pass
    return default

# ─────────────────────────────────────────
# Claude API
# ─────────────────────────────────────────

def _ask_claude(prompt):
    if not OPENROUTER_API_KEY:
        return "Нет ключа", {}
    try:
        r = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://tradebot.local",
                "X-Title": "TradeBot"
            },
            json={
                "model": MODEL, "max_tokens": 700,
                "stream": False,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=60
        )
        if r.status_code != 200:
            return f"Ошибка {r.status_code}", {}
        text = r.json()["choices"][0]["message"]["content"].strip()
        params = _parse_json_params(text)
        return text, params
    except Exception as e:
        return f"Ошибка: {e}", {}

def _parse_json_params(text):
    try:
        x = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not x:
            x = re.search(r'(\{"params".*?\})', text, re.DOTALL)
        if not x:
            return {}
        data = json.loads(x.group(1))
        return data.get("params", {})
    except Exception:
        return {}

# ─────────────────────────────────────────
# Автоприменение параметров
# ─────────────────────────────────────────

def _apply_params(params):
    if not params:
        return []
    applied = []
    try:
        with open(CONFIG_PATH) as f:
            content = f.read()
        new_content = content
        for name, value in params.items():
            if name not in PARAM_LIMITS:
                logger.warning(f"[Advisor] Параметр {name} не в белом списке — пропущен")
                continue
            lo, hi = PARAM_LIMITS[name]
            value = float(value)
            if not (lo <= value <= hi):
                logger.warning(f"[Advisor] {name}={value} вне границ [{lo},{hi}] — пропущен")
                continue
            # Заменяем строку вида: NAME = X.XX или NAME  = X.XX
            pattern = rf"^({re.escape(name)}\s*=\s*)[\d.]+"
            replacement = rf"\g<1>{value}"
            new_content, n = re.subn(pattern, replacement, new_content, flags=re.MULTILINE)
            if n > 0:
                applied.append(f"{name}={value}")
                logger.info(f"[Advisor] ✅ Применено: {name} = {value}")
                _params_history.append({"ts": datetime.now().strftime("%d.%m %H:%M"), "param": name, "value": value})
            else:
                logger.warning(f"[Advisor] Не найдено в config: {name}")
        if applied:
            with open(CONFIG_PATH, "w") as f:
                f.write(new_content)
    except Exception as e:
        logger.error(f"[Advisor] Ошибка применения параметров: {e}")
    return applied

# ─────────────────────────────────────────
# Telegram
# ─────────────────────────────────────────

def _send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    # Telegram лимит 4096 символов
    if len(text) > 4000:
        text = text[:4000] + "\n...[обрезано]"
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10
        )
    except Exception as e:
        logger.error(f"[Advisor] Telegram: {e}")

# ─────────────────────────────────────────
# Основная функция
# ─────────────────────────────────────────

def run_advisor(symbol="BTC/USDT", force=False):
    global _last_advice_ts, _last_regime, _last_precision, _last_errors_count

    now     = time.time()
    logs    = _get_logs(200)
    metrics = _parse_metrics(logs)

    # Триггеры
    regime_changed   = metrics["last_regime"] != _last_regime and _last_regime != ""
    precision_dropped = metrics["last_precision_buy"] < 50 and _last_precision >= 50
    many_holds       = metrics["signals_hold"] > 15 and metrics["signals_buy"] == 0
    new_errors       = len(metrics["errors"]) > _last_errors_count
    timer_trigger    = (now - _last_advice_ts) >= ADVISOR_INTERVAL_HOURS * 3600

    trigger_reason = ""
    if regime_changed:    trigger_reason = f"Режим: {_last_regime}→{metrics['last_regime']}"
    elif precision_dropped: trigger_reason = f"Precision упал: {metrics['last_precision_buy']:.1f}%"
    elif many_holds:      trigger_reason = f"Много HOLD: {metrics['signals_hold']}"
    elif new_errors:      trigger_reason = f"Новые ошибки: {len(metrics['errors'])}"
    elif timer_trigger:   trigger_reason = "Плановый (24ч)"

    if not force and not trigger_reason:
        return ""

    logger.info(f"[Advisor] Триггер: {trigger_reason}")

    advice, params = _ask_claude(_build_prompt(metrics, symbol))

    # Применяем параметры
    applied = _apply_params(params)

    _last_advice_ts    = now
    _last_regime       = metrics["last_regime"]
    _last_precision    = metrics["last_precision_buy"]
    _last_errors_count = len(metrics["errors"])

    ts = datetime.now().strftime("%d.%m %H:%M")

    # Убираем JSON блок из текста для Telegram
    advice_clean = re.sub(r"```json.*?```", "", advice, flags=re.DOTALL).strip()

    # Формируем сообщение
    applied_str = ""
    if applied:
        applied_str = "\n\n⚙️ <b>Применено автоматически:</b>\n" + "\n".join(f"  • {a}" for a in applied)
    else:
        applied_str = "\n\n⚙️ Параметры без изменений"

    _send_telegram(
        f"🤖 <b>Claude Advisor — {symbol}</b> [{ts}]\n"
        f"<i>{trigger_reason}</i>\n\n"
        f"{advice_clean}"
        f"{applied_str}\n\n"
        f"<i>BUY={metrics['signals_buy']} SELL={metrics['signals_sell']} HOLD={metrics['signals_hold']}</i>"
    )
    logger.info(f"[Advisor] Отправлено в Telegram | Применено: {applied}")
    _save_memory({
        "ts": datetime.now().strftime("%d.%m %H:%M"),
        "ts_unix": time.time(),
        "trigger": trigger_reason,
        "regime": metrics["last_regime"],
        "precision": metrics["last_precision_buy"],
        "sharpe": metrics["last_sharpe"],
        "applied": applied,
    })
    if applied:
        logger.info("[Advisor] Перезапуск контейнера для применения параметров...")
        subprocess.Popen(["sh", "-c", "sleep 5 && docker restart tradebot_new"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return advice


# ─────────────────────────────────────────
# Тест
# ─────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    key = os.getenv("OPENROUTER_API_KEY")
    print("Тест API...")
    r = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": MODEL, "max_tokens": 80, "stream": False,
              "messages": [{"role": "user", "content": "Подтверди готовность как советник BTC бота одним предложением."}]},
        timeout=45
    )
    print("Claude:", r.json()["choices"][0]["message"]["content"])

    add_log("Signal v8.0] BUY | p_buy=65.0% | Regime=TREND | ADX=28.5 | Hurst=0.62 | thresh=53.0% | filters=[]")
    add_log("BUY prec=55.3% | sharpe=3.91 | Kelly=2.6%")
    add_trade_result("BUY", "TREND", 28.5, 0.62, "WIN", 1.8)
    add_trade_result("BUY", "RANDOM", 14.0, 0.48, "LOSS", -1.2)

    result = run_advisor(force=True)
    print("\n=== РЕКОМЕНДАЦИЯ ===\n", result)

# ─────────────────────────────────────────
# ПАМЯТЬ — сохранение истории на диск
# ─────────────────────────────────────────
MEMORY_FILE = "/app/data/advisor_memory.json"
MEMORY_DAYS = 3

def _load_memory() -> list:
    try:
        with open(MEMORY_FILE) as f:
            data = json.load(f)
        cutoff = time.time() - MEMORY_DAYS * 86400
        return [r for r in data if r.get("ts_unix", 0) > cutoff]
    except Exception:
        return []

def _save_memory(record: dict):
    try:
        import os
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
        history = _load_memory()
        history.append(record)
        history = history[-50:]
        with open(MEMORY_FILE, "w") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[Memory] Ошибка сохранения: {e}")

def _memory_context() -> str:
    history = _load_memory()
    if not history:
        return "История решений: пусто (первый запуск)"
    lines = ["=== ИСТОРИЯ РЕШЕНИЙ (последние 3 дня) ==="]
    for r in history[-5:]:
        lines.append(
            f"[{r.get('ts','')}] Триггер: {r.get('trigger','')} | "
            f"Применено: {r.get('applied',[])} | "
            f"Режим: {r.get('regime','')} | Precision: {r.get('precision',0):.1f}%"
        )
    return "\n".join(lines)
