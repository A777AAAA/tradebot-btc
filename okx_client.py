import requests
import time
import logging
import random
from config import SYMBOL

logger = logging.getLogger(__name__)

OKX_REST = "https://www.okx.com/api/v5"
DEFAULT_INST = SYMBOL.replace("/", "-")

PROXIES_LIST = [
    "http://smmlbrex:gblito9kdke7@31.59.20.176:6754",
    "http://smmlbrex:gblito9kdke7@23.95.150.145:6114",
    "http://smmlbrex:gblito9kdke7@198.23.239.134:6540",
    "http://smmlbrex:gblito9kdke7@45.38.107.97:6014",
    "http://smmlbrex:gblito9kdke7@107.172.163.27:6543",
    "http://smmlbrex:gblito9kdke7@198.105.121.200:6462",
    "http://smmlbrex:gblito9kdke7@216.10.27.159:6837",
    "http://smmlbrex:gblito9kdke7@142.111.67.146:5611",
    "http://smmlbrex:gblito9kdke7@191.96.254.138:6185",
    "http://smmlbrex:gblito9kdke7@31.58.9.4:6077",
]

def _get_proxy():
    p = "http://smmlbrex:gblito9kdke7@31.59.20.176:6754"
    return {"http": p, "https": p}

def _get(url: str, params: dict = None, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15, proxies=_get_proxy())
            if r.status_code == 429:
                wait = 2 ** attempt * 5
                logger.warning(f"[OKX] Rate limit 429 — ждём {wait}с (попытка {attempt+1})")
                time.sleep(wait)
                continue
            if r.status_code != 200:
                logger.error(f"[OKX] HTTP {r.status_code}: {url}")
                return {}
            data = r.json()
            if data.get("code") != "0":
                logger.warning(f"[OKX] API error: {data.get('msg')} | url={url}")
                return {}
            return data
        except requests.exceptions.Timeout:
            logger.warning(f"[OKX] Timeout (попытка {attempt+1}): {url}")
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"[OKX] Ошибка: {e}")
            time.sleep(2 ** attempt)
    logger.error(f"[OKX] Все {retries} попытки неудачны: {url}")
    return {}

def get_ticker(inst_id: str = DEFAULT_INST) -> dict:
    data = _get(f"{OKX_REST}/market/ticker", {"instId": inst_id})
    items = data.get("data", [])
    return items[0] if items else {}

def get_candles(inst_id: str = DEFAULT_INST, bar: str = "1H",
                limit: int = 300, after: str = None) -> list:
    params = {"instId": inst_id, "bar": bar, "limit": limit}
    if after:
        params["after"] = after
    data = _get(f"{OKX_REST}/market/candles", params)
    return data.get("data", [])

def get_history_candles(inst_id: str = DEFAULT_INST, bar: str = "1H",
                        limit: int = 300, after: str = None) -> list:
    params = {"instId": inst_id, "bar": bar, "limit": limit}
    if after:
        params["after"] = after
    data = _get(f"{OKX_REST}/market/history-candles", params)
    return data.get("data", [])

def get_candles_multi(inst_id: str = DEFAULT_INST, bar: str = "1H",
                      total: int = 2000) -> list:
    all_data = []
    after = None
    max_req = (total // 300) + 2
    for _ in range(max_req):
        batch = get_history_candles(inst_id, bar, 300, after)
        if not batch:
            break
        all_data.extend(batch)
        if len(all_data) >= total:
            break
        after = batch[-1][0]
        time.sleep(0.3)
    return all_data[:total]

def get_orderbook(inst_id: str = DEFAULT_INST, depth: int = 10) -> dict:
    data = _get(f"{OKX_REST}/market/books", {"instId": inst_id, "sz": depth})
    items = data.get("data", [])
    return items[0] if items else {}

def get_funding_rate(inst_id_swap: str = None) -> dict:
    if inst_id_swap is None:
        inst_id_swap = DEFAULT_INST + "-SWAP"
    data = _get(f"{OKX_REST}/public/funding-rate", {"instId": inst_id_swap})
    items = data.get("data", [])
    return items[0] if items else {}

def candles_to_df(data: list):
    import pandas as pd
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=[
        'ts', 'Open', 'High', 'Low', 'Close', 'Volume',
        'VolCcy', 'VolCcyQuote', 'Confirm'
    ])
    df = df[['ts', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['ts'] = pd.to_datetime(df['ts'].astype(float), unit='ms')
    df.set_index('ts', inplace=True)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return df
