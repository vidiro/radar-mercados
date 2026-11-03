"""Descarga de datos desde fuentes públicas y gratuitas."""
from __future__ import annotations

import io
import logging
import threading
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

log = logging.getLogger("radar")

UA = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "*/*",
}
# Algunas webs (FRED) bloquean el User-Agent de navegador; solo Yahoo lo necesita.
PLAIN = {"User-Agent": "curl/8.5.0", "Accept": "*/*"}


def _get(url: str, params: dict | None = None, tries: int = 3, timeout: int = 30,
         headers: dict | None = None) -> requests.Response:
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=headers or PLAIN)
            if r.status_code == 200:
                return r
            last = RuntimeError(f"HTTP {r.status_code}")
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        except requests.RequestException as e:  # red caída, timeout...
            last = e
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"{url} -> {last}")


def _clean(s: pd.Series, name: str) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.name = name
    return s


# ---------------------------------------------------------------- FRED
_FRED_BULK: dict[str, pd.Series] = {}


def fred_bulk(ids: list[str], start: str) -> dict[str, pd.Series]:
    """Descarga muchas series de FRED en UNA petición (devuelve un zip con CSVs por frecuencia)."""
    import zipfile

    r = _get("https://fred.stlouisfed.org/graph/fredgraph.csv",
             {"id": ",".join(ids), "cosd": start}, timeout=90)
    out: dict[str, pd.Series] = {}
    if r.content[:2] == b"PK":
        z = zipfile.ZipFile(io.BytesIO(r.content))
        frames = [pd.read_csv(z.open(n)) for n in z.namelist() if n.endswith(".csv")]
    else:
        frames = [pd.read_csv(io.StringIO(r.text))]
    for df in frames:
        df = df.set_index(df.columns[0])
        for c in df.columns:
            if c in ids:
                out[c] = _clean(df[c], c)
    _FRED_BULK.update(out)
    return out


def fred(series_id: str, start: str) -> pd.Series:
    if series_id in _FRED_BULK:
        return _FRED_BULK[series_id]
    r = _get("https://fred.stlouisfed.org/graph/fredgraph.csv", {"id": series_id, "cosd": start}, timeout=60)
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "value"]
    return _clean(df.set_index("date")["value"], series_id)


# ---------------------------------------------------------------- Yahoo
def yahoo(ticker: str, start: str) -> pd.Series:
    p1 = int(pd.Timestamp(start).timestamp())
    p2 = int(time.time()) + 86400
    for host in ("query2", "query1"):
        try:
            r = _get(
                f"https://{host}.finance.yahoo.com/v8/finance/chart/{ticker}",
                {"period1": p1, "period2": p2, "interval": "1d", "includeAdjustedClose": "true"},
                tries=2, headers=UA,
            )
            res = r.json()["chart"]["result"][0]
            idx = pd.to_datetime(res["timestamp"], unit="s")
            close = res["indicators"]["quote"][0]["close"]
            return _clean(pd.Series(close, index=idx), ticker)
        except Exception as e:  # noqa: BLE001
            log.info("yahoo %s via %s falló: %s", ticker, host, e)
    # último intento: librería yfinance (gestiona cookies/crumb)
    import yfinance as yf  # import perezoso

    df = yf.download(ticker, start=start, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"yahoo sin datos para {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return _clean(close, ticker)


# ---------------------------------------------------------------- Coin Metrics (community, gratis)
_CM_CACHE: dict[tuple, pd.DataFrame] = {}
_CM_LOCK = threading.Lock()


def coinmetrics(asset: str, metric: str, start: str, all_metrics: list[str] | None = None) -> pd.Series:
    metrics = sorted(set(all_metrics or []) | {metric})
    key = (asset, tuple(metrics), start)
    with _CM_LOCK:
        return _cm_series(key, asset, metrics, metric, start)


def _cm_series(key, asset, metrics, metric, start) -> pd.Series:
    if key not in _CM_CACHE:
        rows, url = [], "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
        params = {
            "assets": asset,
            "metrics": ",".join(metrics),
            "frequency": "1d",
            "start_time": start,
            "page_size": 10000,
            "ignore_unsupported_errors": "true",
        }
        while url:
            r = _get(url, params)
            j = r.json()
            rows += j.get("data", [])
            url, params = j.get("next_page_url"), None
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"])
        _CM_CACHE[key] = df.set_index("time")
    df = _CM_CACHE[key]
    if metric not in df:
        raise RuntimeError(f"coinmetrics sin métrica {metric}")
    return _clean(df[metric], f"{asset}.{metric}")


# ---------------------------------------------------------------- Deribit
def deribit_index(instrument: str = "BTC-PERPETUAL") -> float:
    r = _get("https://www.deribit.com/api/v2/public/ticker", {"instrument_name": instrument})
    return float(r.json()["result"]["index_price"])


def deribit_funding(instrument: str, start: str) -> pd.Series:
    """Funding medio diario anualizado (%)."""
    end = datetime.now(timezone.utc)
    begin = max(pd.Timestamp(start, tz="UTC").to_pydatetime(), end - timedelta(days=760))
    rows, t = [], begin
    while t < end:
        t2 = min(t + timedelta(days=30), end)
        r = _get(
            "https://www.deribit.com/api/v2/public/get_funding_rate_history",
            {
                "instrument_name": instrument,
                "start_timestamp": int(t.timestamp() * 1000),
                "end_timestamp": int(t2.timestamp() * 1000),
            },
        )
        rows += r.json()["result"]
        t = t2
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    daily = df.set_index("date")["interest_8h"].resample("1D").mean() * 3 * 365 * 100
    return _clean(daily, "funding")


# ---------------------------------------------------------------- DefiLlama
def defillama_stables(start: str) -> pd.Series:
    r = _get("https://stablecoins.llama.fi/stablecoincharts/all")
    data = r.json()
    idx = pd.to_datetime([int(d["date"]) for d in data], unit="s")
    vals = [d.get("totalCirculatingUSD", {}).get("peggedUSD") for d in data]
    s = _clean(pd.Series(vals, index=idx), "stables")
    return s[s.index >= pd.Timestamp(start)]


# ---------------------------------------------------------------- Fear & Greed
def fng(start: str) -> pd.Series:
    r = _get("https://api.alternative.me/fng/", {"limit": 0, "format": "json"})
    data = r.json()["data"]
    idx = pd.to_datetime([int(d["timestamp"]) for d in data], unit="s")
    s = _clean(pd.Series([d["value"] for d in data], index=idx), "fng")
    return s[s.index >= pd.Timestamp(start)]


# ---------------------------------------------------------------- CFTC COT
def cftc_cot(code: str, start: str) -> pd.Series:
    """Posición neta de 'managed money' como % del interés abierto."""
    r = _get(
        "https://publicreporting.cftc.gov/resource/72hh-3qpy.json",
        {
            "cftc_contract_market_code": code,
            "$where": f"report_date_as_yyyy_mm_dd >= '{start}T00:00:00'",
            "$order": "report_date_as_yyyy_mm_dd ASC",
            "$limit": 5000,
        },
    )
    df = pd.DataFrame(r.json())
    for c in ("m_money_positions_long_all", "m_money_positions_short_all", "open_interest_all"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    net = (df["m_money_positions_long_all"] - df["m_money_positions_short_all"]) / df["open_interest_all"] * 100
    net.index = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    return _clean(net, f"cot_{code}")
