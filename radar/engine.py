"""Carga de series, cálculo de factores y puntuación por activo."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import sources, technical

log = logging.getLogger("radar")
FFILL_LIMIT = 100  # días máximos que se arrastra un dato semanal/mensual


# ================================================================ CARGA
@dataclass
class DataSet:
    raw: dict[str, pd.Series] = field(default_factory=dict)
    status: dict[str, dict] = field(default_factory=dict)
    ohlc: dict[str, pd.DataFrame] = field(default_factory=dict)


def load_ohlc(cfg: dict, ds: "DataSet", start: str, cache_dir: Path | None) -> None:
    """Velas diarias por activo para el análisis técnico (con caché y respaldo a solo cierres)."""
    for k, a in cfg["assets"].items():
        spec = a.get("ohlc")
        df, fuente = None, ""
        if spec:
            try:
                if spec["source"] == "coinbase":
                    df = sources.coinbase_ohlc(spec["id"], start)
                elif spec["source"] == "yahoo":
                    df = sources.yahoo_ohlc(spec["id"], start)
                fuente = f"{spec['source']}:{spec['id']}"
            except Exception as e:  # noqa: BLE001
                log.warning("Velas %s fallaron: %s", k, e)
        f = Path(cache_dir) / f"ohlc_{k}.csv.gz" if cache_dir else None
        if (df is None or df.empty) and f and f.exists():
            df, fuente = pd.read_csv(f, index_col=0, parse_dates=True), "caché"
        if (df is None or df.empty) and a["price"] in ds.raw:
            p = ds.raw[a["price"]]
            df = pd.DataFrame({"open": p, "high": p, "low": p, "close": p, "volume": 0.0})
            fuente = "solo cierres"
        if df is None or df.empty:
            continue
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
        df = df[df.index < today]  # solo velas cerradas
        ds.ohlc[k] = df
        ds.status[f"velas_{k}"] = {"ok": True, "fuente": fuente, "respaldo": fuente in ("caché", "solo cierres"),
                                   "ultimo": df.index[-1].strftime("%Y-%m-%d")}
        if f is not None and fuente not in ("caché",):
            f.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(f, float_format="%.6g")


def _fetch(spec: dict, start: str, cm_metrics: dict[str, list[str]]) -> pd.Series:
    src = spec["source"]
    if src == "fred":
        return sources.fred(spec["id"], start)
    if src == "yahoo":
        return sources.yahoo(spec["id"], start)
    if src == "coinmetrics":
        return sources.coinmetrics(spec["id"], spec["metric"], start, cm_metrics.get(spec["id"]))
    if src == "deribit_funding":
        return sources.deribit_funding(spec["id"], start)
    if src == "defillama_stables":
        return sources.defillama_stables(start)
    if src == "fng":
        return sources.fng(start)
    if src == "cftc_cot":
        return sources.cftc_cot(spec["id"], start)
    raise ValueError(f"fuente desconocida {src}")


def _derived(formula: str, raw: dict[str, pd.Series]) -> pd.Series:
    def daily(k):
        return raw[k].resample("1D").last().ffill(limit=FFILL_LIMIT)

    if formula == "net_liquidity":  # en miles de millones de $
        s = daily("walcl") / 1000 - daily("tga") / 1000 - daily("rrp").reindex(daily("walcl").index).ffill(limit=10).fillna(0)
        return s.dropna()
    if formula == "copper_gold":
        return (daily("copper") / daily("gold") * 1000).dropna()
    if formula == "gold_silver":
        return (daily("gold") / daily("silver")).dropna()
    raise ValueError(formula)


def _load_cache(cache_dir: Path | None) -> dict[str, pd.Series]:
    if not cache_dir:
        return {}
    f = Path(cache_dir) / "series.csv.gz"
    if not f.exists():
        return {}
    try:
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        return {c: df[c].dropna() for c in df.columns}
    except Exception as e:  # noqa: BLE001
        log.warning("No se pudo leer la caché: %s", e)
        return {}


def save_cache(ds: "DataSet", cache_dir: Path) -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({k: v for k, v in ds.raw.items()})
    df.to_csv(Path(cache_dir) / "series.csv.gz", float_format="%.6g")


def load_data(cfg: dict, cache_dir: Path | None = None) -> DataSet:
    years = cfg["settings"].get("historia_anios", 6)
    start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    ds = DataSet()
    specs = cfg["series"]
    # Agrupar métricas de Coin Metrics por activo -> una sola descarga
    cm: dict[str, list[str]] = {}
    for s in specs.values():
        for cand in (s, s.get("fallback") or {}):
            if cand.get("source") == "coinmetrics":
                cm.setdefault(cand["id"], []).append(cand["metric"])

    cached = _load_cache(cache_dir)
    fred_ids = sorted({c["id"] for sp in specs.values() for c in (sp, sp.get("fallback") or {})
                       if c.get("source") == "fred"})
    try:
        sources.fred_bulk(fred_ids, start)
    except Exception as e:  # noqa: BLE001 - si falla, se piden una a una
        log.warning("Descarga conjunta de FRED falló (%s); se intentará serie a serie", e)

    def get_one(key: str, spec: dict):
        tried = [spec] + ([dict(spec["fallback"])] if spec.get("fallback") else [])
        err = ""
        for n, cand in enumerate(tried):
            try:
                s = _fetch(cand, start, cm)
                if s.empty:
                    raise RuntimeError("serie vacía")
                return key, s, {"ok": True, "fuente": f"{cand['source']}:{cand.get('id', '')}",
                                "respaldo": n > 0, "ultimo": s.index[-1].strftime("%Y-%m-%d")}
            except Exception as e:  # noqa: BLE001
                log.warning("Serie %s (%s) falló: %s", key, cand["source"], e)
                err = str(e)[:200]
        if key in cached:  # último recurso: copia de la ejecución anterior
            s = cached[key]
            return key, s, {"ok": True, "fuente": "caché", "respaldo": True,
                            "ultimo": s.index[-1].strftime("%Y-%m-%d"), "error": err}
        return key, None, {"ok": False, "error": err}

    # Coin Metrics primero (comparte descarga entre métricas), el resto en paralelo
    todo = [(k, s) for k, s in specs.items() if s["source"] != "derived"]
    with ThreadPoolExecutor(max_workers=4) as ex:
        for key, s, st in ex.map(lambda kv: get_one(*kv), todo):
            if s is not None:
                ds.raw[key] = s
            ds.status[key] = st

    # Precio en vivo de BTC (Coin Metrics publica con ~1 día de retraso)
    if "btc" in ds.raw and specs["btc"].get("live") == "deribit":
        try:
            px = sources.deribit_index()
            today = pd.Timestamp.utcnow().tz_localize(None).normalize()
            s = ds.raw["btc"].copy()
            s.loc[today] = px
            ds.raw["btc"] = s.sort_index()
            ds.status["btc"]["ultimo"] = today.strftime("%Y-%m-%d") + " (vivo)"
        except Exception as e:  # noqa: BLE001
            log.warning("Precio vivo BTC falló: %s", e)

    for key, spec in specs.items():
        if spec["source"] != "derived":
            continue
        try:
            s = _derived(spec["formula"], ds.raw)
            ds.raw[key] = s
            ds.status[key] = {"ok": True, "fuente": "calculada", "respaldo": False,
                              "ultimo": s.index[-1].strftime("%Y-%m-%d")}
        except Exception as e:  # noqa: BLE001
            ds.status[key] = {"ok": False, "error": f"faltan datos base ({e})"}
    return ds


# ================================================================ FACTORES
def daily(s: pd.Series) -> pd.Series:
    idx = pd.date_range(s.index[0], pd.Timestamp.today().normalize(), freq="1D")
    return s.resample("1D").last().reindex(idx).ffill(limit=FFILL_LIMIT)


def trend_signal(price: pd.Series) -> pd.Series:
    """-2..+2 según precio>MA50, precio>MA200 y MA50>MA200 (sobre observaciones reales)."""
    ma50, ma200 = price.rolling(50).mean(), price.rolling(200).mean()
    cnt = (price > ma50).astype(int) + (price > ma200).astype(int) + (ma50 > ma200).astype(int)
    sig = (cnt * 4 / 3 - 2).where(ma200.notna())
    return daily(sig)


def zscore(x: pd.Series, window: int) -> pd.Series:
    m = x.rolling(window, min_periods=180).mean()
    sd = x.rolling(window, min_periods=180).std()
    return (x - m) / sd.replace(0, np.nan)


@dataclass
class FactorResult:
    spec: dict
    serie: str
    nombre: str
    unidad: str
    valor: pd.Series      # serie diaria del dato
    medida: pd.Series     # lo que se evalúa (cambio, nivel...)
    z: pd.Series          # normalizado
    senal: pd.Series      # -2..+2 ya con signo (positivo = bueno para el activo)


def compute_factor(spec: dict, ds: DataSet, cfg: dict) -> FactorResult | None:
    key = spec["serie"]
    if key not in ds.raw:
        return None
    W = cfg["settings"].get("ventana_z_dias", 730)
    raw = ds.raw[key]
    meta = cfg["series"][key]
    s = daily(raw)
    if spec.get("suavizado"):
        s = s.rolling(int(spec["suavizado"]), min_periods=1).mean()
    lag = int(spec.get("lag", 0))
    t = spec["transform"]
    d = int(spec.get("dias", 28))

    if t == "trend":
        z = trend_signal(raw)
        medida = z
    else:
        base = s.shift(lag) if lag else s
        if t == "chg":
            medida = base - base.shift(d)
        elif t == "pct":
            medida = (base / base.shift(d) - 1) * 100
        elif t == "yoy":
            medida = (base / base.shift(365) - 1) * 100
        elif t in ("level", "band"):
            medida = base
        else:
            raise ValueError(t)
        if t == "band":
            z = (medida - float(spec["centro"])) / float(spec["ancho"])
        else:
            z = zscore(medida, W)
    senal = (z * int(spec["signo"])).clip(-2, 2)
    return FactorResult(spec, key, spec.get("nombre", meta["name"]), meta.get("unit", ""), s, medida, z, senal)


@dataclass
class AssetResult:
    key: str
    cfg: dict
    price: pd.Series
    factors: list[FactorResult]
    score: pd.Series
    cobertura: pd.Series
    tecnico: "technical.ResultadoTecnico | None" = None


def compute_asset(key: str, acfg: dict, ds: DataSet, cfg: dict) -> AssetResult | None:
    if acfg["price"] not in ds.raw:
        log.error("Sin precio para %s", key)
        return None
    facs = [f for f in (compute_factor(sp, ds, cfg) for sp in acfg["factors"]) if f is not None]
    price = daily(ds.raw[acfg["price"]])
    idx = price.index
    S = pd.DataFrame({i: f.senal.reindex(idx) for i, f in enumerate(facs)})
    w = pd.Series({i: float(f.spec["peso"]) for i, f in enumerate(facs)})
    wtot = float(sum(float(sp["peso"]) for sp in acfg["factors"]))
    avail = S.notna().mul(w, axis=1).sum(axis=1)
    score = (S.fillna(0).mul(w, axis=1).sum(axis=1) / avail.replace(0, np.nan)) * 50
    cobertura = avail / wtot
    score = score.where(cobertura >= 0.6)
    tec = None
    if key in ds.ohlc and len(ds.ohlc[key]) > 300:
        try:
            tec = technical.analizar(ds.ohlc[key])
        except Exception as e:  # noqa: BLE001
            log.error("Análisis técnico de %s falló: %s", key, e)
    return AssetResult(key, acfg, price, facs, score.round(1), cobertura, tec)


def regime(score: float, cfg: dict) -> str:
    u = cfg["settings"].get("umbral_favorable", 25)
    if score is None or np.isnan(score):
        return "Sin datos"
    if score >= u:
        return "Favorable"
    if score <= -u:
        return "Adverso"
    return "Neutral"


def backtest(a: AssetResult, cfg: dict, horizon: int = 28) -> dict:
    """Rentabilidad media a `horizon` días según el régimen de la puntuación (orientativo)."""
    px = a.price
    fwd = (px.shift(-horizon) / px - 1) * 100
    df = pd.DataFrame({"s": a.score, "f": fwd}).dropna()
    if len(df) < 200:
        return {}
    u = cfg["settings"].get("umbral_favorable", 25)
    out = {"horizonte": horizon, "dias": int(len(df)), "desde": df.index[0].strftime("%Y-%m-%d")}
    for name, mask in (("Favorable", df.s >= u), ("Neutral", (df.s > -u) & (df.s < u)),
                       ("Adverso", df.s <= -u), ("Todos", df.s == df.s)):
        sub = df[mask]
        out[name] = {
            "n": int(len(sub)),
            "media": round(float(sub.f.mean()), 2) if len(sub) else None,
            "acierto": round(float((sub.f > 0).mean() * 100), 1) if len(sub) else None,
        }
    out["correlacion"] = round(float(df.s.corr(df.f)), 3)
    return out


def run(cfg: dict, cache_dir: Path | None = None) -> tuple[DataSet, dict[str, AssetResult]]:
    ds = load_data(cfg, cache_dir)
    years = cfg["settings"].get("historia_anios", 6)
    start = (pd.Timestamp.today() - pd.DateOffset(years=years + 2)).strftime("%Y-%m-%d")
    load_ohlc(cfg, ds, start, cache_dir)
    assets = {}
    for k, acfg in cfg["assets"].items():
        r = compute_asset(k, acfg, ds, cfg)
        if r is not None:
            assets[k] = r
    return ds, assets
