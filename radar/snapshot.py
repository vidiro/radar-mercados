"""Convierte los resultados del motor en un 'snapshot' JSON (para el panel y Telegram)."""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .engine import AssetResult, DataSet, backtest, regime

HIST_DAYS = 365 * 4


def _f(x, nd=4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(x) or math.isinf(x)) else round(x, nd)


def _last(s: pd.Series):
    s = s.dropna()
    return (s.index[-1], s.iloc[-1]) if len(s) else (None, None)


def _chg(s: pd.Series, days: int):
    s = s.dropna()
    if len(s) < 2:
        return None
    ref = s[s.index <= s.index[-1] - pd.Timedelta(days=days)]
    if ref.empty or ref.iloc[-1] == 0:
        return None
    return _f((s.iloc[-1] / ref.iloc[-1] - 1) * 100, 2)


def medida_txt(f) -> str:
    txt = _medida_txt(f)
    return txt.replace(",", "\x00").replace(".", ",").replace("\x00", ".")  # formato español


def _medida_txt(f) -> str:
    t, d, u = f.spec["transform"], f.spec.get("dias", 28), f.unidad
    _, m = _last(f.medida)
    if m is None:
        return "sin dato"
    if t == "trend":
        n = int(round((m + 2) * 3 / 4))
        return f"{n}/3 señales alcistas"
    if t == "chg":
        unit = "pb" if u == "%" else u
        val = m * 100 if u == "%" else m
        return f"{val:+,.0f} {unit} en {d} días" if abs(val) >= 10 else f"{val:+.2f} {unit} en {d} días"
    if t == "pct":
        return f"{m:+.1f}% en {d} días"
    if t == "yoy":
        lag = f.spec.get("lag")
        return f"{m:+.1f}% interanual" + (f" (hace {lag} días)" if lag else "")
    if abs(m) >= 1000:
        return f"{m:,.0f} {u}"
    return f"{m:.2f} {u}"


def factor_ic(f, price: pd.Series, horizon: int = 28) -> tuple[float | None, float | None]:
    fwd = price.shift(-horizon) / price - 1
    s = f.senal.reindex(price.index)
    ic_all = s.corr(fwd)
    ic_2y = s.iloc[-730:].corr(fwd.iloc[-730:])
    return _f(ic_all, 3), _f(ic_2y, 3)


def _series_payload(s: pd.Series, days: int = HIST_DAYS, step: int = 1) -> dict:
    s = s.dropna()
    s = s[s.index >= s.index[-1] - pd.Timedelta(days=days)] if len(s) else s
    if step > 1:
        s = s.iloc[::-step][::-1]
    return {"t": [d.strftime("%Y-%m-%d") for d in s.index], "v": [_f(v, 6) for v in s.values]}


def build(ds: DataSet, assets: dict[str, AssetResult], cfg: dict, state: dict) -> dict:
    out = {"generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "umbral": cfg["settings"].get("umbral_favorable", 25),
           "umbral_fuerte": cfg["settings"].get("umbral_fuerte", 50),
           "activos": {}, "macro": {}, "estado_datos": ds.status,
           "alertas": state.get("historial", [])[-40:]}
    for k, a in assets.items():
        price = a.price.dropna()
        t_last, p_last = _last(price)
        _, sc = _last(a.score)
        sc7 = a.score.dropna()
        sc7 = sc7[sc7.index <= sc7.index[-1] - pd.Timedelta(days=7)].iloc[-1] if len(sc7) > 8 else None
        avail_w = sum(float(f.spec["peso"]) for f in a.factors if f.senal.dropna().size and float(f.spec["peso"]) > 0)
        facs = []
        for f in a.factors:
            _, sig = _last(f.senal)
            _, z = _last(f.z)
            _, val = _last(f.valor)
            w = float(f.spec["peso"])
            ic, ic2 = factor_ic(f, a.price)
            facs.append({
                "serie": f.serie, "nombre": f.nombre, "grupo": f.spec.get("grupo", ""),
                "porque": f.spec.get("porque", ""), "unidad": f.unidad, "transform": f.spec["transform"],
                "valor": _f(val, 4), "lectura": medida_txt(f), "z": _f(z, 2), "senal": _f(sig, 2),
                "peso": w, "contexto": w == 0,
                "contrib": _f(50 * w * sig / avail_w, 1) if (sig is not None and avail_w and w) else 0,
                "ic": ic, "ic2y": ic2,
                "hist": _series_payload(f.senal.reindex(a.price.index), 730, 3),
            })
        ma200 = a.price.dropna().rolling(200).mean()
        out["activos"][k] = {
            "nombre": a.cfg["name"], "simbolo": a.cfg.get("symbol", k.upper()), "color": a.cfg.get("color"),
            "precio": _f(p_last, 2), "fecha": t_last.strftime("%Y-%m-%d") if t_last is not None else None,
            "cambio": {"1d": _chg(price, 1), "7d": _chg(price, 7), "30d": _chg(price, 30),
                       "90d": _chg(price, 90), "1a": _chg(price, 365)},
            "puntuacion": _f(sc, 1), "puntuacion_7d": _f(sc7, 1),
            "regimen": state.get("regimenes", {}).get(k) or regime(sc, cfg),
            "cobertura": _f(a.cobertura.iloc[-1], 2),
            "factores": facs, "validacion": backtest(a, cfg),
            "historia": {"precio": _series_payload(a.price), "puntuacion": _series_payload(a.score),
                         "ma200": _series_payload(ma200)},
        }
    for key, spec in cfg["series"].items():
        if spec.get("hidden") or key not in ds.raw:
            continue
        s = ds.raw[key].dropna()
        t, v = _last(s)
        prev = s[s.index <= t - pd.Timedelta(days=28)]
        out["macro"][key] = {
            "nombre": spec["name"], "unidad": spec.get("unit", ""), "valor": _f(v, 4),
            "fecha": t.strftime("%Y-%m-%d"),
            "cambio28": _f(v - prev.iloc[-1], 4) if len(prev) else None,
            "cambio28_pct": _f((v / prev.iloc[-1] - 1) * 100, 2) if len(prev) and prev.iloc[-1] else None,
            "hist": _series_payload(s.resample("1D").last().dropna(), HIST_DAYS, 2),
        }
    return out
