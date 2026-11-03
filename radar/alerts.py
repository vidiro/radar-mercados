"""Reglas de alerta con memoria (no repite el mismo aviso) y textos para Telegram."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from .engine import AssetResult

log = logging.getLogger("radar")
EMO = {"Favorable": "🟢", "Neutral": "⚪️", "Adverso": "🔴", "Sin datos": "❔"}
ICON = {"btc": "₿", "gold": "🥇"}


def fmt_num(x: float | None, nd: int = 0) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = f"{x:,.{nd}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_pct(x) -> str:
    return "—" if x is None else ("+" if x >= 0 else "") + fmt_num(x, 1) + "%"


def regime_hyst(score: float, prev: str | None, u: float, h: float = 5) -> str:
    """Régimen con histéresis: para salir de un régimen hay que retroceder `h` puntos."""
    if score is None or np.isnan(score):
        return prev or "Sin datos"
    if prev == "Favorable" and score >= u - h:
        return "Favorable"
    if prev == "Adverso" and score <= -u + h:
        return "Adverso"
    if score >= u:
        return "Favorable"
    if score <= -u:
        return "Adverso"
    return "Neutral"


def _factor(assets: dict[str, AssetResult], serie: str):
    for a in assets.values():
        for f in a.factors:
            if f.serie == serie:
                return f
    return None


def _last(s: pd.Series):
    s = s.dropna()
    return s.iloc[-1] if len(s) else None


class Alerter:
    def __init__(self, cfg: dict, state: dict, today: date):
        self.cfg, self.state, self.today = cfg, state, today
        self.rules = cfg.get("alerts", {})
        self.out: list[dict] = []
        state.setdefault("ultima_alerta", {})
        state.setdefault("regimenes", {})
        state.setdefault("previo", {})

    def on(self, rule: str) -> dict | None:
        r = self.rules.get(rule, {})
        return r if r.get("activa", True) else None

    def fire(self, key: str, text: str, cooldown_days: int = 0, nivel: str = "info"):
        last = self.state["ultima_alerta"].get(key)
        if last and cooldown_days:
            if self.today - date.fromisoformat(last) < timedelta(days=cooldown_days):
                return
        if last == self.today.isoformat() and not cooldown_days:
            return
        self.state["ultima_alerta"][key] = self.today.isoformat()
        self.out.append({"clave": key, "texto": text, "nivel": nivel,
                         "fecha": datetime.utcnow().isoformat(timespec="minutes") + "Z"})

    # ------------------------------------------------------------------
    def evaluate(self, assets: dict[str, AssetResult]) -> list[dict]:
        u = self.cfg["settings"].get("umbral_favorable", 25)
        strong = self.cfg["settings"].get("umbral_fuerte", 50)
        first_run = not self.state["previo"]

        for k, a in assets.items():
            name, ic = a.cfg["name"], ICON.get(k, "•")
            prev = self.state["previo"].get(k, {})
            score = _last(a.score)
            px = a.price.dropna()
            obs = px  # precio diario
            reg_prev = self.state["regimenes"].get(k)
            reg = regime_hyst(score, reg_prev, u)
            self.state["regimenes"][k] = reg

            if not first_run:
                if self.on("cambio_regimen") and reg_prev and reg != reg_prev:
                    self.fire(f"regimen:{k}:{reg}",
                              f"{ic} <b>{name}: cambio de régimen</b>\n{EMO.get(reg_prev)} {reg_prev} → "
                              f"{EMO.get(reg)} <b>{reg}</b> (puntuación {fmt_num(score, 0)})\n"
                              + self._drivers(a), nivel="alta")
                pscore = prev.get("score")
                if self.on("puntuacion_fuerte") and score is not None and pscore is not None:
                    if abs(score) >= strong > abs(pscore):
                        lado = "muy favorable" if score > 0 else "muy adverso"
                        self.fire(f"fuerte:{k}:{np.sign(score)}",
                                  f"{ic} <b>{name}: entorno {lado}</b>\nPuntuación {fmt_num(score, 0)} "
                                  f"(umbral ±{strong})\n" + self._drivers(a), cooldown_days=3, nivel="alta")

            # Cruce de la media de 200 sesiones
            ma = obs.rolling(200).mean()
            above = bool(obs.iloc[-1] > ma.iloc[-1]) if not np.isnan(ma.iloc[-1]) else None
            if (self.on("cruce_media_200") and not first_run and above is not None
                    and prev.get("sobre_ma200") is not None and above != prev.get("sobre_ma200")):
                txt = "recupera" if above else "pierde"
                self.fire(f"ma200:{k}:{above}",
                          f"{ic} <b>{name} {txt} la media de 200 días</b>\nPrecio {fmt_num(obs.iloc[-1], 0)} $ · "
                          f"media {fmt_num(ma.iloc[-1], 0)} $", cooldown_days=2)

            # Movimiento extremo del día
            r = self.on("movimiento_extremo")
            if r:
                rets = np.log(obs / obs.shift(1)).dropna()
                if len(rets) > 70:
                    sd = rets.iloc[-61:-1].std()
                    last = rets.iloc[-1]
                    if sd and abs(last) >= r.get("sigmas", 2.5) * sd:
                        self.fire(f"movimiento:{k}:{obs.index[-1].date()}",
                                  f"{ic} <b>{name}: movimiento fuera de lo normal</b>\n"
                                  f"{fmt_pct((np.exp(last) - 1) * 100)} en el día "
                                  f"({abs(last) / sd:.1f}× su volatilidad habitual)", nivel="alta")

            # Divergencia precio vs. puntuación
            r = self.on("divergencia")
            if r and score is not None:
                d = r.get("dias", 28)
                ref = obs[obs.index <= obs.index[-1] - pd.Timedelta(days=d)]
                if len(ref):
                    ch = (obs.iloc[-1] / ref.iloc[-1] - 1) * 100
                    if ch > 5 and score <= -u:
                        self.fire(f"diverg:{k}:bajista",
                                  f"{ic} <b>{name}: divergencia bajista</b>\nEl precio sube {fmt_pct(ch)} en {d} días "
                                  f"pero el entorno es adverso ({fmt_num(score, 0)}). Subida sin apoyo de fondo.",
                                  cooldown_days=7)
                    elif ch < -5 and score >= u:
                        self.fire(f"diverg:{k}:alcista",
                                  f"{ic} <b>{name}: divergencia alcista</b>\nEl precio cae {fmt_pct(ch)} en {d} días "
                                  f"pero el entorno es favorable ({fmt_num(score, 0)}). Posible oportunidad.",
                                  cooldown_days=7)

            self.state["previo"][k] = {"score": None if score is None else float(score), "sobre_ma200": above}

        # ----- Alertas macro (una vez, no por activo)
        self._z_alert("liquidez_giro", "liq_net", "💧 <b>Liquidez neta de la Fed</b>",
                      "entrando liquidez con fuerza (bueno para activos de riesgo)",
                      "drenaje de liquidez fuerte (viento en contra para BTC)", assets, 1)
        self._z_alert("tipo_real_brusco", "real10", "📉 <b>Tipo real a 10 años</b>",
                      "subida brusca (presión sobre el oro y el BTC)",
                      "caída brusca (impulso para el oro)", assets, 1)
        self._z_alert("dxy_brusco", "dxy", "💵 <b>Dólar (DXY)</b>",
                      "subida brusca del dólar (viento en contra para oro y BTC)",
                      "caída brusca del dólar (viento a favor)", assets, 1)

        r = self.on("funding_extremo")
        f = _factor(assets, "funding")
        if r and f is not None:
            v = _last(f.valor)
            if v is not None and v >= r.get("alto", 30):
                self.fire("funding:alto", f"⚠️ <b>Funding BTC muy alto: {fmt_num(v, 1)}% anual</b>\n"
                          "Exceso de apalancamiento alcista. Riesgo de barrida de largos.",
                          cooldown_days=r.get("enfriamiento_dias", 3), nivel="alta")
            elif v is not None and v <= r.get("bajo", -5):
                self.fire("funding:bajo", f"🧲 <b>Funding BTC negativo: {fmt_num(v, 1)}% anual</b>\n"
                          "Mercado cargado en cortos: combustible para un 'short squeeze'.",
                          cooldown_days=r.get("enfriamiento_dias", 3), nivel="alta")

        r = self.on("miedo_codicia")
        f = _factor(assets, "fng")
        if r and f is not None:
            v = _last(f.valor)
            if v is not None and v <= r.get("miedo", 15):
                self.fire("fng:miedo", f"😱 <b>Miedo extremo en cripto: {int(v)}/100</b>\n"
                          "Históricamente, zonas de capitulación y oportunidad (contrarian).",
                          cooldown_days=r.get("enfriamiento_dias", 3))
            elif v is not None and v >= r.get("codicia", 85):
                self.fire("fng:codicia", f"🤑 <b>Codicia extrema en cripto: {int(v)}/100</b>\n"
                          "Euforia: cuidado con perseguir el precio.",
                          cooldown_days=r.get("enfriamiento_dias", 3))

        hist = self.state.setdefault("historial", [])
        hist.extend(self.out)
        del hist[:-100]
        return self.out

    def _z_alert(self, rule, serie, title, up_txt, down_txt, assets, signo):
        r = self.on(rule)
        f = _factor(assets, serie)
        if not r or f is None:
            return
        z = _last(f.z)
        if z is None:
            return
        th = r.get("z", 2.0)
        if z >= th:
            self.fire(f"{rule}:up", f"{title}: {up_txt}\n{_lect(f)} (z={z:+.1f})",
                      cooldown_days=r.get("enfriamiento_dias", 5))
        elif z <= -th:
            self.fire(f"{rule}:down", f"{title}: {down_txt}\n{_lect(f)} (z={z:+.1f})",
                      cooldown_days=r.get("enfriamiento_dias", 5))

    @staticmethod
    def _drivers(a: AssetResult, n: int = 3) -> str:
        items = []
        for f in a.factors:
            s = _last(f.senal)
            if s is not None and float(f.spec["peso"]) > 0:
                items.append((s * float(f.spec["peso"]), f.nombre))
        items.sort()
        pos = [n_ for v, n_ in reversed(items) if v > 0.2][:n]
        neg = [n_ for v, n_ in items if v < -0.2][:n]
        txt = ""
        if pos:
            txt += "✅ A favor: " + ", ".join(pos) + "\n"
        if neg:
            txt += "❌ En contra: " + ", ".join(neg)
        return txt.strip()


def _lect(f) -> str:
    from .snapshot import medida_txt

    return medida_txt(f)


def summary_text(snap: dict, panel_url: str | None, titulo: str = "📡 <b>Radar de mercados</b>") -> str:
    lines = [titulo]
    for k, a in snap["activos"].items():
        lines.append(asset_text(k, a, short=True))
    if panel_url:
        lines.append(f'🔗 <a href="{panel_url}">Abrir panel</a>')
    return "\n\n".join(lines)


def asset_text(k: str, a: dict, short: bool = False) -> str:
    ic = ICON.get(k, "•")
    reg = a["regimen"]
    sc, sc7 = a.get("puntuacion"), a.get("puntuacion_7d")
    delta = f" ({'+' if sc - sc7 >= 0 else ''}{fmt_num(sc - sc7, 0)} en 7d)" if sc is not None and sc7 is not None else ""
    head = (f"{ic} <b>{a['nombre']}</b>  {fmt_num(a['precio'], 0)} $  "
            f"<i>{fmt_pct(a['cambio']['1d'])} 24h · {fmt_pct(a['cambio']['7d'])} 7d</i>\n"
            f"{EMO.get(reg, '')} <b>{reg}</b> · puntuación {fmt_num(sc, 0)}{delta}")
    facs = [f for f in a["factores"] if not f["contexto"] and f["senal"] is not None]
    facs.sort(key=lambda f: f["contrib"] or 0)
    pos = [f for f in reversed(facs) if (f["contrib"] or 0) > 1][: 3 if short else 5]
    neg = [f for f in facs if (f["contrib"] or 0) < -1][: 3 if short else 5]
    body = []
    if pos:
        body.append("✅ " + ("\n✅ ".join(f"{f['nombre']}: {f['lectura']}" for f in pos) if not short
                            else ", ".join(f["nombre"] for f in pos)))
    if neg:
        body.append("❌ " + ("\n❌ ".join(f"{f['nombre']}: {f['lectura']}" for f in neg) if not short
                            else ", ".join(f["nombre"] for f in neg)))
    if not short:
        ctx = [f for f in a["factores"] if f["contexto"]]
        if ctx:
            body.append("ℹ️ Contexto: " + " · ".join(f"{f['nombre']} {f['lectura']}" for f in ctx))
    return head + ("\n" + "\n".join(body) if body else "")
