"""Análisis técnico: detección de giros, agotamiento y rupturas, con validación histórica.

Las reglas se inspiran en los clásicos (Murphy, Edwards & Magee, Wyckoff), traducidas a
condiciones objetivas que se pueden calcular y medir. Cada señal se evalúa con el histórico
de cada activo: solo las que han funcionado pueden generar alertas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PIVOT_K = 3          # velas a cada lado para confirmar un máximo/mínimo
HORIZONTES = (5, 10, 20)
H_PRINCIPAL = 10

# ------------------------------------------------------------------ catálogo de señales
CATALOGO = {
    "div_rsi": {
        "nombre": "Divergencia RSI", "tipo": "giro",
        "idea": "El precio marca un nuevo extremo pero el impulso (RSI) no lo acompaña: la fuerza "
                "del movimiento se está agotando. Murphy la considera una de las señales de giro más fiables.",
    },
    "rsi_giro": {
        "nombre": "RSI sale de zona extrema", "tipo": "giro",
        "idea": "El RSI estaba en sobreventa (<30) o sobrecompra (>70) y vuelve a la zona normal: "
                "el exceso empieza a corregirse.",
    },
    "bb_reentrada": {
        "nombre": "Reentrada en bandas de Bollinger", "tipo": "giro",
        "idea": "El precio cerró fuera de la banda (movimiento de más de 2 desviaciones) y vuelve a "
                "cerrar dentro: el estiramiento extremo se ha frenado.",
    },
    "rebote_nivel": {
        "nombre": "Rechazo en soporte/resistencia", "tipo": "giro",
        "idea": "El precio toca un nivel donde antes giró y lo rechaza, cerrando lejos del extremo de la "
                "vela: los compradores (o vendedores) defienden la zona.",
    },
    "doble": {
        "nombre": "Doble suelo / doble techo", "tipo": "giro",
        "idea": "Dos intentos fallidos en la misma zona y ruptura de la 'línea de cuello' intermedia. "
                "Figura clásica de cambio de tendencia de Edwards & Magee.",
    },
    "climax": {
        "nombre": "Clímax de volumen (Wyckoff)", "tipo": "giro",
        "idea": "Tras una caída (o subida) fuerte aparece una vela enorme con volumen extremo y cierre "
                "alejado del extremo: suele marcar la capitulación de los vendedores (o la euforia final).",
    },
    "sobreextension": {
        "nombre": "Sobreextensión respecto a la media", "tipo": "giro",
        "idea": "El precio se ha alejado de su media de 50 días mucho más de lo habitual: estadísticamente "
                "tiende a volver hacia ella (agotamiento).",
    },
    "vela_giro": {
        "nombre": "Vela de giro (envolvente / martillo)", "tipo": "giro",
        "idea": "Después de varios días en una dirección aparece una vela que 'envuelve' a la anterior o "
                "una mecha larga de rechazo: cambio de control entre compradores y vendedores.",
    },
    "ruptura": {
        "nombre": "Ruptura de rango de 55 días", "tipo": "continuación",
        "idea": "El precio supera el máximo (o pierde el mínimo) de casi 3 meses con volumen: arranque "
                "o continuación de tendencia (regla de las 'tortugas').",
    },
    "squeeze": {
        "nombre": "Salida de compresión de volatilidad", "tipo": "continuación",
        "idea": "Las bandas de Bollinger estaban en su punto más estrecho en meses y el precio sale con "
                "fuerza: tras la calma suele venir un movimiento amplio.",
    },
    "cruce_medias": {
        "nombre": "Cruce de medias 50/200", "tipo": "continuación",
        "idea": "La media de 50 días cruza la de 200 ('cruce dorado' o 'cruce de la muerte'): cambio de "
                "tendencia de fondo, señal lenta pero vigilada por todo el mercado.",
    },
}


# ------------------------------------------------------------------ indicadores
def rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def pivots(s: pd.Series, k: int, mode: str) -> pd.Series:
    """True en las velas que son mínimo (mode='low') o máximo ('high') local con k velas a cada lado."""
    w = 2 * k + 1
    roll = s.rolling(w, center=True)
    ext = roll.min() if mode == "low" else roll.max()
    return (s == ext) & ext.notna()


@dataclass
class Evento:
    fecha: pd.Timestamp
    i: int
    tipo: str
    dir: int            # +1 alcista, -1 bajista
    precio: float
    detalle: str
    extra: dict = field(default_factory=dict)


def _fmt(x: float) -> str:
    s = f"{x:,.0f}" if abs(x) >= 100 else f"{x:,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


# ------------------------------------------------------------------ detectores
def detectar(df: pd.DataFrame) -> list[Evento]:
    df = df.copy()
    c, h, l, o, v = df.close, df.high, df.low, df.open, df.volume
    n = len(df)
    idx = df.index
    df["rsi"] = r = rsi(c)
    df["atr"] = a = atr(df)
    ma20, ma50, ma200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    sd20 = c.rolling(20).std()
    bbu, bbl = ma20 + 2 * sd20, ma20 - 2 * sd20
    bw = (bbu - bbl) / ma20
    vol_ok = v.fillna(0).gt(0).mean() > 0.8
    vma = v.rolling(50).mean()
    ohlc_ok = (h - l).gt(0).mean() > 0.8
    rets = np.log(c / c.shift(1))
    sig_d = rets.rolling(250, min_periods=60).std()
    ret20 = np.log(c / c.shift(20))
    rango = (h - l)
    pos_cierre = ((c - l) / rango.replace(0, np.nan)).fillna(0.5)
    ev: list[Evento] = []

    def add(i, tipo, d, detalle, **extra):
        ev.append(Evento(idx[i], i, tipo, d, float(c.iloc[i]), detalle, extra))

    # --- pivotes confirmados (sin mirar al futuro: se usan k velas después)
    pl = np.where(pivots(l, PIVOT_K, "low").values)[0]
    ph = np.where(pivots(h, PIVOT_K, "high").values)[0]

    # 1) Divergencias RSI entre dos pivotes consecutivos
    for arr, mode in ((pl, "low"), (ph, "high")):
        for j in range(1, len(arr)):
            p1, p2 = arr[j - 1], arr[j]
            if not (5 <= p2 - p1 <= 60):
                continue
            conf = p2 + PIVOT_K
            if conf >= n:
                continue
            if mode == "low":
                if l.iloc[p2] < l.iloc[p1] and r.iloc[p2] > r.iloc[p1] + 3 and r.iloc[p1] < 40:
                    add(conf, "div_rsi", 1, f"Nuevo mínimo en {_fmt(l.iloc[p2])} (antes {_fmt(l.iloc[p1])}) "
                        f"pero RSI más alto: {r.iloc[p2]:.0f} frente a {r.iloc[p1]:.0f}")
            else:
                if h.iloc[p2] > h.iloc[p1] and r.iloc[p2] < r.iloc[p1] - 3 and r.iloc[p1] > 60:
                    add(conf, "div_rsi", -1, f"Nuevo máximo en {_fmt(h.iloc[p2])} (antes {_fmt(h.iloc[p1])}) "
                        f"pero RSI más bajo: {r.iloc[p2]:.0f} frente a {r.iloc[p1]:.0f}")

    for i in range(201, n):
        # 2) RSI sale de extremos
        if r.iloc[i - 1] < 30 <= r.iloc[i]:
            add(i, "rsi_giro", 1, f"RSI sale de sobreventa ({r.iloc[i-1]:.0f} → {r.iloc[i]:.0f})")
        if r.iloc[i - 1] > 70 >= r.iloc[i]:
            add(i, "rsi_giro", -1, f"RSI sale de sobrecompra ({r.iloc[i-1]:.0f} → {r.iloc[i]:.0f})")
        # 3) Reentrada en Bollinger
        if c.iloc[i - 1] < bbl.iloc[i - 1] and c.iloc[i] > bbl.iloc[i]:
            add(i, "bb_reentrada", 1, f"Vuelve dentro de la banda inferior ({_fmt(bbl.iloc[i])})")
        if c.iloc[i - 1] > bbu.iloc[i - 1] and c.iloc[i] < bbu.iloc[i]:
            add(i, "bb_reentrada", -1, f"Vuelve dentro de la banda superior ({_fmt(bbu.iloc[i])})")
        # 7) Sobreextensión vs MA50 (umbral relativo a la volatilidad propia)
        if not math.isnan(sig_d.iloc[i]) and sig_d.iloc[i] > 0:
            ext = (c.iloc[i] / ma50.iloc[i] - 1)
            lim = 2.2 * sig_d.iloc[i] * math.sqrt(50) / 2
            prev = (c.iloc[i - 1] / ma50.iloc[i - 1] - 1)
            if ext < -lim and prev >= -lim:
                add(i, "sobreextension", 1, f"Precio {ext*100:+.1f}% bajo la media de 50 días (umbral {-lim*100:.1f}%)")
            if ext > lim and prev <= lim:
                add(i, "sobreextension", -1, f"Precio {ext*100:+.1f}% sobre la media de 50 días (umbral {lim*100:.1f}%)")
        # 9) Ruptura de 55 días
        hi55 = h.iloc[i - 55:i].max()
        lo55 = l.iloc[i - 55:i].min()
        vfac = (v.iloc[i] / vma.iloc[i]) if vol_ok and vma.iloc[i] else 1.0
        if c.iloc[i] > hi55 and c.iloc[i - 1] <= h.iloc[i - 56:i - 1].max() and (not vol_ok or vfac > 1.2):
            add(i, "ruptura", 1, f"Cierre en {_fmt(c.iloc[i])} supera el máximo de 55 días ({_fmt(hi55)})"
                + (f" con volumen {vfac:.1f}× la media" if vol_ok else ""))
        if c.iloc[i] < lo55 and c.iloc[i - 1] >= l.iloc[i - 56:i - 1].min() and (not vol_ok or vfac > 1.2):
            add(i, "ruptura", -1, f"Cierre en {_fmt(c.iloc[i])} pierde el mínimo de 55 días ({_fmt(lo55)})"
                + (f" con volumen {vfac:.1f}× la media" if vol_ok else ""))
        # 10) Squeeze
        if i > 180:
            q = bw.iloc[i - 180:i].quantile(0.1)
            if bw.iloc[i - 1] <= q:
                if c.iloc[i] > bbu.iloc[i]:
                    add(i, "squeeze", 1, "Bandas en mínimo de anchura de 6 meses y cierre por encima de la banda superior")
                elif c.iloc[i] < bbl.iloc[i]:
                    add(i, "squeeze", -1, "Bandas en mínimo de anchura de 6 meses y cierre por debajo de la banda inferior")
        # 11) Cruce de medias
        if ma50.iloc[i - 1] <= ma200.iloc[i - 1] and ma50.iloc[i] > ma200.iloc[i]:
            add(i, "cruce_medias", 1, "Cruce dorado: la media de 50 supera a la de 200")
        if ma50.iloc[i - 1] >= ma200.iloc[i - 1] and ma50.iloc[i] < ma200.iloc[i]:
            add(i, "cruce_medias", -1, "Cruce de la muerte: la media de 50 pierde la de 200")

        if not ohlc_ok:
            continue
        # 6) Clímax de volumen (Wyckoff)
        if vol_ok and vma.iloc[i] and not math.isnan(sig_d.iloc[i]):
            z20 = ret20.iloc[i - 1] / (sig_d.iloc[i] * math.sqrt(20))
            big = rango.iloc[i] > 1.8 * a.iloc[i - 1] and v.iloc[i] > 2.3 * vma.iloc[i]
            if big and z20 < -1.3 and pos_cierre.iloc[i] >= 0.45:
                add(i, "climax", 1, f"Clímax vendedor: volumen {v.iloc[i]/vma.iloc[i]:.1f}× la media, vela de "
                    f"{rango.iloc[i]/a.iloc[i-1]:.1f}× el ATR y cierre en la mitad alta tras caer {ret20.iloc[i-1]*100:.0f}% en 20 días")
            if big and z20 > 1.3 and pos_cierre.iloc[i] <= 0.55:
                add(i, "climax", -1, f"Clímax comprador: volumen {v.iloc[i]/vma.iloc[i]:.1f}× la media, vela de "
                    f"{rango.iloc[i]/a.iloc[i-1]:.1f}× el ATR y cierre en la mitad baja tras subir {ret20.iloc[i-1]*100:.0f}% en 20 días")
        # 8) Velas de giro tras un movimiento
        body = abs(c.iloc[i] - o.iloc[i])
        lower = min(c.iloc[i], o.iloc[i]) - l.iloc[i]
        upper = h.iloc[i] - max(c.iloc[i], o.iloc[i])
        down5 = c.iloc[i - 1] < c.iloc[i - 6] and c.iloc[i - 1] < ma20.iloc[i - 1]
        up5 = c.iloc[i - 1] > c.iloc[i - 6] and c.iloc[i - 1] > ma20.iloc[i - 1]
        po, pc_ = o.iloc[i - 1], c.iloc[i - 1]
        if down5 and pc_ < po and c.iloc[i] > o.iloc[i] and o.iloc[i] <= pc_ and c.iloc[i] >= po and body > 0.5 * a.iloc[i]:
            add(i, "vela_giro", 1, "Envolvente alcista tras varios días de caída")
        elif down5 and lower >= 2 * body and upper <= max(body, 0.1 * rango.iloc[i]) and rango.iloc[i] > 0.9 * a.iloc[i]:
            add(i, "vela_giro", 1, "Martillo: mecha inferior larga, rechazo de precios bajos")
        if up5 and pc_ > po and c.iloc[i] < o.iloc[i] and o.iloc[i] >= pc_ and c.iloc[i] <= po and body > 0.5 * a.iloc[i]:
            add(i, "vela_giro", -1, "Envolvente bajista tras varios días de subida")
        elif up5 and upper >= 2 * body and lower <= max(body, 0.1 * rango.iloc[i]) and rango.iloc[i] > 0.9 * a.iloc[i]:
            add(i, "vela_giro", -1, "Estrella fugaz: mecha superior larga, rechazo de precios altos")

    # 4) Rechazo en soportes/resistencias previos y 5) dobles suelos/techos
    lv_lo = [(p, l.iloc[p]) for p in pl]
    lv_hi = [(p, h.iloc[p]) for p in ph]
    for i in range(201, n):
        ai = a.iloc[i]
        if not ohlc_ok or math.isnan(ai):
            break
        sup = [(p, x) for p, x in lv_lo if 20 <= i - p <= 250 and p + PIVOT_K < i]
        res = [(p, x) for p, x in lv_hi if 20 <= i - p <= 250 and p + PIVOT_K < i]
        for p, x in sup[-12:]:
            if x - 1.0 * ai <= l.iloc[i] <= x + 0.3 * ai and c.iloc[i] > x and pos_cierre.iloc[i] >= 0.6 \
                    and c.iloc[i - 1] > x and c.iloc[i - 10] > c.iloc[i - 1]:
                add(i, "rebote_nivel", 1, f"Toca el soporte de {_fmt(x)} (giro del {idx[p]:%d/%m/%Y}) y cierra arriba",
                    nivel=float(x))
                break
        for p, x in res[-12:]:
            if x - 0.3 * ai <= h.iloc[i] <= x + 1.0 * ai and c.iloc[i] < x and pos_cierre.iloc[i] <= 0.4 \
                    and c.iloc[i - 1] < x and c.iloc[i - 10] < c.iloc[i - 1]:
                add(i, "rebote_nivel", -1, f"Toca la resistencia de {_fmt(x)} (giro del {idx[p]:%d/%m/%Y}) y cierra abajo",
                    nivel=float(x))
                break
    for arr, mode in ((pl, "low"), (ph, "high")):
        for j in range(1, len(arr)):
            p1, p2 = arr[j - 1], arr[j]
            if not (10 <= p2 - p1 <= 80) or p2 + PIVOT_K >= n:
                continue
            ai = a.iloc[p2]
            if mode == "low":
                if abs(l.iloc[p2] - l.iloc[p1]) > 1.0 * ai:
                    continue
                neck = h.iloc[p1:p2 + 1].max()
                if neck - max(l.iloc[p1], l.iloc[p2]) < 2.5 * ai:
                    continue
                for i in range(p2 + PIVOT_K, min(n, p2 + 40)):
                    if l.iloc[i] < min(l.iloc[p1], l.iloc[p2]) - 0.5 * ai:
                        break
                    if c.iloc[i] > neck:
                        add(i, "doble", 1, f"Doble suelo en {_fmt(l.iloc[p1])} / {_fmt(l.iloc[p2])} y ruptura del "
                            f"cuello en {_fmt(neck)}", nivel=float(neck))
                        break
            else:
                if abs(h.iloc[p2] - h.iloc[p1]) > 1.0 * ai:
                    continue
                neck = l.iloc[p1:p2 + 1].min()
                if min(h.iloc[p1], h.iloc[p2]) - neck < 2.5 * ai:
                    continue
                for i in range(p2 + PIVOT_K, min(n, p2 + 40)):
                    if h.iloc[i] > max(h.iloc[p1], h.iloc[p2]) + 0.5 * ai:
                        break
                    if c.iloc[i] < neck:
                        add(i, "doble", -1, f"Doble techo en {_fmt(h.iloc[p1])} / {_fmt(h.iloc[p2])} y pérdida del "
                            f"cuello en {_fmt(neck)}", nivel=float(neck))
                        break

    # deduplicar: misma señal y dirección en < 8 velas -> nos quedamos con la primera
    ev.sort(key=lambda e: e.i)
    out, last = [], {}
    for e in ev:
        k = (e.tipo, e.dir)
        if k in last and e.i - last[k] < 8:
            continue
        last[k] = e.i
        out.append(e)
    return out


# ------------------------------------------------------------------ validación
def evaluar(df: pd.DataFrame, eventos: list[Evento]) -> dict:
    c = df.close
    fw = {hz: np.log(c.shift(-hz) / c) for hz in HORIZONTES}
    base = {hz: float(fw[hz].mean()) for hz in HORIZONTES}
    stats = {}
    for tipo in CATALOGO:
        for d in (1, -1):
            es = [e for e in eventos if e.tipo == tipo and e.dir == d]
            res = {"n": len(es)}
            for hz in HORIZONTES:
                vals = [fw[hz].iloc[e.i] * d for e in es if not math.isnan(fw[hz].iloc[e.i])]
                if vals:
                    res[f"acierto{hz}"] = round(100 * float(np.mean([x > 0 for x in vals])), 0)
                    res[f"media{hz}"] = round(100 * (math.exp(float(np.mean(vals))) - 1), 2)
                    res[f"ventaja{hz}"] = round(100 * (float(np.mean(vals)) - base[hz] * d), 2)
                    res[f"n{hz}"] = len(vals)
            nh = res.get(f"n{H_PRINCIPAL}", 0)
            hit = res.get(f"acierto{H_PRINCIPAL}", 0)
            edge = res.get(f"ventaja{H_PRINCIPAL}", -1)
            if nh < 8:
                grado = "pocos casos"
            elif hit >= 60 and edge > 0:
                grado = "fiable"
            elif hit >= 53 and edge > 0:
                grado = "media"
            else:
                grado = "no fiable"
            res["grado"] = grado
            stats[f"{tipo}:{d}"] = res
    return stats


# ------------------------------------------------------------------ momentos clave (zigzag)
def zigzag(c: pd.Series, umbral: float) -> list[tuple[int, int]]:
    """Giros que superan `umbral` (fracción): [(índice, +1 suelo / -1 techo)]."""
    x = c.values
    piv, tend, hi_i, lo_i = [], 0, 0, 0
    for i in range(1, len(x)):
        if tend == 0:
            if x[i] > x[hi_i]:
                hi_i = i
            if x[i] < x[lo_i]:
                lo_i = i
            if x[i] >= x[lo_i] * (1 + umbral):
                piv.append((lo_i, 1)); tend, hi_i = 1, i
            elif x[i] <= x[hi_i] * (1 - umbral):
                piv.append((hi_i, -1)); tend, lo_i = -1, i
        elif tend == 1:
            if x[i] > x[hi_i]:
                hi_i = i
            elif x[i] <= x[hi_i] * (1 - umbral):
                piv.append((hi_i, -1)); tend, lo_i = -1, i
        else:
            if x[i] < x[lo_i]:
                lo_i = i
            elif x[i] >= x[lo_i] * (1 + umbral):
                piv.append((lo_i, 1)); tend, hi_i = 1, i
    return piv


def momentos_clave(df: pd.DataFrame, eventos: list[Evento], stats: dict, anios: int = 3) -> tuple[list[dict], dict]:
    c = df.close
    sig = np.log(c / c.shift(1)).std()
    umbral = float(np.clip(1.6 * sig * math.sqrt(20), 0.05, 0.30))
    piv = zigzag(c, umbral)
    desde = c.index[-1] - pd.DateOffset(years=anios)
    out, con_aviso, total = [], 0, 0
    for j, (i, d) in enumerate(piv):
        fecha = c.index[i]
        sig_ = [e for e in eventos if e.dir == d and -7 <= e.i - i <= 5 and CATALOGO[e.tipo]["tipo"] == "giro"]
        nxt = piv[j + 1][0] if j + 1 < len(piv) else len(c) - 1
        mov = (c.iloc[nxt] / c.iloc[i] - 1) * 100
        if fecha >= c.index[0] + pd.DateOffset(days=250):
            total += 1
            con_aviso += bool(sig_)
        if fecha < desde:
            continue
        out.append({
            "fecha": fecha.strftime("%Y-%m-%d"), "tipo": "suelo" if d == 1 else "techo", "precio": round(float(c.iloc[i]), 2),
            "movimiento": round(float(mov), 1), "dias": int(nxt - i), "en_curso": j + 1 == len(piv),
            "senales": [{"tipo": e.tipo, "nombre": CATALOGO[e.tipo]["nombre"], "detalle": e.detalle,
                         "fecha": e.fecha.strftime("%Y-%m-%d"), "desfase": int(e.i - i),
                         "grado": stats.get(f"{e.tipo}:{e.dir}", {}).get("grado")} for e in sig_],
        })
    giros_idx = [i for i, _ in piv]
    ev_giro = [e for e in eventos if CATALOGO[e.tipo]["tipo"] == "giro" and e.i >= 250]
    acertadas = sum(1 for e in ev_giro if any(-5 <= e.i - g <= 7 and d == e.dir for g, d in piv))
    cobertura = {"umbral_pct": round(umbral * 100, 1), "giros": total, "con_aviso": con_aviso,
                 "pct": round(100 * con_aviso / total, 0) if total else None,
                 "senales_giro": len(ev_giro),
                 "precision": round(100 * acertadas / len(ev_giro), 0) if ev_giro else None}
    return out, cobertura


def explicar_momento(m: dict, simbolo: str) -> str:
    tipo = m["tipo"]
    s = (f"{'Suelo' if tipo == 'suelo' else 'Techo'} del {pd.Timestamp(m['fecha']):%d/%m/%Y} en {_fmt(m['precio'])} $: "
         f"después el precio se movió {m['movimiento']:+.1f}% en {m['dias']} días"
         + (" (movimiento aún en curso)." if m["en_curso"] else "."))
    if not m["senales"]:
        return s + " No hubo una señal técnica clara de agotamiento: este giro lo provocaron factores externos (noticias, macro, flujos)."
    partes = []
    for e in m["senales"]:
        cuando = ("ese mismo día" if e["desfase"] == 0 else
                  f"{-e['desfase']} días antes" if e["desfase"] < 0 else f"{e['desfase']} días después (confirmación)")
        partes.append(f"{e['nombre']} {cuando}: {minus(e['detalle'])}")
    return s + " Lo que se vio: " + "; ".join(partes) + "."


# ------------------------------------------------------------------ niveles y lectura actual
def niveles(df: pd.DataFrame, n_max: int = 3) -> dict:
    c, h, l = df.close, df.high, df.low
    a = float(atr(df).iloc[-1])
    last = float(c.iloc[-1])
    zona = df.iloc[-400:]
    pl = [float(zona.low.iloc[i]) for i in np.where(pivots(zona.low, 5, "low").values)[0]]
    ph = [float(zona.high.iloc[i]) for i in np.where(pivots(zona.high, 5, "high").values)[0]]

    def agrupar(xs):
        xs = sorted(xs)
        grupos = []
        for x in xs:
            if grupos and abs(x - np.mean(grupos[-1])) <= 0.8 * a:
                grupos[-1].append(x)
            else:
                grupos.append([x])
        return [(float(np.mean(g)), len(g)) for g in grupos]

    todos = agrupar(pl + ph)
    sop = sorted([g for g in todos if g[0] < last], key=lambda g: -g[0])[:n_max]
    res = sorted([g for g in todos if g[0] > last], key=lambda g: g[0])[:n_max]
    return {"soportes": [{"nivel": round(x, 2), "toques": t} for x, t in sop],
            "resistencias": [{"nivel": round(x, 2), "toques": t} for x, t in res], "atr": round(a, 2)}


@dataclass
class ResultadoTecnico:
    df: pd.DataFrame
    eventos: list[Evento]
    stats: dict
    momentos: list[dict]
    cobertura: dict
    niveles: dict
    lectura: dict


def analizar(df: pd.DataFrame) -> ResultadoTecnico:
    df = df.dropna(subset=["close"]).copy()
    for col in ("open", "high", "low"):
        df[col] = df[col].fillna(df.close)
    df["volume"] = df["volume"].fillna(0)
    ev = detectar(df)
    st = evaluar(df, ev)
    mom, cob = momentos_clave(df, ev, st)
    nv = niveles(df)
    c = df.close
    r = rsi(c)
    ma50, ma200 = c.rolling(50).mean(), c.rolling(200).mean()
    rec = [e for e in ev if e.i >= len(df) - 10]
    presion = 0.0
    for e in rec:
        g = st.get(f"{e.tipo}:{e.dir}", {}).get("grado")
        w = {"fiable": 1.0, "media": 0.5}.get(g, 0.15)
        presion += e.dir * w * (0.85 ** (len(df) - 1 - e.i))
    texto = lectura_texto(float(r.iloc[-1]), float(c.iloc[-1]), float(ma50.iloc[-1]), float(ma200.iloc[-1]), nv,
                          [evento_dict(e, st) for e in reversed(rec)])
    lectura = {
        "texto": texto,
        "rsi": round(float(r.iloc[-1]), 1),
        "sobre_ma50": bool(c.iloc[-1] > ma50.iloc[-1]), "sobre_ma200": bool(c.iloc[-1] > ma200.iloc[-1]),
        "dist_ma50": round(float((c.iloc[-1] / ma50.iloc[-1] - 1) * 100), 1),
        "dist_ma200": round(float((c.iloc[-1] / ma200.iloc[-1] - 1) * 100), 1),
        "presion": round(float(np.clip(presion, -2, 2)) * 50, 0),
        "recientes": [evento_dict(e, st) for e in reversed(rec)],
    }
    return ResultadoTecnico(df, ev, st, mom, cob, nv, lectura)


def evento_dict(e: Evento, st: dict) -> dict:
    s = st.get(f"{e.tipo}:{e.dir}", {})
    return {"fecha": e.fecha.strftime("%Y-%m-%d"), "tipo": e.tipo, "nombre": CATALOGO[e.tipo]["nombre"],
            "familia": CATALOGO[e.tipo]["tipo"], "dir": e.dir, "precio": round(e.precio, 2), "detalle": e.detalle,
            "grado": s.get("grado"), "acierto": s.get(f"acierto{H_PRINCIPAL}"), "n": s.get(f"n{H_PRINCIPAL}"),
            "media": s.get(f"media{H_PRINCIPAL}")}


def minus(t: str) -> str:
    """Primera letra en minúscula salvo siglas (RSI, ATR...)."""
    return t if t[:2].isupper() else t[:1].lower() + t[1:]


def _pct(x: float) -> str:
    return f"{x:+.1f}%".replace(".", ",")


def lectura_texto(r: float, px: float, m50: float, m200: float, nv: dict, rec: list[dict]) -> str:
    zona = ("sobrecompra" if r >= 70 else "zona alta" if r >= 60 else "sobreventa" if r <= 30 else
            "zona baja" if r <= 40 else "zona neutral")
    tend = ("alcista (sobre las medias de 50 y 200)" if px > m50 and px > m200 else
            "bajista (bajo las medias de 50 y 200)" if px < m50 and px < m200 else
            "en corrección dentro de tendencia alcista" if px < m50 and px > m200 else
            "en rebote dentro de tendencia bajista")
    t = f"Tendencia {tend}. RSI {r:.0f} ({zona})."
    if nv["soportes"]:
        s0 = nv["soportes"][0]
        t += f" Soporte más cercano {_fmt(s0['nivel'])} ({_pct((s0['nivel']/px-1)*100)}, {s0['toques']} toques)"
    if nv["resistencias"]:
        r0 = nv["resistencias"][0]
        t += f"; resistencia {_fmt(r0['nivel'])} ({_pct((r0['nivel']/px-1)*100)}, {r0['toques']} toques)."
    buenas = [e for e in rec if e["grado"] in ("fiable", "media")]
    if buenas:
        e = buenas[0]
        t += (f" Señal reciente con historial a favor: {minus(e['nombre'])} {'alcista' if e['dir'] > 0 else 'bajista'} "
              f"({e['acierto']:.0f}% de acierto a 10 días en {e['n']} casos).")
    elif rec:
        t += " Hay señales recientes, pero ninguna con historial fiable en este activo."
    else:
        t += " Sin señales técnicas en los últimos 10 días."
    return t
