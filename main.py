"""Radar de mercados · ejecución completa.

  python main.py                 -> descarga datos, evalúa alertas, envía Telegram y genera el panel
  python main.py --sin-telegram  -> igual pero sin enviar nada (pruebas)
  python main.py --resumen       -> fuerza el envío del resumen ahora
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from radar import engine, snapshot
from radar.alerts import Alerter, summary_text
from radar.telegram import Bot

ROOT = Path(__file__).parent
log = logging.getLogger("radar")


def panel_url() -> str | None:
    if os.environ.get("PANEL_URL"):
        return os.environ["PANEL_URL"].rstrip("/") + "/"
    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo and "/" in repo:
        owner, name = repo.split("/", 1)
        if name.lower() == f"{owner.lower()}.github.io":
            return f"https://{owner.lower()}.github.io/"
        return f"https://{owner.lower()}.github.io/{name}/"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=str(ROOT / "_site"), help="carpeta de salida del panel/estado")
    ap.add_argument("--sin-telegram", action="store_true")
    ap.add_argument("--resumen", action="store_true", help="forzar resumen ahora")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("yfinance").setLevel(logging.ERROR)

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    tz = ZoneInfo(cfg["settings"].get("timezone", "Europe/Madrid"))
    now = datetime.now(tz)
    site = Path(args.site)
    site.mkdir(parents=True, exist_ok=True)
    state_file = site / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}

    ds, assets = engine.run(cfg, cache_dir=site / "cache")
    if not assets:
        log.error("No se pudo calcular ningún activo")
        return 1

    alerter = Alerter(cfg, state, now.date())
    nuevas = alerter.evaluate(assets)
    snap = snapshot.build(ds, assets, cfg, state)
    url = panel_url()

    bot = Bot(state)
    if args.sin_telegram:
        bot.token = ""
    if nuevas:
        bot.send("🔔 <b>Alertas</b>\n\n" + "\n\n".join(a["texto"] for a in nuevas)
                 + (f'\n\n🔗 <a href="{url}">Panel</a>' if url else ""))
    hora = int(cfg["settings"].get("resumen_diario_hora", 8))
    if args.resumen or (now.hour >= hora and state.get("ultimo_resumen") != now.date().isoformat()):
        if bot.send(summary_text(snap, url, f"☀️ <b>Resumen diario</b> · {now:%d/%m/%Y}")) or args.sin_telegram:
            state["ultimo_resumen"] = now.date().isoformat()
    bot.process_commands(snap, url)

    # ---- escribir panel y estado
    snap["alertas"] = state.get("historial", [])[-40:]
    snap["panel_url"] = url
    (site / "data.js").write_text("window.RADAR = " + json.dumps(snap, ensure_ascii=False, separators=(",", ":"))
                                  + ";\n", encoding="utf-8")
    shutil.copy(ROOT / "panel" / "index.html", site / "index.html")
    (site / ".nojekyll").write_text("")
    engine.save_cache(ds, site / "cache")
    state["ultima_ejecucion"] = now.isoformat(timespec="minutes")
    state_file.write_text(json.dumps(state, ensure_ascii=False, indent=1))

    for k, a in snap["activos"].items():
        log.info("%s: %s $ · puntuación %s · %s", a["nombre"], a["precio"], a["puntuacion"], a["regimen"])
    log.info("Alertas nuevas: %d · panel en %s", len(nuevas), site / "index.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
