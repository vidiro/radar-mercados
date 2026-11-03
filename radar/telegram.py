"""Bot de Telegram: enviar mensajes y responder comandos (/estado, /btc, /oro...).

No necesita servidor: en cada ejecución programada lee los mensajes nuevos (getUpdates)
y contesta. El primer chat que escribe /start al bot queda autorizado.
"""
from __future__ import annotations

import logging
import os

import requests

from .alerts import asset_text, summary_text

log = logging.getLogger("radar")
API = "https://api.telegram.org/bot{token}/{method}"

AYUDA = (
    "🤖 <b>Radar de mercados</b>\n\n"
    "/estado – resumen de todos los activos\n"
    "/btc – detalle de Bitcoin\n"
    "/oro – detalle del oro\n"
    "/alertas – últimas alertas\n"
    "/panel – enlace al panel\n"
    "/ayuda – esta ayuda\n\n"
    "Respondo en cada ejecución programada (cada ~30 min). Las alertas te llegan solas."
)
ALIAS = {"oro": "gold", "xau": "gold", "gold": "gold", "btc": "btc", "bitcoin": "btc"}


class Bot:
    def __init__(self, state: dict):
        self.token = os.environ.get("TELEGRAM_TOKEN", "").strip()
        self.state = state
        env_chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if env_chat:
            state["chat_id"] = env_chat

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    @property
    def chat_id(self):
        return self.state.get("chat_id")

    def _call(self, method: str, **params):
        r = requests.post(API.format(token=self.token, method=method), json=params, timeout=30)
        j = r.json()
        if not j.get("ok"):
            raise RuntimeError(f"Telegram {method}: {j.get('description')}")
        return j["result"]

    def send(self, text: str, chat_id=None) -> bool:
        cid = chat_id or self.chat_id
        if not (self.enabled and cid):
            log.info("Telegram no configurado; mensaje:\n%s", text)
            return False
        try:
            for chunk in _chunks(text):
                self._call("sendMessage", chat_id=cid, text=chunk, parse_mode="HTML",
                           disable_web_page_preview=True)
            return True
        except Exception as e:  # noqa: BLE001
            log.error("Error enviando a Telegram: %s", e)
            return False

    def process_commands(self, snap: dict, panel_url: str | None) -> int:
        if not self.enabled:
            return 0
        try:
            me = self._call("getMe")
            log.info("Bot de Telegram: @%s · chat autorizado: %s", me.get("username"),
                     "sí" if self.chat_id else "todavía no (escribe /start al bot)")
            try:
                wh = self._call("getWebhookInfo")
                if wh.get("url"):
                    log.info("El bot tenía un webhook; se elimina para poder leer mensajes")
                    self._call("deleteWebhook")
            except Exception:  # noqa: BLE001
                pass
            updates = self._call("getUpdates", offset=self.state.get("tg_offset", 0) + 1, timeout=0,
                                 allowed_updates=["message"])
            log.info("Mensajes nuevos recibidos por el bot: %d", len(updates))
        except Exception as e:  # noqa: BLE001
            log.error("Telegram getUpdates: %s", e)
            return 0
        n = 0
        for u in updates:
            self.state["tg_offset"] = max(self.state.get("tg_offset", 0), u["update_id"])
            msg = u.get("message") or {}
            text = (msg.get("text") or "").strip()
            cid = str((msg.get("chat") or {}).get("id", ""))
            if not text or not cid:
                continue
            if not self.chat_id:
                self.state["chat_id"] = cid
                self.send("✅ <b>Conectado.</b> A partir de ahora te enviaré aquí las alertas.\n\n" + AYUDA, cid)
                self.send(summary_text(snap, panel_url), cid)
                n += 1
                continue
            if cid != str(self.chat_id):
                continue  # ignorar desconocidos
            cmd = text.split()[0].split("@")[0].lstrip("/").lower()
            if cmd in ("start", "ayuda", "help"):
                self.send(AYUDA)
            elif cmd in ("estado", "resumen", "status"):
                self.send(summary_text(snap, panel_url))
            elif cmd in ALIAS and ALIAS[cmd] in snap["activos"]:
                k = ALIAS[cmd]
                self.send(asset_text(k, snap["activos"][k], short=False)
                          + (f'\n\n🔗 <a href="{panel_url}#{k}">Ver en el panel</a>' if panel_url else ""))
            elif cmd in snap["activos"]:
                self.send(asset_text(cmd, snap["activos"][cmd], short=False))
            elif cmd == "alertas":
                al = snap.get("alertas", [])[-8:]
                self.send("🔔 <b>Últimas alertas</b>\n\n" + ("\n\n".join(
                    f"<i>{a['fecha'][:10]}</i>\n{a['texto']}" for a in al) if al else "Ninguna todavía."))
            elif cmd == "panel":
                self.send(f'🔗 <a href="{panel_url}">{panel_url}</a>' if panel_url else "Panel no publicado aún.")
            else:
                self.send("No entiendo ese comando. Escribe /ayuda")
            n += 1
        return n


def _chunks(text: str, size: int = 3900):
    while len(text) > size:
        cut = text.rfind("\n", 0, size)
        cut = cut if cut > 0 else size
        yield text[:cut]
        text = text[cut:]
    yield text
