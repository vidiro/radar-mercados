#!/bin/bash
# Doble clic: actualiza los datos y abre el panel en tu navegador (no envía nada a Telegram).
cd "$(dirname "$0")" || exit 1
echo "📡 Radar de mercados · preparando…"
if ! command -v python3 >/dev/null 2>&1; then
  osascript -e 'display dialog "Necesitas Python 3. Se abrirá la página de descarga." buttons {"OK"}' >/dev/null 2>&1
  open "https://www.python.org/downloads/macos/"; exit 1
fi
if [ ! -d .venv ]; then python3 -m venv .venv || exit 1; fi
source .venv/bin/activate
pip install --quiet --upgrade pip >/dev/null 2>&1
pip install --quiet -r requirements.txt || { echo "No se pudieron instalar las dependencias"; read -r; exit 1; }
python main.py --sin-telegram --site _site_local && open "_site_local/index.html"
echo "✅ Listo. Puedes cerrar esta ventana."
