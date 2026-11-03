# 📡 Radar de Mercados

Panel que reúne todo lo que mueve el precio del **Bitcoin** y del **oro**: liquidez de la Fed, tipos reales,
dólar, inflación esperada, apetito por el riesgo, posicionamiento y datos on-chain. Lo resume en una
**puntuación de −100 (entorno adverso) a +100 (entorno favorable)** y te manda **alertas por Telegram**.

Funciona solo, en la nube y gratis (GitHub Actions cada 30 minutos). El panel se abre con un enlace
desde el móvil o el ordenador.

---

## Puesta en marcha (unos 10 minutos y una sola vez)

### 1. Crea tu bot de Telegram (2 min)
1. En Telegram busca **@BotFather** → escribe `/newbot`.
2. Ponle un nombre (p. ej. *Mi Radar*) y un usuario acabado en `bot` (p. ej. `miradar_vicente_bot`).
3. BotFather te da un **token** del tipo `123456789:AAH...`. **Guárdalo**, es la llave del bot.
4. Abre el chat con tu bot y pulsa **Iniciar** (`/start`). Así el radar sabrá a quién escribir.

### 2. Sube el proyecto a GitHub (5 min)
1. Crea una cuenta gratis en <https://github.com> si no tienes.
2. Arriba a la derecha **+ → New repository**. Nombre: `radar-mercados`. Visibilidad **Public**
   (necesario para que el panel web y las ejecuciones sean gratis; en el repositorio no hay
   nada privado: el token va en un "secreto" que nadie ve).
3. En el repositorio vacío pulsa **uploading an existing file** y arrastra **todo el contenido** de
   esta carpeta. ⚠️ Incluye la carpeta oculta `.github` → en el Finder pulsa **⌘ + ⇧ + .** para ver
   los archivos ocultos antes de arrastrar. Pulsa **Commit changes**.

### 3. Mete el token del bot como secreto (1 min)
**Settings → Secrets and variables → Actions → New repository secret**
- Name: `TELEGRAM_TOKEN`
- Secret: el token de BotFather

### 4. Activa el panel web (30 s)
**Settings → Pages → Build and deployment → Source: _GitHub Actions_**.

### 5. Primera ejecución
**Actions → Radar de mercados → Run workflow → Run workflow**.
En ~2 minutos te llegará a Telegram “✅ Conectado” y el primer resumen.
Tu panel queda en: `https://TU-USUARIO.github.io/radar-mercados/`

> A partir de ahí se ejecuta solo cada 30 minutos. El resumen diario llega a las 8:00 (hora de Madrid).

---

## Qué te llega al móvil

| Alerta | Cuándo salta |
|---|---|
| 🔄 Cambio de régimen | Un activo pasa de Neutral a Favorable/Adverso o viceversa |
| ⚡ Entorno muy favorable / adverso | La puntuación supera ±50 |
| 📈 Cruce de la media de 200 días | El precio la recupera o la pierde |
| 💥 Movimiento extremo | El precio se mueve más de 2,5× su volatilidad diaria habitual |
| ↔️ Divergencia | El precio va en una dirección y el entorno en la contraria |
| 💧 Giro de liquidez | La liquidez neta de la Fed cambia con fuerza |
| 📉 Tipo real brusco | El tipo real a 10 años se mueve de forma anómala |
| 💵 Dólar brusco | Movimiento anómalo del DXY |
| ⚠️ Funding extremo | Apalancamiento excesivo en los perpetuos de BTC |
| 😱 / 🤑 Miedo o codicia extremos | Fear & Greed ≤ 15 o ≥ 85 |

Los avisos **no se repiten**: cada uno tiene un periodo de enfriamiento.

**Comandos del bot:** `/estado`, `/btc`, `/oro`, `/alertas`, `/panel`, `/ayuda`.
El bot responde en la siguiente ejecución, como mucho en 30 minutos (no hay un servidor encendido
24 h, y por eso es gratis).

---

## Cómo se calcula la puntuación

Para cada factor se mide su cambio reciente (normalmente 4 semanas) o su nivel, se normaliza frente a
los últimos 2 años (*z-score*) y se le aplica el signo esperado (p. ej. un dólar que sube es malo para
el oro). Luego se hace una media ponderada.

**Bitcoin:** liquidez neta de la Fed (balance − TGA − RRP), M2 con 10 semanas de retraso, oferta de
stablecoins, dólar, tipo real, diferencial *high yield*, VIX, tendencia del Nasdaq, *funding* de los
perpetuos, BTC entrando a exchanges y tendencia propia. *Contexto:* Fear & Greed y MVRV.

**Oro:** tipo real 10A, dólar, bono 2A (expectativas de la Fed), inflación esperada 5y5y, VIX,
diferencial *high yield*, posicionamiento de los especuladores (COT, de forma *contrarian*) y tendencia.
*Contexto:* liquidez y ratio cobre/oro.

En el panel, la columna **Fiabilidad** dice qué factores han funcionado de verdad en los últimos 2 años,
y la tarjeta **¿Funciona la puntuación?** muestra la rentabilidad histórica en cada régimen. Hoy el modelo
del oro discrimina razonablemente bien. El de BTC es más débil, porque su precio depende mucho de flujos
propios (ETF, apalancamiento). Úsalo como termómetro, no como gatillo.

> ⚠️ Herramienta de análisis. **No es una recomendación de inversión.**

---

## Análisis técnico (giros y agotamiento)

Cada día, con las velas cerradas, el radar busca señales técnicas clásicas inspiradas en Murphy,
Edwards & Magee y Wyckoff:

- **Giros y agotamiento:** divergencias RSI, salida de sobrecompra/sobreventa, reentrada en bandas de
  Bollinger, rechazo en soportes y resistencias, dobles suelos y techos, clímax de volumen (Wyckoff),
  sobreextensión respecto a la media de 50 y velas de giro (envolventes, martillos, estrellas fugaces).
- **Continuación:** rupturas del rango de 55 días, salida de compresión de volatilidad y cruce de las
  medias de 50 y 200.

**Cada señal se valida con el histórico de cada activo.** Se mide su % de acierto a 5, 10 y 20 días y el
resultado se compara con lo que habría hecho el precio de todas formas. Con eso recibe un veredicto:
*fiable*, *media*, *no fiable* o *pocos casos*. **Solo las fiables y las medias te llegan por Telegram.**
Si además coinciden con el entorno macro, el aviso lleva ⭐ **Confluencia**.

En el panel verás:
- Las velas con las señales marcadas (▲ ▼) y los grandes giros (◆).
- Los soportes y resistencias actuales.
- Los **momentos clave explicados**: qué señales había antes de cada gran suelo o techo de los últimos 3 años.
- La tabla con qué señales funcionan de verdad en cada activo.

Para cambiar qué grados te llegan al móvil, o para recibir solo los giros, usa `alerts.senal_tecnica` en `config.yaml`.

---

## Personalizar

Todo está en **`config.yaml`** (se edita desde la web de GitHub con el lápiz ✏️):
- **Pesos y factores:** `peso: 0` deja un factor solo como contexto.
- **Umbrales de alertas:** sección `alerts` (`activa: false` para apagar una).
- **Hora del resumen:** `resumen_diario_hora`.

### Añadir otro activo o ETF (p. ej. plata, el S&P 500 o un ETF)
1. En `series` añade su precio desde Yahoo:
   `spy: {source: yahoo, id: "SPY", name: "S&P 500 (SPY)", unit: "$"}`
2. En `assets` copia el bloque `gold:` y cámbialo a `spy:`, con `price: spy` y los factores que
   quieras.

## Ver el panel en local (opcional)
Doble clic en **`Abrir panel en mi Mac.command`**. Descarga los datos y abre el panel en el navegador
(necesita Python 3; la primera vez tarda un poco).

## Fuentes (todas gratuitas, sin claves)
FRED (Reserva Federal de St. Louis) · Yahoo Finance · Coin Metrics Community · Deribit · DefiLlama ·
CFTC (informe COT) · alternative.me (Fear & Greed). Si una fuente falla, se usa el último dato
guardado y el panel lo indica.
