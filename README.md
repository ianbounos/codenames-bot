# Codenames Bot (Código Secreto) — en español, con embeddings y LLMs

Un bot de [Código Secreto](https://es.wikipedia.org/wiki/Codenames) (Codenames)
en español que da y decodifica pistas usando embeddings de palabras y,
opcionalmente, la API de Gemini. Incluye un motor de simulación para hacer
jugar a distintas versiones del bot entre sí, miles de partidas, y medir
qué tan bien juegan.

Este es un proyecto **educativo**: el objetivo principal fue aprender
desarrollo de software, Git/GitHub, y conceptos de NLP/embeddings desde
cero, construyendo algo divertido en el camino. La [nota técnica](docs/NOTA_TECNICA.md)
cuenta la historia completa (diseño, iteraciones, y qué encontramos).

---

## Instalación

### 1. Requisitos

- Python 3.10 o más nuevo
- ~1GB de espacio libre (para los modelos de embeddings)

### 2. Clonar el repo y crear un entorno virtual

```bash
git clone https://github.com/ianbounos/codenames-bot.git
cd codenames-bot
python -m venv venv
```

Activar el entorno virtual:

```bash
# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Mac / Linux
source venv/bin/activate
```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 4. Descargar el modelo de español de spaCy

```bash
python -m spacy download es_core_news_md
```

### 5. Descargar la lista de frecuencia de palabras en español

```bash
python scripts/download_lista_frecuencia.py
```

### 6. (Opcional) Configurar la API de Gemini

Solo si querés usar los bots basados en LLM (`GeminiSpymasterBot` /
`GeminiOperativeBot`). Conseguí una API key gratis en
[aistudio.google.com/apikey](https://aistudio.google.com/apikey), y
configurala como variable de entorno (nunca la escribas en el código):

```bash
# Windows (PowerShell)
$env:GEMINI_API_KEY = "tu-key-aca"

# Mac / Linux
export GEMINI_API_KEY="tu-key-aca"
```

Hay que repetir esto cada vez que abrís una terminal nueva.

---

## Cómo probarlo

### Ver una partida jugada por los bots locales (rápido, sin Gemini)

```bash
python pruebas/test_bots_real.py
```

Genera un tablero al azar, hace que el Spymaster elija una pista, y
muestra cómo la decodifica el Operative.

### Jugar vos mismo de Spymaster

```bash
python pruebas/decodificar_pista_manual.py
```

Te muestra un tablero completo (con los colores, como si fueras el
espía) y te deja escribir tu propia pista para ver cómo la interpretaría
el bot.

### Comparar qué pista sugiere cada versión del bot (v1, v2, v3)

```bash
python pruebas/comparar_perfiles_manual.py
```

### Simular cientos de partidas y generar un dashboard

```bash
python pruebas/simular_partidas.py
```

Corre un torneo entre las distintas versiones del bot (ver la
[nota técnica](docs/NOTA_TECNICA.md) para qué significa cada una), y
guarda en `results/`:
- `resultados_simulacion.csv` — resumen de cada partida
- `historial_completo.jsonl` — cada pista y adivinanza, turno a turno
- `dashboard_simulacion.png` — panel visual con las estadísticas

### Simular partidas con Gemini (requiere API key, ver arriba)

```bash
python pruebas/torneo_completo_gemini.py
```

Te muestra una estimación de costo y partidas antes de arrancar, y pide
confirmación. Con la capa gratuita de Gemini (sin facturación activada),
esto no debería generarte ningún costo.

---

## Estructura del proyecto

```
codenames-bot/
├── engine/              # Motor del juego: tablero, reglas, bots, embeddings
│   ├── board.py         # Representación del tablero
│   ├── game.py          # Lógica de turnos y condiciones de victoria
│   ├── embeddings.py    # Carga y manejo de embeddings (spaCy, sentence-transformers)
│   ├── embedding_ensemble.py   # Combinación ponderada de varios modelos (v3)
│   ├── embedding_hibrido.py    # Fallback dinámico para palabras raras
│   ├── spymaster.py     # Bot que da pistas
│   ├── operative.py     # Bot que decodifica pistas
│   ├── operative_hibrido.py    # Versión híbrida del operative
│   ├── gemini_bots.py   # Bots respaldados por la API de Gemini
│   ├── lemmatizador.py  # Chequeo de "misma palabra" por lema (no substring)
│   ├── vocab_utils.py   # Filtrado del vocabulario de pistas
│   └── simulacion.py    # Motor de simulación masiva de partidas
├── data/
│   ├── vocabulario_tablero.py  # ~180 palabras para el tablero
│   └── stopwords_es.py
├── scripts/             # Utilidades de setup (descargas)
├── pruebas/             # Scripts para correr y probar el proyecto
├── results/             # Salida de las simulaciones (se genera al correr)
└── docs/
    └── NOTA_TECNICA.md  # Metodología, resultados, y conclusiones del proyecto
```

## Licencia

MIT — usalo, modificalo, lo que quieras.
