# Codenames Bot (en español)

Bot de Código Secreto / Codenames que usa embeddings de palabras para dar
y recibir pistas.

## Cómo correr esto en tu computadora (guía paso a paso)

### 1. Instalar Python

Primero fijate si ya lo tenés instalado. Abrí una terminal:
- **Mac**: buscá "Terminal" en Spotlight (Cmd+Espacio)
- **Windows**: buscá "PowerShell" o "Símbolo del sistema" en el menú inicio
- **Linux**: ya sabés dónde está 🙂

Y corré:

```bash
python3 --version
```

Si te muestra algo como `Python 3.11.x` (cualquier 3.10 o más nuevo sirve), ya lo tenés. Si da error o dice "command not found":

- **Mac**: instalá desde https://www.python.org/downloads/ (bajá el instalador, abrilo, seguí los pasos)
- **Windows**: instalá desde https://www.python.org/downloads/ — **IMPORTANTE**: en el instalador, tildá la casilla que dice "Add Python to PATH" antes de darle a instalar
- **Linux**: `sudo apt install python3 python3-venv` (Ubuntu/Debian)

### 2. Bajar el proyecto

Te voy a pasar el proyecto como un .zip. Descomprimilo en una carpeta que encuentres fácil, por ejemplo `Documentos/codenames-bot`.

*(Más adelante, cuando subamos esto a GitHub, este paso va a ser simplemente `git clone` — pero para ir probando ya, el zip alcanza.)*

### 3. Abrir una terminal DENTRO de la carpeta del proyecto

Esto es importante: todos los comandos que siguen hay que correrlos estando parado en la carpeta `codenames-bot`.

- **Mac/Linux**: en la Terminal, escribí `cd ` (con un espacio después) y arrastrá la carpeta descomprimida hacia la ventana de la terminal, después Enter.
- **Windows**: abrí la carpeta en el Explorador de archivos, hacé click en la barra de direcciones de arriba, escribí `powershell` y Enter — eso abre una terminal ya parada en esa carpeta.

Verificá que estás en el lugar correcto:

```bash
ls        # Mac/Linux
dir       # Windows
```

Deberías ver carpetas como `engine`, `scripts`, `data`, etc.

### 4. Crear un "entorno virtual"

Esto es una carpeta aislada donde se instalan las librerías de Python **solo para este proyecto**, sin mezclarse con nada más de tu computadora. Se hace una sola vez:

```bash
python3 -m venv venv
```

Esto crea una carpeta `venv/` (no la toques, no hace falta subirla a GitHub).

### 5. Activar el entorno virtual

Esto hay que hacerlo **cada vez que abrís una terminal nueva** para trabajar en el proyecto:

```bash
# Mac/Linux:
source venv/bin/activate

# Windows (PowerShell):
venv\Scripts\Activate.ps1

# Windows (símbolo del sistema / cmd):
venv\Scripts\activate.bat
```

Si funcionó, vas a ver `(venv)` al principio de la línea de tu terminal.

> Si Windows te da un error de "permisos de ejecución de scripts" al activar, corré esto una vez:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 6. Instalar las librerías necesarias

Con el entorno activado (tenés que ver el `(venv)`):

```bash
pip install -r requirements.txt
```

Esto instala numpy y todo lo que el proyecto necesite (vamos a ir agregando más cosas a `requirements.txt` a medida que avancemos).

### 7. Correr el primer test

```bash
python3 scripts/test_embeddings_sintetico.py
```

Si ves una lista de similitudes con números tipo `0.970`, `0.093`, etc., ¡anda todo bien!

### 8. (Más adelante) Descargar los embeddings reales en español

Cuando lleguemos a esa parte:

```bash
bash scripts/download_embeddings.sh
```

Esto baja un archivo de ~380MB, así que puede tardar unos minutos.

---

## Resumen de comandos del día a día

Cada vez que quieras seguir trabajando en el proyecto:

```bash
cd Documentos/codenames-bot     # entrar a la carpeta (ajustá la ruta)
source venv/bin/activate        # activar entorno (Mac/Linux)
# o venv\Scripts\Activate.ps1   # (Windows)

# ...trabajar, correr scripts...

deactivate                      # salir del entorno cuando termines (opcional)
```
