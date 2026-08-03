"""
Prueba real de GeminiSpymasterBot + GeminiOperativeBot, llamando a la
API de verdad (esto SÍ gasta de tu cuota/presupuesto, aunque con
gemini-2.5-flash-lite el costo de esta prueba es de fracciones de centavo).

Requiere la variable de entorno GEMINI_API_KEY. En PowerShell:
    $env:GEMINI_API_KEY = "tu-key-aca"

Uso:
    python pruebas/test_gemini_real.py
"""
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.board import Tablero, Dueno
from engine.gemini_bots import GeminiSpymasterBot, GeminiOperativeBot
from data.vocabulario_tablero import VOCABULARIO_TABLERO

seed = random.randint(0, 999_999)
tablero = Tablero.generar(VOCABULARIO_TABLERO, n_palabras=25, equipo_inicial=Dueno.ROJO, seed=seed)

print(f"Tablero (semilla {seed}):")
print(tablero.resumen_texto(mostrar_duenos=True))
print()

print("Pidiéndole una pista a Gemini (Spymaster)...")
spymaster = GeminiSpymasterBot()
pista = spymaster.elegir_pista(tablero, Dueno.ROJO)

if pista is None:
    print("Gemini no devolvió una pista válida (revisá tu API key o probá de nuevo).")
    sys.exit(1)

print(f"  Pista: '{pista.palabra}', número={pista.numero}")
print(f"  Objetivo (según Gemini): {pista.palabras_objetivo}")
print()

print("Pidiéndole a Gemini que decodifique su propia pista (Operative)...")
operative = GeminiOperativeBot()
plan = operative.planear(tablero, pista.palabra, pista.numero)

print(f"\nOrden que propone Gemini para '{pista.palabra}, {pista.numero}':")
aciertos = 0
for i, palabra in enumerate(plan.orden[:pista.numero], 1):
    dueno = tablero.carta_de(palabra).dueno.value
    marca = {"rojo": "✅", "azul": "❌", "neutral": "➖", "asesino": "💀"}[dueno]
    if dueno == "rojo":
        aciertos += 1
    print(f"  {i}. {marca} {palabra:15s} (dueño real: {dueno})")

print(f"\nResultado: {aciertos}/{pista.numero} correctas")
