"""
Prueba manual del motor de reglas (Tablero + Partida), sin bots todavía.
Simulamos una partida "guionada" para validar que las reglas se cumplen:
turnos, límite de intentos, y las 3 formas de terminar el juego.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.board import Tablero, Dueno

VOCAB_PRUEBA = [
    "oceano", "ballena", "agua", "barco", "submarino",
    "silla", "mesa", "sofa", "lampara", "gato",
    "perro", "montana", "rio", "sol", "luna",
    "estrella", "planeta", "cohete", "robot", "libro",
    "lapiz", "papel", "puerta", "ventana", "reloj",
    "asesino_de_prueba",  # 26 palabras para tener margen
]

print("=== Test 1: generación de tablero ===")
tablero = Tablero.generar(VOCAB_PRUEBA, n_palabras=25, equipo_inicial=Dueno.ROJO, seed=42)

n_rojo = len(tablero.palabras_de(Dueno.ROJO))
n_azul = len(tablero.palabras_de(Dueno.AZUL))
n_neutral = len(tablero.palabras_de(Dueno.NEUTRAL))
n_asesino = len(tablero.palabras_de(Dueno.ASESINO))

print(f"  Rojo: {n_rojo}, Azul: {n_azul}, Neutral: {n_neutral}, Asesino: {n_asesino}")
assert n_rojo == 9, "Rojo debería tener 9 (equipo inicial)"
assert n_azul == 8, "Azul debería tener 8"
assert n_neutral == 7
assert n_asesino == 1
print("  ✅ Distribución correcta (9/8/7/1)\n")

print("=== Test 2: partida con adivinanza correcta simple ===")
from engine.game import Partida

partida = Partida(tablero, equipo_inicial=Dueno.ROJO)
palabra_asesino = tablero.palabras_de(Dueno.ASESINO)[0]
palabras_rojas = tablero.palabras_de(Dueno.ROJO)
palabras_azules = tablero.palabras_de(Dueno.AZUL)
palabras_neutrales = tablero.palabras_de(Dueno.NEUTRAL)

print(f"  (para referencia, en este test SÍ miramos las cartas: asesino={palabra_asesino})")

partida.dar_pista("prueba", 1)
resultado = partida.adivinar(palabras_rojas[0])
print(f"  Adiviné una palabra roja, dueño real: {resultado.value}")
assert resultado == Dueno.ROJO

# Como acertó y el límite era 1+1=2 intentos, todavía puede seguir
print(f"  Equipo actual después de acertar (debería seguir siendo rojo): {partida.equipo_actual.value}")
assert partida.equipo_actual == Dueno.ROJO

# Ahora falla (toca neutral) -> debería pasar el turno
resultado2 = partida.adivinar(palabras_neutrales[0])
print(f"  Toqué neutral, dueño real: {resultado2.value}")
print(f"  Equipo actual después de fallar (debería pasar a azul): {partida.equipo_actual.value}")
assert partida.equipo_actual == Dueno.AZUL
print("  ✅ Cambio de turno correcto\n")

print("=== Test 3: tocar el asesino termina la partida ===")
partida.dar_pista("prueba2", 1)
resultado3 = partida.adivinar(palabra_asesino)
print(f"  Equipo azul tocó el asesino")
print(f"  Resultado de la partida: {partida.resultado.value}")
from engine.game import Resultado
assert partida.resultado == Resultado.GANA_ROJO, "Si azul toca el asesino, gana rojo"
assert not partida.en_curso()
print("  ✅ El asesino termina la partida correctamente y gana el equipo rival\n")

print("=== Test 4: ganar completando todas las palabras propias ===")
tablero2 = Tablero.generar(VOCAB_PRUEBA, n_palabras=25, equipo_inicial=Dueno.ROJO, seed=7)
partida2 = Partida(tablero2, equipo_inicial=Dueno.ROJO)
palabras_rojas2 = tablero2.palabras_de(Dueno.ROJO)

# Una sola pista que "cubre" todas las palabras rojas, y las vamos
# adivinando una por una en el mismo turno (como si el spymaster hubiera
# dado una pista perfecta conectando las 9 de una).
partida2.dar_pista("pista_perfecta", len(palabras_rojas2))
for palabra in palabras_rojas2:
    if not partida2.en_curso():
        break
    partida2.adivinar(palabra)

print(f"  Resultado final: {partida2.resultado.value}")
assert partida2.resultado == Resultado.GANA_ROJO
print("  ✅ Completar todas las palabras propias gana la partida\n")

print("🎉 Todos los tests pasaron correctamente.")
