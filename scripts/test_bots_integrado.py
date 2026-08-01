"""
Test integrador: arma un mini-tablero, y hace que el SpymasterBot dé una
pista y el OperativeBot decida qué tocar, usando el mismo vocabulario
sintético de antes. Esto NO es todavía una partida completa turno a
turno (eso lo armamos en la Fase 2, simulación masiva) — es solo para
validar que el Spymaster y el Operative "se entienden" bien entre sí.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.embeddings import EmbeddingStore
from engine.board import Tablero, Dueno, Carta
from engine.spymaster import SpymasterBot
from engine.operative import OperativeBot

# Mismo mini-espacio semántico sintético de antes, un poco ampliado
# para tener más candidatas de pista disponibles.
word_vectors = {
    # cluster "mar"
    "oceano":    [0.9, 0.6, 0.0, 0.1],
    "ballena":   [0.85, 0.9, 0.0, 0.0],
    "agua":      [0.95, 0.1, 0.0, 0.3],
    "barco":     [0.8, 0.5, 0.0, 0.2],
    "submarino": [0.7, 0.2, 0.0, 0.4],  # todavía "marino" pero más alejado del cluster fuerte

    "pez":       [0.88, 0.3, 0.0, 0.15],
    "vela":      [0.5, 0.3, 0.1, 0.2],
    # cluster "muebles"
    "silla":     [0.0, 0.1, 0.9, 0.4],
    "mesa":      [0.0, 0.3, 0.95, 0.3],
    "sofa":      [0.0, 0.4, 0.9, 0.5],
    "lampara":   [0.05, 0.1, 0.7, 0.6],
    # cluster "cotidiano / animales"
    "gato":      [0.1, 0.05, 0.1, 0.9],
    "perro":     [0.12, 0.1, 0.15, 0.88],
    "libro":     [0.02, 0.05, 0.3, 0.7],
}

embeddings = EmbeddingStore.from_dict(word_vectors)

# Armamos un tablero A MANO (no al azar) para tener un caso controlado y entendible.
asignacion = {
    "ballena": Dueno.ROJO,
    "barco": Dueno.ROJO,
    "pez": Dueno.ROJO,
    "silla": Dueno.AZUL,
    "mesa": Dueno.AZUL,
    "sofa": Dueno.AZUL,
    "gato": Dueno.NEUTRAL,
    "perro": Dueno.NEUTRAL,
    "libro": Dueno.NEUTRAL,
    "lampara": Dueno.NEUTRAL,
    "submarino": Dueno.ASESINO,  # a propósito, muy cerca semánticamente del cluster "mar"
}
tablero = Tablero(cartas=[Carta(palabra=p, dueno=d) for p, d in asignacion.items()])

print("=== Tablero (vista de espía, para debug) ===")
print(tablero.resumen_texto(mostrar_duenos=True))
print()

# Vocabulario de candidatas para pistas: el resto de las palabras que
# no están en el tablero (en un caso real esto serían miles de palabras).
palabras_tablero = set(asignacion.keys())
vocab_pistas = [w for w in word_vectors.keys() if w not in palabras_tablero]
print(f"Vocabulario de candidatas para pistas: {vocab_pistas}\n")

print("=== Spymaster ROJO decide una pista ===")
spymaster = SpymasterBot(
    embeddings=embeddings,
    vocabulario_pistas=vocab_pistas,
    beta_asesino=0.10,
    beta_resto=0.05,
)
pista = spymaster.elegir_pista(tablero, Dueno.ROJO)

if pista is None:
    print("  El spymaster no encontró ninguna pista lo suficientemente segura.")
else:
    print(f"  Pista elegida: '{pista.palabra}', número = {pista.numero}")
    print(f"  Palabras objetivo: {pista.palabras_objetivo}")
    print(f"  Margen vs asesino: {pista.margen_asesino:.3f}")
    print(f"  Margen vs resto (enemigas/neutrales): {pista.margen_resto:.3f}")
print()

if pista is not None:
    print(f"=== Operative ROJO recibe la pista '{pista.palabra}, {pista.numero}' ===")
    operative = OperativeBot(embeddings=embeddings, arriesgar_extra=False)
    plan = operative.planear(tablero, pista.palabra, pista.numero)

    print("  Orden de las palabras a tocar (con su similitud a la pista):")
    for palabra, sim in zip(plan.orden, plan.similitudes):
        dueno_real = tablero.carta_de(palabra).dueno.value
        marca = "✅" if dueno_real == "rojo" else ("💀" if dueno_real == "asesino" else "❌")
        print(f"    {marca} {palabra:12s} sim={sim:.3f}  (dueño real: {dueno_real})")

    aciertos = sum(1 for p in plan.orden if tablero.carta_de(p).dueno == Dueno.ROJO)
    toco_asesino = any(tablero.carta_de(p).dueno == Dueno.ASESINO for p in plan.orden)

    print()
    if toco_asesino:
        print("  💀 ¡El plan del operative habría tocado el asesino! Mal resultado.")
    elif aciertos == len(plan.orden):
        print(f"  🎉 El operative acertó las {aciertos} palabras correctamente, sin errores.")
    else:
        print(f"  ⚠️ El operative acertó {aciertos} de {len(plan.orden)} intentos.")
