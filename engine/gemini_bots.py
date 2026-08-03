"""
Bots de Codenames que usan la API de Gemini (Google) en vez de embeddings
locales. Misma interfaz que SpymasterBot/OperativeBot, así se pueden usar
en las mismas herramientas (comparar_perfiles_manual.py, etc.) sin
cambios en el resto del código.

REQUIERE una variable de entorno GEMINI_API_KEY con tu API key de
Google AI Studio (https://aistudio.google.com/apikey -- es gratis
conseguirla).

IMPORTANTE - seguridad: la API key NUNCA debe escribirse directo en el
código ni subirse a GitHub. Se lee desde una variable de entorno.
En PowerShell, para esta sesión de terminal:
    $env:GEMINI_API_KEY = "tu-key-aca"
"""
from __future__ import annotations

import json
import os
import re
import time
import unicodedata

import requests

from engine.board import Tablero, Dueno
from engine.spymaster import CandidatoPista
from engine.operative import PlanDeJuego

MODELO_DEFAULT = "gemini-3.1-flash-lite"  # gemini-2.5-flash-lite fue discontinuado para cuentas nuevas
URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
PAUSA_ENTRE_LLAMADAS_SEGUNDOS = 1.5  # con facturación activa, límites más altos que la capa gratuita
VERBOSE = True  # si es True, imprime el motivo real de cada fallo (para diagnosticar)

# --- LÍMITE DURO DE SEGURIDAD ---
# Si el número total de llamadas a Gemini en esta ejecución supera este
# valor, se frena TODO el script inmediatamente (no solo la partida en
# curso), para poner un techo firme al gasto pase lo que pase -- incluso
# si algo entra en un loop de reintentos inesperado.
#
# El torneo completo (49 matchups x 10 partidas) necesita ~3.360 llamadas
# en el caso normal (costo real ~$1 con gemini-3.1-flash-lite). Este
# límite no busca ahorrar centavos -- busca atrapar un BUG real (como el
# loop de reintentos con la misma palabra rechazada) antes de que se
# descontrole, no frenar una corrida legítima a mitad de camino.
MAX_LLAMADAS_TOTAL = 8000
_contador_llamadas = 0


class LimiteDeSeguridadExcedido(Exception):
    """Se superó MAX_LLAMADAS_TOTAL. Frena todo el script a propósito."""
    pass


def obtener_contador_llamadas() -> int:
    return _contador_llamadas


# --- LÍMITE POR PARTIDA (el que realmente importa en la práctica) ---
# Si UNA MISMA partida acumula más de este número de fallos de Gemini
# (entre todos sus turnos, dando pistas o decodificando), se da esa
# partida por ANULADA y se pasa a la siguiente -- en vez de insistir
# indefinidamente con algo que evidentemente está trabado (como el loop
# de "mar" rechazada una y otra vez).
LIMITE_FALLOS_POR_PARTIDA = 5
_fallos_partida_actual = 0


class PartidaAnuladaPorFallosRepetidos(Exception):
    """Se superó LIMITE_FALLOS_POR_PARTIDA fallos en la partida actual."""
    pass


def reset_contador_fallos_partida() -> None:
    """Hay que llamar esto al arrancar CADA partida nueva."""
    global _fallos_partida_actual
    _fallos_partida_actual = 0


def _registrar_fallo() -> None:
    global _fallos_partida_actual
    _fallos_partida_actual += 1
    if _fallos_partida_actual > LIMITE_FALLOS_POR_PARTIDA:
        raise PartidaAnuladaPorFallosRepetidos(
            f"Se superaron {LIMITE_FALLOS_POR_PARTIDA} fallos de Gemini en esta partida."
        )


def _sin_tildes(s: str) -> str:
    """Normaliza para comparar tolerando diferencias de acentuación
    (ej. 'cientifico' == 'científico'), porque el LLM no siempre
    reproduce las tildes exactas de nuestro vocabulario."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s.lower())
        if unicodedata.category(c) != "Mn"
    )


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "Falta la variable de entorno GEMINI_API_KEY. "
            "Conseguila gratis en https://aistudio.google.com/apikey y "
            "configurala con: $env:GEMINI_API_KEY = \"tu-key\" (PowerShell)"
        )
    return key


def _llamar_gemini(prompt: str, modelo: str = MODELO_DEFAULT, timeout: int = 30) -> str:
    """Llama a la API de Gemini y devuelve el texto crudo de la respuesta."""
    global _contador_llamadas

    if _contador_llamadas >= MAX_LLAMADAS_TOTAL:
        print(f"\n{'!'*70}")
        print(f"LÍMITE DE SEGURIDAD ALCANZADO: {MAX_LLAMADAS_TOTAL} llamadas a Gemini.")
        print("Frenando TODO el script a propósito, para no seguir gastando.")
        print(f"{'!'*70}\n")
        raise LimiteDeSeguridadExcedido(
            f"Se alcanzó el límite de {MAX_LLAMADAS_TOTAL} llamadas a Gemini."
        )

    _contador_llamadas += 1
    if _contador_llamadas % 20 == 0:
        print(f"  [Gemini] llamadas hechas hasta ahora: {_contador_llamadas}/{MAX_LLAMADAS_TOTAL}")
    time.sleep(PAUSA_ENTRE_LLAMADAS_SEGUNDOS)  # evitar saturar el límite gratuito de RPM

    url = f"{URL_BASE}/{modelo}:generateContent?key={_api_key()}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4},
    }
    resp = requests.post(url, json=body, timeout=timeout)

    if resp.status_code != 200:
        if VERBOSE:
            print(f"  [Gemini ERROR {resp.status_code}]: {resp.text[:300]}")
        if resp.status_code == 429:
            # Backoff: esperamos bastante más antes de que el próximo
            # reintento del loop llamador vuelva a golpear la API
            time.sleep(15.0)
        resp.raise_for_status()

    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        if VERBOSE:
            print(f"  [Gemini ERROR de formato]: {data}")
        raise RuntimeError(f"Respuesta inesperada de Gemini: {data}") from e


def _parsear_json(texto: str) -> dict:
    """Extrae JSON de la respuesta, tolerando que venga envuelto en
    bloques de código markdown (```json ... ```) como suelen hacer los LLMs."""
    limpio = texto.strip()
    limpio = re.sub(r"^```(?:json)?\s*", "", limpio)
    limpio = re.sub(r"\s*```$", "", limpio)
    return json.loads(limpio)


class GeminiSpymasterBot:
    def __init__(self, modelo: str = MODELO_DEFAULT, max_reintentos: int = 2):
        self.modelo = modelo
        self.max_reintentos = max_reintentos

    def elegir_pista(self, tablero: Tablero, equipo: Dueno) -> CandidatoPista | None:
        equipo_rival = Dueno.AZUL if equipo == Dueno.ROJO else Dueno.ROJO
        propias = tablero.palabras_de(equipo)
        enemigas = tablero.palabras_de(equipo_rival)
        neutrales = tablero.palabras_de(Dueno.NEUTRAL)
        asesino = tablero.palabras_de(Dueno.ASESINO)
        todas_tablero = set(w.lower() for w in tablero.palabras_visibles())

        prompt_base = self._armar_prompt(propias, enemigas, neutrales, asesino)
        rechazadas: list[str] = []

        for intento in range(self.max_reintentos + 1):
            prompt = prompt_base
            if rechazadas:
                prompt += (
                    f"\n\nIMPORTANTE: ya rechazamos estas palabras, NO las repitas "
                    f"(elegí una completamente distinta): {', '.join(rechazadas)}"
                )
            try:
                texto = _llamar_gemini(prompt, self.modelo)
                data = _parsear_json(texto)

                palabra = str(data["pista"]).strip().lower()
                numero = int(data["numero"])
                objetivo_crudo = [str(w).strip().lower() for w in data.get("objetivo", [])]

                # Mapeamos cada palabra objetivo a la ortografía EXACTA del
                # tablero, tolerando diferencias de tildes (ver _sin_tildes)
                propias_por_forma_normalizada = {_sin_tildes(p): p for p in propias}
                objetivo = [
                    propias_por_forma_normalizada[_sin_tildes(o)]
                    for o in objetivo_crudo
                    if _sin_tildes(o) in propias_por_forma_normalizada
                ]

                if self._viola_regla_substring(palabra, todas_tablero):
                    if VERBOSE:
                        print(f"  [Gemini Spymaster] descartada '{palabra}': viola regla de substring")
                    rechazadas.append(palabra)
                    _registrar_fallo()
                    continue
                if not objetivo:
                    if VERBOSE:
                        print(f"  [Gemini Spymaster] descartada: objetivo inválido {objetivo_crudo} "
                              f"(propias válidas: {propias})")
                    rechazadas.append(palabra)
                    _registrar_fallo()
                    continue

                return CandidatoPista(
                    palabra=palabra,
                    numero=numero,
                    palabras_objetivo=objetivo,
                    margen_asesino=float("nan"),  # no aplica: no usamos embeddings acá
                    margen_resto=float("nan"),
                    score=float("nan"),
                )
            except (LimiteDeSeguridadExcedido, PartidaAnuladaPorFallosRepetidos):
                raise  # NUNCA atrapar estas -- tienen que propagarse
            except Exception as e:
                if VERBOSE:
                    print(f"  [Gemini Spymaster] excepción en intento {intento}: {type(e).__name__}: {e}")
                _registrar_fallo()
                continue

        return None

    @staticmethod
    def _armar_prompt(propias, enemigas, neutrales, asesino) -> str:
        return f"""Estás jugando Código Secreto (Codenames) en español, como Spymaster.

REGLAS:
- Das UNA sola palabra como pista, más un número.
- El número indica cuántas palabras del tablero creés que tu compañero
  va a poder identificar a partir de tu pista.
- La pista NO puede ser ninguna palabra del tablero, ni contenerla, ni
  estar contenida en ella.
- Si tu compañero toca la palabra ASESINO, pierden inmediatamente.
  Evitá cualquier ambigüedad con esa palabra, incluso si eso significa
  una pista más chica y segura.

Palabras de tu equipo (las que querés que adivinen):
{", ".join(propias)}

Palabras del rival (evitar):
{", ".join(enemigas)}

Palabras neutrales (evitar si es posible):
{", ".join(neutrales)}

Palabra ASESINO (evitar a toda costa):
{", ".join(asesino)}

Respondé ÚNICAMENTE con JSON válido, sin texto adicional ni backticks,
con este formato exacto:
{{"pista": "palabra", "numero": N, "objetivo": ["palabra1", "palabra2"]}}
"""

    @staticmethod
    def _viola_regla_substring(candidata: str, palabras_tablero: set[str]) -> bool:
        from engine.lemmatizador import viola_regla_palabra_relacionada
        return viola_regla_palabra_relacionada(candidata, palabras_tablero)


class GeminiOperativeBot:
    def __init__(self, modelo: str = MODELO_DEFAULT, max_reintentos: int = 2):
        self.modelo = modelo
        self.max_reintentos = max_reintentos

    def planear(self, tablero: Tablero, pista_palabra: str, pista_numero: int) -> PlanDeJuego:
        visibles = tablero.palabras_visibles()
        prompt = self._armar_prompt(visibles, pista_palabra, pista_numero)

        for intento in range(self.max_reintentos + 1):
            try:
                texto = _llamar_gemini(prompt, self.modelo)
                data = _parsear_json(texto)
                orden_crudo = [str(w).strip().lower() for w in data["orden"]]

                # Igual que en el Spymaster: toleramos diferencias de tilde
                visibles_por_forma_normalizada = {_sin_tildes(v): v for v in visibles}
                orden_validado = [
                    visibles_por_forma_normalizada[_sin_tildes(w)]
                    for w in orden_crudo
                    if _sin_tildes(w) in visibles_por_forma_normalizada
                ]

                if not orden_validado:
                    _registrar_fallo()
                    continue

                # Gemini no da un puntaje numérico de similitud real; usamos
                # un ranking decreciente sintético solo para mantener la
                # misma interfaz que PlanDeJuego espera
                n = len(orden_validado)
                similitudes = [1.0 - i / max(n, 1) for i in range(n)]

                return PlanDeJuego(orden=orden_validado, similitudes=similitudes)
            except (LimiteDeSeguridadExcedido, PartidaAnuladaPorFallosRepetidos):
                raise  # NUNCA atrapar estas -- tienen que propagarse
            except Exception:
                _registrar_fallo()
                continue

        # Última instancia: no se pudo obtener nada usable de Gemini
        return PlanDeJuego(orden=[], similitudes=[])

    @staticmethod
    def _armar_prompt(visibles, pista_palabra, pista_numero) -> str:
        return f"""Estás jugando Código Secreto (Codenames) en español, como
Field Operative (el que adivina). NO sabés los colores de las cartas.

Tu Spymaster te dio la pista: "{pista_palabra}", número {pista_numero}

Palabras visibles en el tablero (en cualquier orden):
{", ".join(visibles)}

Ordená TODAS estas palabras de más a menos relacionadas con la pista,
de acuerdo a cuáles creés que tu Spymaster quiso señalar.

Respondé ÚNICAMENTE con JSON válido, sin texto adicional ni backticks,
con este formato exacto:
{{"orden": ["palabra_mas_relacionada", "palabra_siguiente", "..."]}}
"""
