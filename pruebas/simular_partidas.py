"""
SIMULACIÓN MASIVA: corre un torneo todos-contra-todos entre distintos
perfiles de bot, y genera un dashboard con las estadísticas.

Perfiles:
  pista1_decoder1 / pista1_decoder2 / pista2_decoder1 / pista2_decoder2:
      las 4 combinaciones cruzadas entre Spymaster v1 (solo spaCy) o v2
      (doble validación con veto estricto), y Operative v1 o v2 (híbrido)
  v3_ensamble: Spymaster + Operative usando un ENSAMBLE de 3 modelos
      (spaCy + 2 sentence-transformers) combinados con PROMEDIO PONDERADO
      en vez de veto estricto (ver engine/embedding_ensemble.py) -- pensado
      para evitar el problema de v2, que resultó demasiado lento por ser
      excesivamente conservador.

  Los perfiles que dependen de sentence-transformers solo se incluyen si
  la librería está instalada y logra descargar los modelos; si no, corre
  solo con pista1_decoder1.

Uso:
    python pruebas/simular_partidas.py
"""
import sys
import random
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.embeddings import EmbeddingStore
from engine.spymaster import SpymasterBot
from engine.operative import OperativeBot
from engine.simulacion import PerfilBot, simular_lote
from data.vocabulario_tablero import VOCABULARIO_TABLERO

LISTA_FRECUENCIA = Path(__file__).parent.parent / "data" / "es_50k.txt"
RESULTADOS_DIR = Path(__file__).parent.parent / "results"
RESULTADOS_DIR.mkdir(exist_ok=True)

N_JUEGOS_POR_MATCHUP = 50  # ajustable -- con varios perfiles ya son varios matchups; empezamos con 50 para que no tarde una eternidad


def cargar_perfiles() -> dict[str, PerfilBot]:
    print("Cargando modelo principal (spaCy)...")
    kwargs = dict(palabras_incluir=VOCABULARIO_TABLERO)
    if LISTA_FRECUENCIA.exists():
        kwargs["lista_frecuencia_path"] = str(LISTA_FRECUENCIA)
        kwargs["max_candidatas"] = 50_000
    store_principal = EmbeddingStore.from_spacy_model(**kwargs)
    print(f"  {len(store_principal)} palabras\n")

    def construir_sm_v1(vocab_pistas: list[str]) -> SpymasterBot:
        return SpymasterBot(
            embeddings=store_principal,
            vocabulario_pistas=vocab_pistas,
            beta_asesino=0.05,
            beta_resto=0.02,
        )

    operative_v1 = OperativeBot(embeddings=store_principal)

    perfiles: dict[str, PerfilBot] = {
        # nombre: "pista_X_decoder_Y" -- deja clarísimo qué combinación es cada uno
        "pista1_decoder1": PerfilBot(
            nombre="pista1_decoder1",
            vocab_completo=store_principal.words,
            construir_spymaster=construir_sm_v1,
            operative=operative_v1,
        ),
    }

    try:
        print("Intentando cargar modelo secundario (sentence-transformers)...")
        from sentence_transformers import SentenceTransformer
        from engine.embedding_hibrido import EmbeddingHibrido
        from engine.operative_hibrido import OperativeBotHibrido

        modelo_raw = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        store_secundario = EmbeddingStore.from_sentence_transformer(
            words=store_principal.words,
        )
        print(f"  {len(store_secundario)} palabras precalculadas en el modelo secundario\n")

        def construir_sm_v2(vocab_pistas: list[str]) -> SpymasterBot:
            return SpymasterBot(
                embeddings=store_principal,
                vocabulario_pistas=vocab_pistas,
                beta_asesino=0.05,
                beta_resto=0.02,
                embeddings_secundario=store_secundario,
                beta_asesino_secundario=0.05,
                beta_resto_secundario=0.02,
            )

        hibrido = EmbeddingHibrido(store_principal, store_secundario, modelo_raw)
        operative_v2 = OperativeBotHibrido(hibrido=hibrido)

        # Las 4 combinaciones cruzadas: Spymaster v1 o v2, Operative v1 o v2,
        # independientes entre sí -- esto es lo que permite comparar, por
        # ejemplo, si el Operative v2 (híbrido) decodifica mejor o peor
        # las pistas de un Spymaster v1, y viceversa.
        perfiles["pista1_decoder2"] = PerfilBot(
            nombre="pista1_decoder2",
            vocab_completo=store_principal.words,
            construir_spymaster=construir_sm_v1,
            operative=operative_v2,
        )
        perfiles["pista2_decoder1"] = PerfilBot(
            nombre="pista2_decoder1",
            vocab_completo=store_principal.words,
            construir_spymaster=construir_sm_v2,
            operative=operative_v1,
        )
        perfiles["pista2_decoder2"] = PerfilBot(
            nombre="pista2_decoder2",
            vocab_completo=store_principal.words,
            construir_spymaster=construir_sm_v2,
            operative=operative_v2,
        )
    except Exception as e:
        print(f"  No se pudo cargar el modelo secundario ({e}). Corriendo solo con pista1_decoder1.\n")
        store_secundario = None

    # --- v3: ensamble de 3 modelos con promedio ponderado (en vez del
    # veto estricto de v2) -- ver engine/embedding_ensemble.py ---
    if store_secundario is not None:
        try:
            from engine.embedding_ensemble import EmbeddingEnsemble

            print("Cargando tercer modelo para v3 (mpnet, más grande, puede tardar más)...")
            store_terciario = EmbeddingStore.from_sentence_transformer(
                words=store_principal.words,
                model_name="paraphrase-multilingual-mpnet-base-v2",
            )
            print(f"  {len(store_terciario)} palabras precalculadas en el tercer modelo\n")

            ensamble_v3 = EmbeddingEnsemble([
                ("spacy", store_principal, 1.0),
                ("minilm", store_secundario, 1.0),
                ("mpnet", store_terciario, 1.0),
            ])

            def construir_sm_v3(vocab_pistas: list[str]) -> SpymasterBot:
                return SpymasterBot(
                    embeddings=ensamble_v3,
                    vocabulario_pistas=vocab_pistas,
                    beta_asesino=0.05,
                    beta_resto=0.02,
                )

            # v3 no necesita ninguna clase especial de Operative -- el
            # ensamble se hace pasar por un EmbeddingStore normal
            perfiles["v3_ensamble"] = PerfilBot(
                nombre="v3_ensamble",
                vocab_completo=store_principal.words,
                construir_spymaster=construir_sm_v3,
                operative=OperativeBot(embeddings=ensamble_v3),
            )
        except Exception as e:
            print(f"  No se pudo cargar v3 ({e}). Sigo sin ese perfil.\n")

    return perfiles


def main():
    perfiles = cargar_perfiles()
    print(f"Perfiles disponibles: {list(perfiles.keys())}\n")

    print(f"Corriendo torneo ({N_JUEGOS_POR_MATCHUP} partidas por matchup)...")
    resultados = simular_lote(
        vocabulario_tablero=VOCABULARIO_TABLERO,
        perfiles=perfiles,
        n_juegos_por_matchup=N_JUEGOS_POR_MATCHUP,
        seed_base=1000,
    )
    print(f"\nTotal de partidas simuladas: {len(resultados)}\n")

    # Guardar CSV crudo
    import csv
    csv_path = RESULTADOS_DIR / "resultados_simulacion.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "perfil_rojo", "perfil_azul", "equipo_inicial", "ganador", "razon",
            "equipo_toco_asesino", "turnos_totales",
            "pistas_forzadas_rojo", "pistas_forzadas_azul", "seed",
        ])
        for r in resultados:
            writer.writerow([
                r.perfil_rojo, r.perfil_azul, r.equipo_inicial, r.ganador, r.razon,
                r.equipo_toco_asesino, r.turnos_totales,
                r.pistas_forzadas_rojo, r.pistas_forzadas_azul, r.seed,
            ])
    print(f"CSV guardado en: {csv_path}\n")

    # Resumen de texto rápido en consola
    print("=" * 60)
    print("RESUMEN RÁPIDO")
    print("=" * 60)

    razones = Counter(r.razon for r in resultados)
    print(f"\nCómo terminaron las partidas: {dict(razones)}")

    ganadores_por_asesino = Counter(
        r.equipo_toco_asesino for r in resultados if r.razon == "asesino"
    )
    print(f"Quién tocó el asesino (y por lo tanto perdió): {dict(ganadores_por_asesino)}")

    gano_iniciando = sum(1 for r in resultados if r.ganador == r.equipo_inicial)
    print(f"\nGanó el equipo que arrancó: {gano_iniciando}/{len(resultados)} "
          f"({100*gano_iniciando/len(resultados):.1f}%)")

    print(f"\nTurnos promedio por partida: "
          f"{sum(r.turnos_totales for r in resultados)/len(resultados):.1f}")

    print("\nGenerando dashboard visual...")
    sys.path.insert(0, str(Path(__file__).parent))
    from generar_dashboard import generar_dashboard
    generar_dashboard(resultados, list(perfiles.keys()), RESULTADOS_DIR)


if __name__ == "__main__":
    main()
