"""
Vocabulario de palabras para el TABLERO (las 25 que aparecen en juego).

Criterio de selección: sustantivos concretos, de uso muy común, que un
hablante de español promedio reconoce sin dudar. Evitamos:
  - palabras abstractas (amor, libertad) -> son más difíciles de "anclar"
    en un embedding de forma consistente
  - términos muy técnicos o regionales
  - palabras con múltiples significados muy distintos entre sí (ambigüedad
    excesiva complica la evaluación, aunque un poco de ambigüedad es
    parte natural del juego)

IMPORTANTE: estas palabras llevan tildes correctas donde corresponde.
Se comprobó que sin tilde, algunas palabras (ej. "leon" vs "león") dan
vectores de embedding muy distintos o de mala calidad -- el modelo las
trata como si fueran otra palabra (o una desconocida), no como una
variante ortográfica de la misma. Por eso mantenemos la ortografía real.

Esta lista es un punto de partida (~180 palabras) inspirada en las
categorías del Codenames original: animales, objetos cotidianos,
lugares, naturaleza, tecnología, cuerpo humano, comida, etc.
Se puede (y conviene) ampliar con el tiempo.
"""

VOCABULARIO_TABLERO = [
    # Animales
    "gato", "perro", "caballo", "elefante", "león", "tigre", "oso", "lobo",
    "águila", "serpiente", "araña", "abeja", "ballena", "tiburón", "delfín",
    "pulpo", "rana", "mono", "conejo", "ratón", "vaca", "cerdo", "oveja",
    "gallina", "pato", "pingüino", "cocodrilo", "camaleón", "murciélago",

    # Objetos cotidianos / muebles
    "silla", "mesa", "sofá", "cama", "lámpara", "espejo", "reloj", "puerta",
    "ventana", "libro", "lápiz", "papel", "teléfono", "computadora", "televisión",
    "cámara", "llave", "tijera", "cepillo", "jabón", "toalla", "paraguas",
    "mochila", "maleta", "cartera", "anillo", "collar",

    # Lugares
    "playa", "montaña", "bosque", "desierto", "isla", "río", "lago", "océano",
    "ciudad", "pueblo", "castillo", "iglesia", "escuela", "hospital", "aeropuerto",
    "estación", "puente", "torre", "jardín", "parque", "granja", "cueva",
    "volcán", "selva",

    # Naturaleza / clima
    "sol", "luna", "estrella", "planeta", "nube", "lluvia", "nieve", "viento",
    "trueno", "arcoíris", "fuego", "hielo", "arena", "roca", "flor", "árbol",
    "hoja", "semilla", "raíz",

    # Tecnología / transporte
    "robot", "cohete", "satélite", "barco", "submarino", "avión", "tren",
    "motocicleta", "helicóptero", "globo",

    # Cuerpo humano
    "mano", "pie", "ojo", "oreja", "corazón", "cerebro", "hueso", "sangre",
    "diente", "cabello",

    # Comida
    "manzana", "banana", "naranja", "uva", "fresa", "limón", "pan", "queso",
    "leche", "huevo", "arroz", "sopa", "pastel", "chocolate", "café", "té",
    "azúcar", "sal", "miel",

    # Profesiones / personas
    "médico", "maestro", "policía", "bombero", "piloto", "cocinero", "artista",
    "músico", "científico", "granjero", "rey", "reina", "princesa", "pirata",
    "soldado", "mago",

    # Deportes / juegos
    "pelota", "cancha", "raqueta", "bicicleta", "medalla", "trofeo", "dado",
    "carta", "ajedrez",

    # Música / arte
    "guitarra", "piano", "tambor", "violín", "pincel", "pintura", "escultura",

    # Fantasía / mitología
    "dragón", "fantasma", "vampiro", "bruja", "hada", "monstruo", "gigante",
    "unicornio", "sirena",
]

# Sanity check simple: sin duplicados
assert len(VOCABULARIO_TABLERO) == len(set(VOCABULARIO_TABLERO)), \
    "Hay palabras duplicadas en el vocabulario del tablero"
