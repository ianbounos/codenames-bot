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

Esta lista es un punto de partida (~150 palabras) inspirada en las
categorías del Codenames original: animales, objetos cotidianos,
lugares, naturaleza, tecnología, cuerpo humano, comida, etc.
Se puede (y conviene) ampliar con el tiempo.
"""

VOCABULARIO_TABLERO = [
    # Animales
    "gato", "perro", "caballo", "elefante", "leon", "tigre", "oso", "lobo",
    "aguila", "serpiente", "araña", "abeja", "ballena", "tiburon", "delfin",
    "pulpo", "rana", "mono", "conejo", "raton", "vaca", "cerdo", "oveja",
    "gallina", "pato", "pinguino", "cocodrilo", "camaleon", "murcielago",

    # Objetos cotidianos / muebles
    "silla", "mesa", "sofa", "cama", "lampara", "espejo", "reloj", "puerta",
    "ventana", "libro", "lapiz", "papel", "telefono", "computadora", "television",
    "camara", "llave", "tijera", "cepillo", "jabon", "toalla", "paraguas",
    "mochila", "maleta", "cartera", "anillo", "collar",

    # Lugares
    "playa", "montana", "bosque", "desierto", "isla", "rio", "lago", "oceano",
    "ciudad", "pueblo", "castillo", "iglesia", "escuela", "hospital", "aeropuerto",
    "estacion", "puente", "torre", "jardin", "parque", "granja", "cueva",
    "volcan", "selva",

    # Naturaleza / clima
    "sol", "luna", "estrella", "planeta", "nube", "lluvia", "nieve", "viento",
    "trueno", "arcoiris", "fuego", "hielo", "arena", "roca", "flor", "arbol",
    "hoja", "semilla", "raiz",

    # Tecnología / transporte
    "robot", "cohete", "satelite", "barco", "submarino", "avion", "tren",
    "motocicleta", "helicoptero", "globo",

    # Cuerpo humano
    "mano", "pie", "ojo", "oreja", "corazon", "cerebro", "hueso", "sangre",
    "diente", "cabello",

    # Comida
    "manzana", "banana", "naranja", "uva", "fresa", "limon", "pan", "queso",
    "leche", "huevo", "arroz", "sopa", "pastel", "chocolate", "cafe", "te",
    "azucar", "sal", "miel",

    # Profesiones / personas
    "medico", "maestro", "policia", "bombero", "piloto", "cocinero", "artista",
    "musico", "cientifico", "granjero", "rey", "reina", "princesa", "pirata",
    "soldado", "mago",

    # Deportes / juegos
    "pelota", "cancha", "raqueta", "bicicleta", "medalla", "trofeo", "dado",
    "carta", "ajedrez",

    # Música / arte
    "guitarra", "piano", "tambor", "violin", "pincel", "pintura", "escultura",

    # Fantasía / mitología
    "dragon", "fantasma", "vampiro", "bruja", "hada", "monstruo", "gigante",
    "unicornio", "sirena",
]

# Sanity check simple: sin duplicados
assert len(VOCABULARIO_TABLERO) == len(set(VOCABULARIO_TABLERO)), \
    "Hay palabras duplicadas en el vocabulario del tablero"
