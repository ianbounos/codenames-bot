"""
Palabras funcionales (stopwords) en español: artículos, preposiciones,
pronombres, conjunciones, etc. No sirven como pistas de Codenames porque
no tienen significado semántico propio con el que asociar otras palabras.
"""

STOPWORDS_ES = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "lo",
    "de", "del", "a", "al", "en", "por", "para", "con", "sin", "sobre",
    "entre", "hasta", "desde", "hacia", "durante", "mediante", "segun",
    "y", "o", "u", "e", "ni", "pero", "aunque", "sino", "porque", "pues",
    "que", "cual", "cuales", "quien", "quienes", "cuyo", "cuya",
    "yo", "tu", "el", "ella", "nosotros", "nosotras", "vosotros", "ellos",
    "ellas", "me", "te", "se", "nos", "os", "le", "les", "mi", "tu", "su",
    "mis", "tus", "sus", "nuestro", "nuestra", "vuestro", "vuestra",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "aquel", "aquella", "aquellos", "aquellas", "esto", "eso", "aquello",
    "es", "son", "fue", "fueron", "ser", "estar", "esta", "estan",
    "haber", "hay", "habia", "he", "has", "ha", "hemos", "han",
    "no", "si", "muy", "mas", "menos", "tan", "tanto", "como",
    "cuando", "donde", "adonde", "mientras", "asi", "tambien",
    "todo", "toda", "todos", "todas", "algo", "alguien", "alguno",
    "alguna", "algunos", "algunas", "ninguno", "ninguna", "nada", "nadie",
    "otro", "otra", "otros", "otras", "mismo", "misma", "mismos", "mismas",
    "cada", "cierto", "cierta", "varios", "varias", "poco", "pocos",
    "mucho", "muchos", "mucha", "muchas",
}
