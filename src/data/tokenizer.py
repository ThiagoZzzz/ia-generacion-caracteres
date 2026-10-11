from src.data.preprocess  import read_text

text = read_text();
# obtenemos ocurrencias únicas de cada caracter en el texto (crear set) y lo ordenamos (sorted)
vocabulary = sorted(set(text))
print(f"Vocabulario ordenado: {vocabulary} \n {len(set(vocabulary))} caracteres únicos.")

# Tokenizar - Generar dos diccionarios de búsqueda: convierte cada caracter del vocabulario en un Id numérico y vicersa, obtiene el caracter al que pertenece cada id

# capas/funciones transformadoras: asignar a cada cada caracter su id
ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocabulary), mask_token=None)
# encontrar el caracter asigando para el id X
chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(),
    invert=True,
    mask_token=None
)

# procesar todo el texto
all_tokens = tf.strings.unicode_split(text, 'UTF-8')

all_ids = ids_from_chars(all_tokens)

print(f"All tokens: {all_tokens}")

def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)