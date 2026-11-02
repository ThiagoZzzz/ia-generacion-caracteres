from src.data.preprocess  import read_text
import tensorflow as tf
from tensorflow.keras.layers import StringLookup

# Tokenizar - Generar dos diccionarios de búsqueda: convierte cada caracter del vocabulario en un Id numérico y vicersa, obtiene el caracter al que pertenece cada id
class Tokenizer:
    def __init__(self, vocabulary: list[str]) -> None:
        # capa/función transformadora: asignar a cada cada caracter su id
        self._ids_from_chars = StringLookup(
            vocabulary=vocabulary, mask_token=None
        )
        # capa/función transformadora: encontrar el caracter asigando para el id X
        self._chars_from_ids = StringLookup(
            vocabulary=self._ids_from_chars.get_vocabulary(),
            invert=True,
            mask_token=None
        )

    # codificar: text_raw -> tokens -> ids
    def encode(self, text: str) -> tf.Tensor:
        tokens = tf.strings.unicode_split(text, 'UTF-8')
        return self._ids_from_chars(tokens)

    # decodificar: ids -> caracteres -> unir caracteres con reduce_join (formar palabras)
    def decode(self, ids: tf.Tensor) -> tf.Tensor:
        return tf.strings.reduce_join(
            self._chars_from_ids(ids), axis=-1
        )
    
    def get_vocabulary(self) -> list[str]:
        return self._ids_from_chars.get_vocabulary()


# genera el vocabulario ordenado y crea la instancia Tokenizer para acceder a las transformaciones sobre el texto
def build_tokenizer(text: str) -> Tokenizer:
    vocab = sorted(set(text))
    print(f"[tokenizer] Vocabulario ordenado: {vocab} \n[tokenizer] Vocabulario con {len(set(vocab))} caracteres únicos.")

    return Tokenizer(vocabulary=vocab)