from tensorflow import Tensor
from tensorflow.data import Dataset
import tensorflow as tf
from src.utils.config import BATCH_SIZE, BUFFER_SIZE, SEQ_LENGTH

# Alias de compatibilidad para módulos y notebooks que importan seq_length
seq_length = SEQ_LENGTH


def create_dataset(ids_tensor: Tensor, shuffle: bool = False) -> Dataset:
    # Crear flujo de datos desde los IDs
    ds = Dataset.from_tensor_slices(ids_tensor)

    # Agrupar en fragmentos de tamaño seq_length + 1
    sequences = ds.batch(seq_length + 1, drop_remainder=True)

    # Separar entrada (X) y etiqueta (y)
    # Ejemplo: 'Hola' -> input: 'Hol', target: 'ola'
    def split_input_target(sequence):
        return sequence[:-1], sequence[1:]

    dataset = sequences.map(split_input_target)

    if shuffle:
        dataset = dataset.shuffle(BUFFER_SIZE)

    return dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


def create_words_dataset(words: list[str]) -> list[dict]:
    dataset = [
        {
            "word": word,
            "length": len(word),
        }
        for word in words
    ]

    print(f"[dataset] Dataset de palabras generado con {len(dataset)} registros.")
    return dataset
