from tensorflow import Tensor
from tensorflow.data import Dataset
import tensorflow as tf
from src.utils.config import BATCH_SIZE, BUFFER_SIZE, SEQ_LENGTH, TRAIN_WINDOW_STRIDE
from src.utils.types import validate_positive_int

# Alias de compatibilidad para módulos y notebooks que importan seq_length
seq_length = SEQ_LENGTH


def create_dataset(
    ids_tensor: Tensor,
    shuffle: bool = False,
    stride: int | None = None,
) -> Dataset:
    """
    Construye un `tf.data.Dataset` de pares autorregresivos `(X, Y)` de longitud `SEQ_LENGTH`.

    - Cuando `stride` es menor que `SEQ_LENGTH + 1` (por defecto `TRAIN_WINDOW_STRIDE` si `shuffle=True`),
      aplica una ventana deslizante con solapamiento para multiplicar los contextos de entrenamiento.
    - En validación y test (`shuffle=False`), usa por defecto ventanas disjuntas (`SEQ_LENGTH + 1`)
      para evaluar cada token exactamente una vez sin sesgos por repetición.
    """
    window_size = seq_length + 1
    effective_stride = stride if stride is not None else (TRAIN_WINDOW_STRIDE if shuffle else window_size)
    validate_positive_int(effective_stride, "stride", context="create_dataset")

    ds = Dataset.from_tensor_slices(ids_tensor)

    if effective_stride == window_size:
        sequences = ds.batch(window_size, drop_remainder=True)
    else:
        sequences = ds.window(window_size, shift=effective_stride, drop_remainder=True).flat_map(
            lambda window: window.batch(window_size)
        )

    def split_input_target(sequence: Tensor) -> tuple[Tensor, Tensor]:
        return sequence[:-1], sequence[1:]

    dataset = sequences.map(split_input_target, num_parallel_calls=tf.data.AUTOTUNE).cache()

    if shuffle:
        dataset = dataset.shuffle(BUFFER_SIZE)

    # drop_remainder=False en el lote externo para no descartar secuencias de validación/test
    return dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)


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
