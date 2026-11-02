from tensorflow import Tensor
from tensorflow.data import Dataset
import tensorflow as tf

# ventana de contexto (cantidad de caracteres de la secuencia de entrada)
seq_length = 100;
# tamaño de los bloques de entrenamiento
BATCH_SIZE = 16;
# memoria asignada para tomar muestras al azar (si es que se requiere)
BUFFER_SIZE = 100000;

def create_dataset(ids_tensor: Tensor, shuffle=False) -> Dataset:
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