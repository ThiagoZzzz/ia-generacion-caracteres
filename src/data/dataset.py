from tensorflow.data import Dataset
import tensorflow as tf

seq_length = 100;
BATCH_SIZE = 64;
BUFFER_SIZE = 100000;

def create_dataset(ids_tensor, shuffle=False) -> Dataset:
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