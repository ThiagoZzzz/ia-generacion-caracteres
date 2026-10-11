from pathlib import Path
import tensorflow as tf

# FASE 0 - INGESTA DE DATOS
BASE_DIR = Path(__file__).resolve().parent
db_path = BASE_DIR / 'data' / 'The_Festival.txt'

text = db_path.read_text(encoding='utf-8')
print(f'Longitud del texto: {len(text)} caracteres.\n')

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

# partición de datos
# Total de tokens del texto ya vectorizado
n_tokens = len(all_ids)

# 1. Definir los índices de corte (80% Train, 10% Dev, 10% Test)
train_end = int(n_tokens * 0.80)
dev_end = int(n_tokens * 0.90)

# 2. Partición contigua del tensor
train_ids = all_ids[:train_end]
dev_ids = all_ids[train_end:dev_end]
test_ids = all_ids[dev_end:]

print(f"Tokens Train: {len(train_ids)}")
print(f"Tokens Dev:   {len(dev_ids)}")
print(f"Tokens Test:  {len(test_ids)}")

seq_length = 100
BATCH_SIZE = 64
BUFFER_SIZE = 10000

def create_dataset(ids_tensor, shuffle=False):
    # Crear flujo de datos desde los IDs
    ds = tf.data.Dataset.from_tensor_slices(ids_tensor)
    
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

# Generar los 3 datasets
train_dataset = create_dataset(train_ids, shuffle=True)
dev_dataset   = create_dataset(dev_ids, shuffle=False)
test_dataset  = create_dataset(test_ids, shuffle=False)
