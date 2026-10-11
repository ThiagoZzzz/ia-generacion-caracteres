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