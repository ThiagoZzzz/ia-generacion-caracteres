from pathlib import Path

# 1. RUTAS DEL PROYECTO Y DIRECTORIOS
# Raíz del proyecto (donde está main.py)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Directorio de datos y rutas a archivos del texto
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "The_Festival_raw.txt"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "The_Festival_clean.txt"

# Directorio para almacenar checkpoints de modelos y resultados de experimentos
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"


# 2. HIPERPARÁMETROS GLOBALES DE DATOS Y ENTRENAMIENTO
SEED = 42

# Parámetros de ventana temporal y pipeline tf.data
SEQ_LENGTH = 100
BATCH_SIZE = 16
BUFFER_SIZE = 100_000

# Parámetros generales del bucle de entrenamiento y regularización
TRAINING_CONFIG = {
    "epochs": 30,
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 1e-3,
    "reduce_learning_rate_patience": 3,
    "reduce_learning_rate_factor": 0.5,
    "min_learning_rate": 1e-5,
    "pretrain_concurrence_window": 5,  # Ventana de contexto para embeddings pre-entrenados
}

# Parámetros para inferencia / generación de texto y auditoría de errores
EVALUATION_CONFIG = {
    "hard_examples_count": 10,       # Mínimo de ejemplos difíciles exigidos
    "hard_examples_context": 40,     # Cantidad de caracteres previos a mostrar como contexto
    "generation_temperatures": [0.3, 0.7, 1.1],
    "generation_num_chars": 200,
    "generation_seed_text": "The ancient town of Kingsport",
}


# 3. MÉTRICAS DE REFERENCIA — ETAPA 2 (REPORTE DE MEJORA)

# Valores de referencia de la Etapa 2 (MLP / Red multicapa previa) para calcular la diferencia de la métrica de número único (Perplejidad y Loss).
STAGE_2_REFERENCE = {
    "model_name": "Etapa 2 - MLP (3 capas ocultas)",
    "val_loss": 0,
    "val_perplexity": 0,
    "val_accuracy": 0,
}


# 4. DICCIONARIO DE CONFIGURACIONES DE MODELOS (ETAPA 3 - LSTM)

# Permite iterar sobre distintas arquitecturas y estrategias de embeddings:
# - use_pretrained: False (inicialización aleatoria desde cero) vs True (matriz pre-entrenada)
# - trainable_embedding: True (pesos optimizables en backpropagation) vs False (congelados)
MODEL_CONFIGS = {
    "lstm_baseline": {
        "description": "LSTM base con embeddings entrenados desde cero y sin dropout",
        "embedding_dim": 64,
        "rnn_units": 128,
        "dense_units": 64,
        "dropout": 0.0,
        "recurrent_dropout": 0.0,
        "use_pretrained": False,
        "trainable_embedding": True,
        "learning_rate": 1e-3,
    },
    "lstm_regularized": {
        "description": "LSTM de mayor capacidad con embeddings desde cero y regularización (Dropout)",
        "embedding_dim": 128,
        "rnn_units": 256,
        "dense_units": 128,
        "dropout": 0.2,
        "recurrent_dropout": 0.1,
        "use_pretrained": False,
        "trainable_embedding": True,
        "learning_rate": 5e-4,
    },
    "lstm_pretrained_frozen": {
        "description": "LSTM con embeddings pre-entrenados congelados (trainable=False)",
        "embedding_dim": 64,
        "rnn_units": 128,
        "dense_units": 64,
        "dropout": 0.2,
        "recurrent_dropout": 0.0,
        "use_pretrained": True,
        "trainable_embedding": False,
        "learning_rate": 1e-3,
    },
    "lstm_pretrained_finetuned": {
        "description": "LSTM con embeddings pre-entrenados y ajuste fino (trainable=True)",
        "embedding_dim": 64,
        "rnn_units": 128,
        "dense_units": 64,
        "dropout": 0.2,
        "recurrent_dropout": 0.0,
        "use_pretrained": True,
        "trainable_embedding": True,
        "learning_rate": 5e-4,
    },
}
