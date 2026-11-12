from src.utils.types.config_types import EvaluationConfig, TrainingConfig

# Semilla global de reproducibilidad
SEED = 42

# Parámetros de ventana temporal y pipeline tf.data
SEQ_LENGTH = 100
BATCH_SIZE = 16
BUFFER_SIZE = 100_000

# Parámetros generales del bucle de entrenamiento y regularización
TRAINING_CONFIG: TrainingConfig = {
    "epochs": 30,
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 1e-3,
    "reduce_learning_rate_patience": 3,
    "reduce_learning_rate_factor": 0.5,
    "min_learning_rate": 1e-5,
    "pretrain_concurrence_window": 5,  # Ventana de contexto para embeddings pre-entrenados
}

# Parámetros para inferencia / generación de texto y auditoría de errores
EVALUATION_CONFIG: EvaluationConfig = {
    "hard_examples_count": 10,       # Mínimo de ejemplos difíciles exigidos
    "hard_examples_context": 40,     # Cantidad de caracteres previos a mostrar como contexto
    "generation_temperatures": [0.3, 0.7, 1.1],
    "generation_num_chars": 200,
    "generation_seed_text": "The ancient town of Kingsport",
}
