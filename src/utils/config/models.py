from src.utils.types.config_types import ModelConfig, Stage2Reference

# Valores de referencia de la Etapa 2 (MLP / Red multicapa previa) para calcular
# la diferencia de la métrica de número único (Perplejidad y Loss).
STAGE_2_REFERENCE: Stage2Reference = {
    "model_name": "Etapa 2 - MLP (3 capas ocultas)",
    "val_loss": 0.0,
    "val_perplexity": 0.0,
    "val_accuracy": 0.0,
}

# Diccionario de configuraciones de modelos (Etapa 3 - LSTM)
# - use_pretrained: False (inicialización aleatoria desde cero) vs True (matriz pre-entrenada)
# - trainable_embedding: True (pesos optimizables en backpropagation) vs False (congelados)
MODEL_CONFIGS: dict[str, ModelConfig] = {
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
