import numpy as np
import tensorflow as tf
from src.data.tokenizer import Tokenizer
from src.utils.config import SEQ_LENGTH, TRAINING_CONFIG
from src.utils.types import (
    ModelConfig,
    validate_model_config,
    validate_positive_int,
)


def build_pretrained_embedding_matrix(
    train_ids: tf.Tensor | np.ndarray,
    vocab_size: int,
    embedding_dim: int,
    window_size: int = TRAINING_CONFIG["pretrain_concurrence_window"],
) -> np.ndarray:
    """
    Construye una matriz de embeddings pre-entrenados a nivel de carácter (|V| x embedding_dim)
    calculada exclusivamente sobre el conjunto de entrenamiento (train_ids) para evitar Data Leakage.

    Utiliza una matriz de co-ocurrencia contextual ponderada por distancia + Positive Pointwise
    Mutual Information (PPMI) y factorización SVD truncada (equivalente matricial a Skip-Gram / Word2Vec).
    """
    validate_positive_int(vocab_size, "vocab_size", context="build_pretrained_embedding_matrix")
    validate_positive_int(embedding_dim, "embedding_dim", context="build_pretrained_embedding_matrix")
    validate_positive_int(window_size, "window_size", context="build_pretrained_embedding_matrix")

    if train_ids is None or len(train_ids) <= window_size:
        raise ValueError(
            "[build_pretrained_embedding_matrix] 'train_ids' debe contener más tokens que 'window_size'."
        )

    ids_array = train_ids.numpy() if hasattr(train_ids, "numpy") else np.asarray(train_ids)
    cooccurrence = np.zeros((vocab_size, vocab_size), dtype=np.float64)

    # 1. Conteo de co-ocurrencias simétricas dentro de la ventana deslizante
    for offset in range(1, window_size + 1):
        weight = 1.0 / offset  # Mayor peso a caracteres más cercanos
        left = ids_array[:-offset]
        right = ids_array[offset:]
        np.add.at(cooccurrence, (left, right), weight)
        np.add.at(cooccurrence, (right, left), weight)

    # 2. Cálculo de PPMI (Positive Pointwise Mutual Information)
    total_sum = np.sum(cooccurrence) + 1e-12
    row_sums = np.sum(cooccurrence, axis=1, keepdims=True) + 1e-12
    col_sums = np.sum(cooccurrence, axis=0, keepdims=True) + 1e-12

    expected = (row_sums @ col_sums) / total_sum
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log2((cooccurrence + 1e-12) / expected)
    ppmi = np.maximum(pmi, 0.0)

    # 3. Reducción de dimensionalidad mediante SVD (U * sqrt(S))
    u_mat, singular_vals, _ = np.linalg.svd(ppmi, full_matrices=False)
    k = min(embedding_dim, vocab_size)
    embeddings = u_mat[:, :k] * np.sqrt(singular_vals[:k])

    # Si embedding_dim supera el tamaño del vocabulario, completar dimensiones restantes
    if embedding_dim > vocab_size:
        rng = np.random.default_rng(42)
        pad = rng.normal(loc=0.0, scale=0.05, size=(vocab_size, embedding_dim - vocab_size))
        embeddings = np.hstack([embeddings, pad])

    # 4. Estandarización a escala compatible con inicializaciones de redes neuronales
    std = np.std(embeddings)
    if std > 0:
        embeddings = (embeddings - np.mean(embeddings)) / std * 0.1

    return embeddings.astype(np.float32)


def build_lstm_model(
    vocab_size: int,
    config: ModelConfig,
    pretrained_matrix: np.ndarray | None = None,
    model_name: str = "lstm_model",
) -> tf.keras.Model:
    """
    Construye la arquitectura secuencial LSTM con capa Embedding integrada y cabezal denso.

    Estructura:
        Input (IDs discretos)
          -> Embedding (desde cero o pre-entrenada, entrenable o congelada)
          -> LSTM (con soporte para dropout y recurrent_dropout)
          -> Dense intermedia (proyección no lineal sobre estados/embeddings) + Dropout opcional
          -> Dense final (logits sobre el vocabulario |V|)
    """
    validate_positive_int(vocab_size, "vocab_size", context=model_name)
    validate_model_config(config, name=model_name)

    embedding_dim = int(config["embedding_dim"])
    rnn_units = int(config["rnn_units"])
    dense_units = int(config["dense_units"])
    dropout_rate = float(config["dropout"])
    recurrent_dropout_rate = float(config["recurrent_dropout"])
    use_pretrained = bool(config["use_pretrained"])
    trainable_embedding = bool(config["trainable_embedding"])

    # 1. Configuración de la capa Embedding dentro de la arquitectura del modelo
    if use_pretrained:
        if pretrained_matrix is None:
            raise ValueError(
                f"[{model_name}] La configuración requiere 'use_pretrained=True', "
                "pero no se proporcionó 'pretrained_matrix'."
            )
        if pretrained_matrix.shape != (vocab_size, embedding_dim):
            raise ValueError(
                f"[{model_name}] Forma de 'pretrained_matrix' inválida: {pretrained_matrix.shape}. "
                f"Se esperaba ({vocab_size}, {embedding_dim})."
            )
        emb_initializer = tf.keras.initializers.Constant(pretrained_matrix)
    else:
        emb_initializer = "glorot_uniform"

    layers = [
        tf.keras.layers.Input(shape=(None,), dtype=tf.int64, name="char_ids_input"),
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            embeddings_initializer=emb_initializer,
            trainable=trainable_embedding,
            name="char_embedding",
        ),
        # 2. Capa recurrente LSTM para modelar dependencias de corto y largo plazo
        tf.keras.layers.LSTM(
            units=rnn_units,
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout_rate,
            recurrent_initializer="glorot_uniform",
            name="recurrent_lstm",
        ),
    ]

    # 3. Cabezal denso intermedio sobre los estados recurrentes
    if dense_units > 0:
        layers.append(
            tf.keras.layers.Dense(
                units=dense_units,
                activation="relu",
                name="dense_hidden",
            )
        )
        if dropout_rate > 0.0:
            layers.append(
                tf.keras.layers.Dropout(
                    rate=dropout_rate,
                    name="dense_dropout",
                )
            )

    # 4. Capa de proyección final a logits del vocabulario
    layers.append(
        tf.keras.layers.Dense(
            units=vocab_size,
            name="logits_output",
        )
    )

    return tf.keras.Sequential(layers, name=model_name)


def generate_text(
    model: tf.keras.Model,
    tokenizer: Tokenizer,
    seed_text: str = "The ancient town of Kingsport",
    num_chars: int = 200,
    temperature: float = 0.7,
    max_context_length: int = SEQ_LENGTH,
) -> str:
    """
    Genera secuencias de texto carácter por carácter de forma autorregresiva
    aplicando muestreo estocástico reescalado por temperatura (T).
    """
    if not isinstance(model, tf.keras.Model):
        raise TypeError("[generate_text] 'model' debe ser una instancia de tf.keras.Model.")
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError("[generate_text] 'tokenizer' debe ser una instancia de Tokenizer.")
    if not isinstance(seed_text, str) or not seed_text:
        raise ValueError("[generate_text] 'seed_text' debe ser un string no vacío.")
    validate_positive_int(num_chars, "num_chars", context="generate_text")
    validate_positive_int(max_context_length, "max_context_length", context="generate_text")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or temperature <= 0.0:
        raise ValueError(f"[generate_text] 'temperature' debe ser > 0.0 (recibido: {temperature}).")

    input_ids = tokenizer.encode(seed_text)
    input_tensor = tf.expand_dims(input_ids, axis=0)  # Forma: [1, seq_len]

    generated_chars: list[str] = []
    safe_temp = max(float(temperature), 1e-5)

    for _ in range(num_chars):
        # Mantener la ventana de contexto acotada a los últimos max_context_length caracteres
        context_window = input_tensor[:, -max_context_length:]

        # Predicción de logits: [1, window_len, vocab_size]
        predictions = model(context_window, training=False)

        # Tomar logits del último paso temporal y escalar por temperatura
        last_step_logits = predictions[:, -1, :] / safe_temp

        # Muestreo categórico sobre la distribución de probabilidad resultante
        sampled_id = tf.random.categorical(last_step_logits, num_samples=1)[-1, 0].numpy()

        # Concatenar el nuevo ID predicho para el siguiente paso autorregresivo
        next_id_tensor = tf.constant([[sampled_id]], dtype=input_tensor.dtype)
        input_tensor = tf.concat([input_tensor, next_id_tensor], axis=-1)

        # Decodificar carácter generado
        decoded_char = tokenizer.decode([sampled_id]).numpy().decode("utf-8", errors="replace")
        generated_chars.append(decoded_char)

    return seed_text + "".join(generated_chars)
