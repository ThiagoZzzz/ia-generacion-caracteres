import time
import numpy as np
import tensorflow as tf
from src.models.baseline import build_kgram_xy
from src.utils.config import SEED


def build_one_hot_flat(x_mat: np.ndarray, vocab_size: int) -> np.ndarray:
    """Convierte una matriz (N, K) de IDs en vectores One-Hot aplanados (N, K * vocab_size)."""
    n, k = x_mat.shape
    oh = np.eye(vocab_size, dtype=np.float32)[x_mat]
    return oh.reshape(n, k * vocab_size)


def build_mlp_model(
    k: int = 5,
    vocab_size: int = 66,
    hidden_units: tuple[int, ...] = (128, 64, 32),
    dropout_rate: float = 0.05,
    l2_reg: float = 0.0,
    model_name: str = "mlp_stage2",
) -> tf.keras.Model:
    """
    Construye el MLP de 3 capas ocultas (128 -> 64 -> 32, ReLU) de la Etapa 2
    sobre la representación One-Hot aplanada de los últimos K caracteres.
    """
    reg = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0.0 else None
    layers: list[tf.keras.layers.Layer] = [tf.keras.layers.Input(shape=(k * vocab_size,))]

    for idx, units in enumerate(hidden_units, start=1):
        layers.append(
            tf.keras.layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=reg,
                name=f"dense_hidden_{idx}",
            )
        )
        if dropout_rate > 0.0:
            layers.append(tf.keras.layers.Dropout(dropout_rate, name=f"dropout_{idx}"))

    layers.append(
        tf.keras.layers.Dense(
            vocab_size,
            activation="softmax",
            kernel_regularizer=reg,
            name="probs_output",
        )
    )
    return tf.keras.Sequential(layers, name=model_name)


def train_evaluate_mlp(
    train_ids: np.ndarray,
    dev_ids: np.ndarray,
    vocab_size: int,
    k: int = 5,
    dropout_rate: float = 0.05,
    l2_reg: float = 0.0,
    learning_rate: float = 5e-4,
    epochs: int = 100,
    batch_size: int = 128,
    use_early_stopping: bool = False,
    patience: int = 10,
    seed: int = SEED,
    model_name: str = "mlp_dropout_005",
) -> dict:
    """
    Entrena y evalúa el MLP de la Etapa 2 sobre la partición 80/10/10, reportando
    tanto las métricas en el punto de menor pérdida de validación (best_val_perplexity)
    como al final del entrenamiento.
    """
    tf.keras.utils.set_random_seed(seed)

    x_train, y_train = build_kgram_xy(train_ids, k=k)
    x_dev, y_dev = build_kgram_xy(dev_ids, k=k)

    x_train_oh = build_one_hot_flat(x_train, vocab_size=vocab_size)
    x_dev_oh = build_one_hot_flat(x_dev, vocab_size=vocab_size)

    model = build_mlp_model(
        k=k,
        vocab_size=vocab_size,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
        model_name=model_name,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    cbs: list[tf.keras.callbacks.Callback] = []
    if use_early_stopping:
        cbs.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )
        )

    t0 = time.perf_counter()
    history = model.fit(
        x_train_oh,
        y_train,
        validation_data=(x_dev_oh, y_dev),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cbs,
        verbose=0,
    )
    elapsed = time.perf_counter() - t0

    val_losses = [float(v) for v in history.history["val_loss"]]
    val_accs = [float(v) for v in history.history["val_accuracy"]]
    train_accs = [float(v) for v in history.history["accuracy"]]

    best_loss_idx = int(np.argmin(val_losses))
    best_val_loss = float(val_losses[best_loss_idx])
    best_val_ppl = float(np.exp(best_val_loss))
    best_val_bpc = float(best_val_loss / np.log(2.0))
    best_val_acc = float(val_accs[best_loss_idx])

    final_val_loss = float(val_losses[-1])
    final_val_ppl = float(np.exp(final_val_loss))
    final_val_acc = float(val_accs[-1])
    peak_val_acc = float(np.max(val_accs))
    peak_acc_epoch = int(np.argmax(val_accs)) + 1

    return {
        "model_name": model_name,
        "k": k,
        "dropout": dropout_rate,
        "l2_reg": l2_reg,
        "learning_rate": learning_rate,
        "epochs_trained": len(val_losses),
        "best_epoch": best_loss_idx + 1,
        "trainable_params": int(model.count_params()),
        "training_time_s": float(elapsed),
        "train_accuracy_best": float(train_accs[best_loss_idx]),
        "train_accuracy_final": float(train_accs[-1]),
        "best_val_loss": best_val_loss,
        "best_val_perplexity": best_val_ppl,
        "best_val_bpc": best_val_bpc,
        "best_val_accuracy": best_val_acc,
        "final_val_loss": final_val_loss,
        "final_val_perplexity": final_val_ppl,
        "final_val_accuracy": final_val_acc,
        "peak_val_accuracy": peak_val_acc,
        "peak_acc_epoch": peak_acc_epoch,
    }
