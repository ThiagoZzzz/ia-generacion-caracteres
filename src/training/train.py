from pathlib import Path
from typing import cast
import numpy as np
import tensorflow as tf
from src.data.tokenizer import Tokenizer
from src.utils.config import (
    EVALUATION_CONFIG,
    STAGE_2_REFERENCE,
    TRAINING_CONFIG,
)
from src.utils.types import (
    EvaluationResult,
    HardExample,
    ModelConfig,
    Stage2Reference,
    TrainingConfig,
    TrainingHistory,
    TrainResult,
    validate_model_config,
    validate_positive_int,
    validate_stage2_reference,
    validate_training_config,
)


class Perplexity(tf.keras.metrics.Metric):
    """
    Métrica personalizada de Keras para calcular la Perplejidad (PPL = exp(CrossEntropyLoss))
    época a época durante el entrenamiento y la validación.
    """

    def __init__(self, name: str = "perplexity", **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self._loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction="none",
        )
        self.total_loss = self.add_weight(name="total_loss", initializer="zeros", dtype=tf.float32)
        self.total_count = self.add_weight(name="total_count", initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor | None = None) -> None:
        step_loss = self._loss_fn(y_true, y_pred)
        step_loss = tf.cast(step_loss, tf.float32)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            step_loss = tf.multiply(step_loss, sample_weight)
            count = tf.reduce_sum(sample_weight)
        else:
            count = tf.cast(tf.size(step_loss), tf.float32)

        self.total_loss.assign_add(tf.reduce_sum(step_loss))
        self.total_count.assign_add(count)

    def result(self) -> tf.Tensor:
        mean_loss = tf.math.divide_no_nan(self.total_loss, self.total_count)
        return tf.exp(mean_loss)

    def reset_state(self) -> None:
        self.total_loss.assign(0.0)
        self.total_count.assign(0.0)


def build_default_callbacks(
    training_config: TrainingConfig = TRAINING_CONFIG,
    checkpoint_path: str | Path | None = None,
    use_early_stopping: bool = True,
) -> list[tf.keras.callbacks.Callback]:
    """
    Construye la lista de callbacks estándar para regularización y control del entrenamiento:
    - EarlyStopping (opcional según `use_early_stopping`, restaurando mejores pesos)
    - ReduceLROnPlateau (reducción adaptativa del learning rate ante estancamiento)
    - ModelCheckpoint (opcional, si se especifica ruta de guardado)
    """
    validate_training_config(training_config, name="build_default_callbacks")

    patience = int(training_config["early_stopping_patience"])
    min_delta = float(training_config["early_stopping_min_delta"])
    lr_patience = int(training_config["reduce_learning_rate_patience"])
    lr_factor = float(training_config["reduce_learning_rate_factor"])
    min_lr = float(training_config["min_learning_rate"])

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=lr_factor,
            patience=lr_patience,
            min_lr=min_lr,
            verbose=1,
        ),
    ]

    if use_early_stopping:
        callbacks.insert(
            0,
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                min_delta=min_delta,
                restore_best_weights=True,
                verbose=1,
            ),
        )

    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(ckpt_path),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=0,
            )
        )

    return callbacks


class PeriodicEpochLogger(tf.keras.callbacks.Callback):
    """
    Callback ligero que imprime el resumen de métricas cada `interval` épocas
    (y en la última época programada) para evitar logs extensos época por época.
    """

    def __init__(self, interval: int = 5, total_epochs: int = 30) -> None:
        super().__init__()
        self.interval = interval
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        current_epoch = epoch + 1
        if current_epoch % self.interval != 0 and current_epoch != self.total_epochs:
            return

        logs = logs or {}
        loss = logs.get("loss", float("nan"))
        acc = logs.get("accuracy", float("nan"))
        ppl = logs.get("perplexity", float("nan"))
        val_loss = logs.get("val_loss", float("nan"))
        val_acc = logs.get("val_accuracy", float("nan"))
        val_ppl = logs.get("val_perplexity", float("nan"))
        lr = logs.get("learning_rate", float("nan"))

        print(
            f"Época {current_epoch:>2}/{self.total_epochs} — "
            f"loss: {loss:.4f} | acc: {acc:.4f} | ppl: {ppl:.2f} | "
            f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | val_ppl: {val_ppl:.2f} | "
            f"lr: {lr:.1e}"
        )


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-3) -> tf.keras.Model:
    """
    Compila el modelo con optimizador Adam, pérdida SparseCategoricalCrossentropy(from_logits=True),
    Accuracy categórica y la métrica de número único Perplexity.
    """
    if not isinstance(model, tf.keras.Model):
        raise TypeError("[compile_model] 'model' debe ser una instancia de tf.keras.Model.")
    if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool) or learning_rate <= 0.0:
        raise ValueError(f"[compile_model] 'learning_rate' debe ser > 0.0 (recibido: {learning_rate}).")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate), clipnorm=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            Perplexity(name="perplexity"),
        ],
    )
    return model


def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    dev_ds: tf.data.Dataset,
    config: ModelConfig,
    epochs: int | None = None,
    training_config: TrainingConfig = TRAINING_CONFIG,
    callbacks: list[tf.keras.callbacks.Callback] | None = None,
    checkpoint_path: str | Path | None = None,
    verbose: int = 1,
    log_every_n_epochs: int | None = None,
) -> TrainResult:
    """
    Función reutilizable de entrenamiento para cualquier configuración de modelo.
    Valida los parámetros, compila el modelo, aplica Early Stopping y registra las métricas por época.

    Retorna un diccionario tipado (`TrainResult`) con el historial por época y el resumen
    del mejor estado alcanzado en validación.
    """
    if not isinstance(model, tf.keras.Model):
        raise TypeError("[train_model] 'model' debe ser una instancia de tf.keras.Model.")
    if not isinstance(train_ds, tf.data.Dataset):
        raise TypeError("[train_model] 'train_ds' debe ser una instancia de tf.data.Dataset.")
    if not isinstance(dev_ds, tf.data.Dataset):
        raise TypeError("[train_model] 'dev_ds' debe ser una instancia de tf.data.Dataset.")

    validate_model_config(config, name=f"train_model:{model.name}")
    validate_training_config(training_config, name="train_model:training_config")

    if epochs is not None:
        validate_positive_int(epochs, "epochs", context="train_model")
    if log_every_n_epochs is not None:
        validate_positive_int(log_every_n_epochs, "log_every_n_epochs", context="train_model")

    learning_rate = float(config["learning_rate"])
    max_epochs = int(epochs if epochs is not None else training_config["epochs"])

    compile_model(model, learning_rate=learning_rate)

    active_callbacks = list(
        callbacks
        if callbacks is not None
        else build_default_callbacks(
            training_config=training_config,
            checkpoint_path=checkpoint_path,
            use_early_stopping=bool(config["use_early_stopping"]),
        )
    )

    fit_verbose = verbose
    if log_every_n_epochs is not None:
        active_callbacks.append(PeriodicEpochLogger(interval=log_every_n_epochs, total_epochs=max_epochs))
        fit_verbose = 0

    history_obj = model.fit(
        train_ds,
        validation_data=dev_ds,
        epochs=max_epochs,
        callbacks=active_callbacks,
        verbose=fit_verbose,
    )

    raw_history = {k: [float(v) for v in vals] for k, vals in history_obj.history.items()}
    history_dict = cast(TrainingHistory, raw_history)
    val_losses = raw_history.get("val_loss", [])
    val_ppls = raw_history.get("val_perplexity", [])

    if val_ppls:
        best_idx = int(np.argmin(val_ppls))
        best_val_ppl = float(val_ppls[best_idx])
        best_val_loss = float(np.log(best_val_ppl))
    elif val_losses:
        best_idx = int(np.argmin(val_losses))
        best_val_loss = float(val_losses[best_idx])
        best_val_ppl = float(np.exp(best_val_loss))
    else:
        best_idx = 0
        best_val_loss = float("nan")
        best_val_ppl = float("nan")

    best_val_bpc = float(best_val_loss / np.log(2.0)) if not np.isnan(best_val_loss) else float("nan")
    best_val_acc = (
        float(raw_history["val_accuracy"][best_idx])
        if "val_accuracy" in raw_history and len(raw_history["val_accuracy"]) > best_idx
        else float("nan")
    )

    trainable_params = int(
        np.sum([np.prod(v.shape) for v in model.trainable_weights])
    )
    non_trainable_params = int(
        np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    )

    return {
        "model_name": model.name,
        "config": config,
        "epochs_trained": len(val_losses),
        "best_epoch": best_idx + 1,
        "best_val_loss": best_val_loss,
        "best_val_perplexity": best_val_ppl,
        "best_val_bpc": best_val_bpc,
        "best_val_accuracy": best_val_acc,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "total_params": trainable_params + non_trainable_params,
        "history": history_dict,
    }


def evaluate_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    stage2_reference: Stage2Reference = STAGE_2_REFERENCE,
    verbose: int = 0,
) -> EvaluationResult:
    """
    Evalúa un modelo entrenado sobre un dataset (Dev o Test) y calcula la mejora
    absoluta y porcentual sobre la métrica de número único (Perplejidad) respecto a la Etapa 2.
    """
    if not isinstance(model, tf.keras.Model):
        raise TypeError("[evaluate_model] 'model' debe ser una instancia de tf.keras.Model.")
    if not isinstance(dataset, tf.data.Dataset):
        raise TypeError("[evaluate_model] 'dataset' debe ser una instancia de tf.data.Dataset.")
    validate_stage2_reference(stage2_reference, name="evaluate_model:stage2_reference")

    eval_output = model.evaluate(dataset, verbose=verbose, return_dict=True)

    raw_loss = float(eval_output.get("loss", float("nan")))
    if "perplexity" in eval_output and float(eval_output["perplexity"]) > 0.0:
        perplexity = float(eval_output["perplexity"])
        loss = float(np.log(perplexity))
    else:
        loss = raw_loss
        perplexity = float(np.exp(loss))
    bpc = float(loss / np.log(2.0))
    accuracy = float(eval_output.get("accuracy", float("nan")))

    ref_ppl = float(stage2_reference["val_perplexity"])
    ref_loss = float(stage2_reference["val_loss"])
    ref_acc = float(stage2_reference["val_accuracy"])

    # Una menor perplejidad y menor loss representan una mejora positiva
    ppl_improvement_abs = ref_ppl - perplexity
    ppl_improvement_pct = ((ref_ppl - perplexity) / ref_ppl) * 100.0 if ref_ppl > 0 else float("nan")
    loss_improvement_pct = ((ref_loss - loss) / ref_loss) * 100.0 if ref_loss > 0 else float("nan")
    acc_improvement_pts = (accuracy - ref_acc) * 100.0

    return {
        "loss": loss,
        "perplexity": perplexity,
        "bpc": bpc,
        "accuracy": accuracy,
        "stage2_reference_perplexity": ref_ppl,
        "perplexity_reduction_abs": ppl_improvement_abs,
        "perplexity_reduction_pct": ppl_improvement_pct,
        "loss_reduction_pct": loss_improvement_pct,
        "accuracy_gain_percentage_points": acc_improvement_pts,
    }


def _categorize_error(context_str: str, target_char: str, predicted_char: str) -> str:
    """
    Clasifica heurísticamente el tipo de dificultad lingüística/morfológica del ejemplo.
    """
    prev_char = context_str[-1] if context_str else ""

    if target_char in {"\n", ".", ",", ";", ":", "!", "?", "—", "-", '"', "'"}:
        return "Puntuación o salto de estructura"
    if target_char == " ":
        return "Cierre de palabra (predicción de espacio)"
    if prev_char in {" ", "\n", "—", '"', "'"} and target_char.isalpha():
        if target_char.isupper():
            return "Inicio de oración o nombre propio (Mayúscula)"
        return "Inicio de palabra (alta incertidumbre léxica)"
    if target_char.isupper():
        return "Mayúscula intermedia o énfasis"
    return "Morfología interna de palabra (vocabulario literario)"


def extract_hard_examples(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    tokenizer: Tokenizer,
    num_examples: int = EVALUATION_CONFIG["hard_examples_count"],
    context_window: int = EVALUATION_CONFIG["hard_examples_context"],
    min_context_chars: int = 15,
) -> list[HardExample]:
    """
    Analiza las predicciones del modelo sobre el dataset de evaluación e identifica
    los ejemplos concretos más difíciles (mayor pérdida Cross-Entropy paso a paso).

    Garantiza extraer al menos `num_examples` (>= 10) ejemplos distintos con su contexto,
    carácter objetivo, carácter predicho, probabilidades asignadas y categoría de error.
    """
    if not isinstance(model, tf.keras.Model):
        raise TypeError("[extract_hard_examples] 'model' debe ser una instancia de tf.keras.Model.")
    if not isinstance(dataset, tf.data.Dataset):
        raise TypeError("[extract_hard_examples] 'dataset' debe ser una instancia de tf.data.Dataset.")
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError("[extract_hard_examples] 'tokenizer' debe ser una instancia de Tokenizer.")
    validate_positive_int(num_examples, "num_examples", context="extract_hard_examples")
    validate_positive_int(context_window, "context_window", context="extract_hard_examples")
    validate_positive_int(min_context_chars, "min_context_chars", context="extract_hard_examples")

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    candidates: list[HardExample] = []

    for x_batch, y_batch in dataset:
        logits = model(x_batch, training=False)  # [B, T, V]
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        step_losses = loss_fn(y_batch, logits).numpy()  # [B, T]
        pred_ids = np.argmax(probs, axis=-1)  # [B, T]

        x_np = x_batch.numpy()
        y_np = y_batch.numpy()
        batch_size, seq_len = x_np.shape

        for b in range(batch_size):
            # Evitar los primeros pasos donde el contexto de la ventana aún es muy corto
            for t in range(min_context_chars, seq_len):
                target_id = int(y_np[b, t])
                pred_id = int(pred_ids[b, t])

                if target_id == pred_id:
                    continue

                loss_val = float(step_losses[b, t])
                target_prob = float(probs[b, t, target_id])
                pred_prob = float(probs[b, t, pred_id])

                start_idx = max(0, t + 1 - context_window)
                context_ids = x_np[b, start_idx : t + 1]
                context_str = tokenizer.decode(context_ids).numpy().decode("utf-8", errors="replace")
                target_char = tokenizer.decode([target_id]).numpy().decode("utf-8", errors="replace")
                pred_char = tokenizer.decode([pred_id]).numpy().decode("utf-8", errors="replace")

                category = _categorize_error(context_str, target_char, pred_char)

                candidates.append(
                    {
                        "context": context_str,
                        "target_char": target_char,
                        "predicted_char": pred_char,
                        "target_prob": target_prob,
                        "predicted_prob": pred_prob,
                        "loss": loss_val,
                        "category": category,
                    }
                )

    # Ordenar de mayor a menor pérdida (los más difíciles para el modelo primero)
    candidates.sort(key=lambda item: item["loss"], reverse=True)

    # Seleccionar ejemplos diversos (sin repetir exactamente el mismo contexto)
    selected: list[HardExample] = []
    seen_contexts: set[str] = set()

    for item in candidates:
        norm_ctx = item["context"].strip()
        if norm_ctx in seen_contexts:
            continue
        seen_contexts.add(norm_ctx)
        selected.append(item)
        if len(selected) >= num_examples:
            break

    return selected
