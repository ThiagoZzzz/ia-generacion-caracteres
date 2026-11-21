import random
import numpy as np
import tensorflow as tf
from src.data.dataset import create_dataset, create_words_dataset
from src.data.preprocess import clean_txt, extract_words, read_text
from src.data.tokenizer import build_tokenizer
from src.models.lstm import (
    build_lstm_model,
    build_pretrained_embedding_matrix,
    generate_text,
)
from src.training.train import (
    evaluate_model,
    extract_hard_examples,
    train_model,
)
from src.utils.config import (
    CHECKPOINTS_DIR,
    EVALUATION_CONFIG,
    MODEL_CONFIGS,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    SEED,
    STAGE_2_REFERENCE,
    TRAINING_CONFIG,
)
from src.utils.plots import (
    format_comparison_table,
    format_hard_examples_table,
    plot_char_frequencies,
    plot_perplexity_comparison,
    plot_splits_summary,
    plot_training_curves,
    plot_word_frequencies,
    plot_word_lengths,
)
from src.utils.types import TrainResult


def set_global_seeds(seed: int = SEED) -> None:
    """Fija las semillas aleatorias para garantizar reproducibilidad entre corridas."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main() -> None:
    print("=" * 20 + " INICIANDO PIPELINE " + "=" * 20)
    set_global_seeds(SEED)

    # -------------------------------------------------------------
    # 0. INGESTA Y LIMPIEZA DE DATOS (Data)
    # -------------------------------------------------------------
    print("\n### Etapa 0 - Ingesta y limpieza de datos...")

    clean_txt(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    text = read_text(PROCESSED_DATA_PATH)
    words = extract_words(text)

    # -------------------------------------------------------------
    # 1. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
    # -------------------------------------------------------------
    print("\n### Etapa 1 - Análisis exploratorio de datos (EDA)...")

    ds_words = create_words_dataset(words)

    print(f"[main] -> Total de caracteres:     {len(text):,}")
    print(f"[main] -> Total de palabras:       {len(words):,}")
    print(f"[main] -> Vocabulario (palabras):  {len(set(words)):,}")

    # Visualizaciones del corpus
    plot_word_frequencies(words, top_n=20, title="Top 20 palabras más frecuentes")
    plot_word_lengths(ds_words, title="Distribución de longitud de palabras")

    # -------------------------------------------------------------
    # 2. TOKENIZACIÓN (Text encoding)
    # -------------------------------------------------------------
    print("\n### Etapa 2 - Tokenización (char-level)...")
    tokenizer = build_tokenizer(text)
    vocab_size = len(tokenizer.get_vocabulary())
    ids = tokenizer.encode(text)

    # Visualización de la distribución del vocabulario de tokens/caracteres
    plot_char_frequencies(ids, tokenizer, top_n=20)

    # -------------------------------------------------------------
    # 3. PARTICIÓN DE DATOS (Data split)
    # -------------------------------------------------------------
    print("\n### Etapa 3 - Partición de datos (Train / Dev / Test)...")
    text_tokens = len(ids)
    train_end = int(text_tokens * 0.80)
    dev_end = int(text_tokens * 0.90)

    # Partición secuencial contigua (sin shuffle previo para preservar orden temporal)
    train_ids = ids[:train_end]
    dev_ids = ids[train_end:dev_end]
    test_ids = ids[dev_end:]

    print(f"[main] -> Total tokens:  {len(ids):,}")
    print(f"[main] -> Train  (80%):  {len(train_ids):,}")
    print(f"[main] -> Dev    (10%):  {len(dev_ids):,}")
    print(f"[main] -> Test   (10%):  {len(test_ids):,}")

    # Visualización de distribución de tokens en cada set del split
    plot_splits_summary(train_ids, dev_ids, test_ids)

    # -------------------------------------------------------------
    # 4. PREPROCESAMIENTO (Data transform)
    # -------------------------------------------------------------
    print("\n### Etapa 4 - Preprocesamiento y creación de datasets (tf.data)...")
    # Genera secuencias de contexto (x: t_0..t_n, y: t_1..t_n+1) y batching
    ds_train = create_dataset(train_ids, shuffle=True)
    ds_dev = create_dataset(dev_ids, shuffle=False)
    ds_test = create_dataset(test_ids, shuffle=False)

    # Ejemplo de entrenamiento: x -> y
    for input_example, target_example in ds_train.take(1):
        sample_input = tokenizer.decode(input_example[0]).numpy().decode("utf-8", errors="replace")
        sample_target = tokenizer.decode(target_example[0]).numpy().decode("utf-8", errors="replace")
        print(f"[main] -> Ejemplo X (Input):  {repr(sample_input)}")
        print(f"[main] -> Ejemplo Y (Target): {repr(sample_target)}")

    # -------------------------------------------------------------
    # 5. MODELADO LSTM Y COMPARATIVA DE EMBEDDINGS (Etapa 3 del TP)
    # -------------------------------------------------------------
    print("\n### Etapa 5 - Entrenamiento y comparación de arquitecturas LSTM...")

    results_dict: dict[str, TrainResult] = {}
    trained_models: dict[str, tf.keras.Model] = {}
    pretrained_matrices_cache: dict[int, np.ndarray] = {}

    for model_key, model_cfg in MODEL_CONFIGS.items():
        print(f"\n--- Entrenando configuración: [{model_key}] ---")
        print(f"Descripción: {model_cfg['description']}")

        pretrained_matrix = None
        if model_cfg["use_pretrained"]:
            emb_dim = model_cfg["embedding_dim"]
            if emb_dim not in pretrained_matrices_cache:
                pretrained_matrices_cache[emb_dim] = build_pretrained_embedding_matrix(
                    vocabulary=tokenizer.get_vocabulary(),
                    embedding_dim=emb_dim,
                )
            pretrained_matrix = pretrained_matrices_cache[emb_dim]

        model = build_lstm_model(
            vocab_size=vocab_size,
            config=model_cfg,
            pretrained_matrix=pretrained_matrix,
            model_name=model_key,
        )

        ckpt_file = CHECKPOINTS_DIR / f"{model_key}.weights.h5"
        train_result = train_model(
            model=model,
            train_ds=ds_train,
            dev_ds=ds_dev,
            config=model_cfg,
            training_config=TRAINING_CONFIG,
            checkpoint_path=ckpt_file,
            log_every_n_epochs=10
        )

        results_dict[model_key] = train_result
        trained_models[model_key] = model

        print(
            f"[{model_key}] Mejor época: {train_result['best_epoch']} | "
            f"Val Loss: {train_result['best_val_loss']:.4f} | "
            f"Val PPL: {train_result['best_val_perplexity']:.2f} | "
            f"Val Acc: {train_result['best_val_accuracy'] * 100.0:.2f}%"
        )

    # 5.1 Tabla comparativa integral y gráficos de curvas de entrenamiento
    format_comparison_table(results_dict, stage2_reference=STAGE_2_REFERENCE)
    plot_training_curves(results_dict)
    plot_perplexity_comparison(results_dict, stage2_reference=STAGE_2_REFERENCE)

    # 5.2 Selección del mejor modelo según la métrica de número único (menor Perplejidad en Dev)
    best_model_key = min(results_dict, key=lambda k: results_dict[k]["best_val_perplexity"])
    best_model = trained_models[best_model_key]
    print(f"\n[main] -> Mejor modelo seleccionado en Dev: '{best_model_key}'")

    # 5.3 Evaluación final sobre el conjunto ciego de Test y reporte de mejora vs Etapa 2
    test_metrics = evaluate_model(
        model=best_model,
        dataset=ds_test,
        stage2_reference=STAGE_2_REFERENCE,
        verbose=0,
    )
    print("\n--- Resultados Finales en Conjunto de Test ---")
    print(f"  Test Loss:                {test_metrics['loss']:.4f}")
    print(f"  Test Perplexity (Única):  {test_metrics['perplexity']:.2f}")
    print(f"  Test BPC:                 {test_metrics['bpc']:.3f}")
    print(f"  Test Accuracy:            {test_metrics['accuracy'] * 100.0:.2f}%")
    if STAGE_2_REFERENCE["val_perplexity"] > 0:
        print(
            f"  Mejora PPL vs Etapa 2:    {test_metrics['perplexity_reduction_abs']:+.2f} puntos "
            f"({test_metrics['perplexity_reduction_pct']:+.2f}%)"
        )

    # 5.4 Auditoría de errores: extracción de >= 10 ejemplos concretos difíciles
    hard_examples = extract_hard_examples(
        model=best_model,
        dataset=ds_test,
        tokenizer=tokenizer,
        num_examples=EVALUATION_CONFIG["hard_examples_count"],
        context_window=EVALUATION_CONFIG["hard_examples_context"],
    )
    format_hard_examples_table(hard_examples)

    # 5.5 Generación de muestras de texto literario variando la temperatura
    print("\n--- Generación de Texto Literario con el Mejor Modelo ---")
    for temp in EVALUATION_CONFIG["generation_temperatures"]:
        generated_sample = generate_text(
            model=best_model,
            tokenizer=tokenizer,
            seed_text=EVALUATION_CONFIG["generation_seed_text"],
            num_chars=EVALUATION_CONFIG["generation_num_chars"],
            temperature=temp,
        )
        print(f"\n[Temperatura T = {temp}]:\n{generated_sample}\n" + "-" * 60)

    print("=" * 20 + " PIPELINE EJECUTADO CORRECTAMENTE " + "=" * 20)


if __name__ == "__main__":
    main()
