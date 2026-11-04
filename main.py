from src.utils.config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.data.preprocess import clean_txt, read_text, extract_words
from src.data.tokenizer import build_tokenizer
from src.data.dataset import create_dataset, create_words_dataset
from src.utils.plots import (
    plot_splits_summary,
    plot_char_frequencies,
    plot_word_lengths,
    plot_word_frequencies,
)

def main() -> None:
    print("=" *20 + " INICIANDO PIPELINE " + "=" *20);

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
    
    print(f"[main] -> Total de caracteres: {len(text):,}")
    print(f"[main] -> Total de palabras:   {len(words):,}")
    print(f"[main] -> Vocabulario (palabras):  {len(set(words)):,}")

    # Visualizaciones del corpus
    plot_word_frequencies(words, top_n=20, title="Top 20 palabras más frecuentes")
    plot_word_lengths(ds_words, title="Distribución de longitud de palabras")

    # -------------------------------------------------------------
    # 2. TOKENIZACIÓN (Text encoding)
    # -------------------------------------------------------------
    print("\n### Etapa 2 - Tokenización (char-level)...")
    tokenizer = build_tokenizer(text)
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
        sample_input = tokenizer.decode(input_example[0]).numpy().decode('utf-8', errors='replace')
        sample_target = tokenizer.decode(target_example[0]).numpy().decode('utf-8', errors='replace')
        print(f"[main] -> Ejemplo X (Input):  {repr(sample_input)}")
        print(f"[main] -> Ejemplo Y (Target): {repr(sample_target)}")

    print("=" *20 + " PIPELINE EJECUTADO CORRECTAMENTE " + "=" *20);

if __name__ == "__main__":
    main()