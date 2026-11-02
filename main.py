from src.data import tokenizer
from src.utils.config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.data.preprocess import clean_txt, read_text
from src.data.tokenizer import build_tokenizer
from src.data.dataset import create_dataset
from src.utils.plots import plot_splits_summary, plot_char_frequencies

def main() -> None:
    print("[main] Iniciando pipeline...");
    
    clean_txt(RAW_DATA_PATH, PROCESSED_DATA_PATH);

    text = read_text(PROCESSED_DATA_PATH);

    tokenizer = build_tokenizer(text);
    ids = tokenizer.encode(text);

    text_tokens = len(ids)

    # 1. Definir los índices de corte (80% Train, 10% Dev, 10% Test)
    train_end = int(text_tokens * 0.80)
    dev_end = int(text_tokens * 0.90)

    # 2. Partición contigua del tensor
    train_ids = ids[:train_end]
    dev_ids = ids[train_end:dev_end]
    test_ids = ids[dev_end:]

    print(f'[main] Total Ids: {len(ids)}')
    print(f'[main] Train Ids: {len(train_ids)}')
    print(f'[main] Dev Ids:   {len(dev_ids)}')
    print(f'[main] Test Ids:  {len(test_ids)}')

    ds_train = create_dataset(train_ids);

    # Ejemplo de entrenamiento, x -> y
    for input_example, target_example in ds_train.take(1):
        print(f"[main] Input:  {tokenizer.decode(input_example[0]).numpy().decode('utf-8')}\n[main] Target: {tokenizer.decode(target_example[0]).numpy().decode('utf-8')}")

    plot_splits_summary(train_ids, dev_ids, test_ids)
    plot_char_frequencies(ids, tokenizer, top_n=20)

    print("[main] Fin del pipeline.");

if __name__ == "__main__":
    main()