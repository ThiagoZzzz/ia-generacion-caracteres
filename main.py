from src.data.preprocess import clean_txt
from src.utils.config import RAW_DATA_PATH, PROCESSED_DATA_PATH

def main():
    print("Iniciando pipeline...")
    
    clean_txt(RAW_DATA_PATH, PROCESSED_DATA_PATH)

if __name__ == "__main__":
    main()