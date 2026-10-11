import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_TXT_PATH = BASE_DIR / 'data' / 'raw' / 'The_Festival_raw.txt'

PROCESSED_TXT_PATH = BASE_DIR / 'data' / 'processed' / 'The_Festival_cleaned.txt'

def clean_txt(input_path: Path, output_path: Path) -> None:
    if (Path(str(PROCESSED_TXT_PATH)).exists()):
        print (f"[preprocess] El texto limpio ya existe en ${PROCESSED_TXT_PATH}.")
        return

    # abrir raw text
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    start_pattern = r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*"
    end_pattern = r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*"

    start_match = re.search(start_pattern, text)
    end_match = re.search(end_pattern, text)

    start_idx = start_match.end() if start_match else 0
    end_idx = end_match.start() if end_match else len(text)

    # realizar recorte y normalización del cuerpo del texto
    clean_text = text[start_idx:end_idx].strip()
    clean_text = clean_text.replace("\r\n", "\n")

    # guardar cuerpo limpio
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(clean_text)

    print("[preprocess] Limpieza finalizada.")
    print(f"[preprocess] Caracteres antes: {len(text)} | Caracteres conservados: {len(clean_text)}")

def read_text () -> Path:
    if (not Path(str(PROCESSED_TXT_PATH)).exists()):
        print (f"[preprocess] El archivo con el texto limpio no existe en ${PROCESSED_TXT_PATH}, pruebe crearlo primero.")
        return
    
    return PROCESSED_TXT_PATH.read_text(encoding='utf-8')

if __name__ == "__main__":
    clean_txt(RAW_TXT_PATH, PROCESSED_TXT_PATH)
