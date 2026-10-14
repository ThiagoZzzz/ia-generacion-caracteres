from pathlib import Path

# raíz del proyecto (donde está main.py)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
# directorio de datos
DATA_DIR = ROOT_DIR / "data"
# rutas a archivos del texto
RAW_DATA_PATH = DATA_DIR / "raw" / "The_Festival_raw.txt"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "The_Festival_clean.txt"