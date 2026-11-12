from pathlib import Path

# Raíz del proyecto (donde está main.py)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Directorio de datos y rutas a archivos del texto
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "The_Festival_raw.txt"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "The_Festival_clean.txt"

# Directorio para almacenar checkpoints de modelos y resultados de experimentos
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
