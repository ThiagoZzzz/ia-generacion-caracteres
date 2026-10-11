import re
from pathlib import Path

def clean_txt(input_path: Path, output_path: Path) -> None:
    if (Path(str(output_path)).exists()):
        print (f"[preprocess] El texto limpio ya existe en '{output_path}'.")
        return

    # abrir raw text
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. delimitación del contenido útil (Gutenberg header/footer + epígrafe)
    start_pattern = r"--Lactantius\."
    end_pattern = r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*"

    start_match = re.search(start_pattern, text)
    end_match = re.search(end_pattern, text)

    start_idx = start_match.end() if start_match else 0
    end_idx = end_match.start() if end_match else len(text)

    clean_text = text[start_idx:end_idx].strip()

    # 2. normalización 
    clean_text = clean_text.replace("\r\n", "\n")

    # eliminar notas del transcriptor 
    clean_text = re.sub(r"\[Transcriber’s Note:.*?\]", "", clean_text, flags=re.DOTALL)

    # eliminar separadores de escena con asteriscos (*   *   *   *   *)
    clean_text = re.sub(r"\n\s*(?:\*\s*){3,}\n", "\n\n", clean_text)

    # normalizar cursivas de Gutenberg (_palabra_ -> palabra)
    clean_text = re.sub(r"_([^_]+)_", r"\1", clean_text)

    # estandarizar comillas y caracteres tipográficos
    clean_text = clean_text.replace("“", '"').replace("”", '"')
    clean_text = clean_text.replace("‘", "'").replace("’", "'")
    clean_text = clean_text.replace("--", "—")

    # permitir letras, números, saltos de línea y puntuación usual; el resto de caracteres son borrados
    clean_text = re.sub(r"[^a-zA-Z0-9\s.,;:!?\'\"—\-]", "", clean_text)

    # limpiar espacios redundantes por línea y saltos de línea excesivos
    clean_text = re.sub(r"[ \t]+", " ", clean_text)
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
    clean_text = clean_text.strip()

    # guardar cuerpo limpio
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(clean_text)

    print("[preprocess] Limpieza finalizada.")
    print(f"[preprocess] Caracteres antes: {len(text)} | Caracteres conservados: {len(clean_text)}.")

def read_text (text_path: str) -> Path:
    if (not Path(text_path)).exists():
        print (f"[preprocess] El archivo con el texto limpio no existe en {text_path}, pruebe crearlo primero.")
        return
    
    return Path(text_path).read_text(encoding='utf-8')
