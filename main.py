from pathlib import Path

# FASE 0 - INGESTA DE DATOS
BASE_DIR = Path(__file__).resolve().parent
db_path = BASE_DIR / 'db' / 'The_Festival.txt'

text = db_path.read_text(encoding='utf-8')
print(f'Longitud del texto: {len(text)} caracteres.\n')

# obtenemos ocurrencias únicas de cada caracter en el texto (crear set) y lo ordenamos (sorted)
vocabulary = sorted(set(text))
print(f"Vocabulario ordenado: {vocabulary} \n {len(set(vocabulary))} caracteres únicos.")