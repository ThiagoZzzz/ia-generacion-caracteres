from tensorflow import Tensor
from src.data.tokenizer import Tokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

def plot_splits_summary(train_ids: Tensor, dev_ids: Tensor, test_ids: Tensor):
    """Gráfico de barras con la proporción de tokens en cada split."""
    counts = [len(train_ids), len(dev_ids), len(test_ids)]
    labels = ['Train (80%)', 'Dev (10%)', 'Test (10%)']
    colors = ['#2b5c8f', '#e67e22', '#27ae60']

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    
    ax = sns.barplot(x=labels, y=counts, palette=colors, hue=labels, legend=False)
    ax.set_title("Cantidad de Tokens por Partición", fontsize=14, fontweight='bold')
    ax.set_xlabel("Partición", fontsize=12)
    ax.set_ylabel("Total de IDs", fontsize=12)
    
    for i, v in enumerate(counts):
        ax.text(i, v + 200, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def plot_char_frequencies(ids: Tensor, tokenizer: Tokenizer, top_n: int = 20):
    """Histograma / gráfico de barras con los caracteres/IDs más frecuentes."""
    counts = Counter(ids.numpy() if hasattr(ids, 'numpy') else ids)
    common = counts.most_common(top_n)
    
    char_ids = [c[0] for c in common]
    frequencies = [c[1] for c in common]
    
    # Decodificar el carácter para mostrarlo en el eje X
    labels = []
    for cid in char_ids:
        raw_char = tokenizer.decode([cid]).numpy().decode('utf-8')
        if raw_char == ' ':
            labels.append("'espacio'")
        elif raw_char == '\n':
            labels.append(r"'\n'")
        else:
            labels.append(f"'{raw_char}'")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 5))
    ax = sns.barplot(x=labels, y=frequencies, color='#34495e')
    ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel("Carácter", fontsize=12)
    ax.set_ylabel("Frecuencia", fontsize=12)
    ax.set_title(f"Top {top_n} caracteres / IDs más frecuentes", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_word_lengths(
    data: list | np.ndarray,
    title: str = "Distribución de longitud de palabras",
    show_kde: bool = False
) -> None:
    """Gráfico de la distribución de la longitud de palabras + promedio y media."""
    
    if data is None or len(data) == 0:
        print("[plots] No hay datos disponibles para graficar.")
        return

    # Extraer longitudes según el tipo de datos recibido (lista de palpabras, lista de palpabras estructurada (lista de diccionarios) o array de longitudes numéricas)
    first_elem = data[0]
    if isinstance(first_elem, dict) and "length" in first_elem:
        lengths = np.array([item["length"] for item in data])
    elif isinstance(first_elem, str):
        lengths = np.array([len(item) for item in data])
    else:
        lengths = np.array(data)

    mean_val = float(np.mean(lengths))
    median_val = float(np.median(lengths))

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histograma con KDE para observar la distribución
    sns.histplot(
        lengths,
        discrete=True,
        color="#2b5c8f",
        edgecolor="#1b3c5d",
        alpha=0.65,
        kde=show_kde
    )
    # Escala eje X, marca cada longitud entera
    ax.set_xticks(range(1, int(lengths.max()) + 1))
    # Línea vertical señalando el promedio de todos los datos
    ax.axvline(
        mean_val,
        color="#e74c3c",
        linestyle="--",
        linewidth=2.5,
        label=f"Promedio: {mean_val:.2f} caracteres"
    )
    # Línea vertical señalando la mediana como referencia adicional
    ax.axvline(
        median_val,
        color="#e8d120",
        linestyle=":",
        linewidth=2,
        label=f"Mediana: {median_val:.1f} caracteres"
    )

    # Anotación visual para el promedio
    y_max = ax.get_ylim()[1]
    ax.text(
        mean_val + 0.35,
        y_max * 0.85,
        f"Promedio = {mean_val:.2f}",
        color="#c0392b",
        fontweight="bold",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#e74c3c", alpha=0.9)
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Longitud de la palabra (cantidad de caracteres)", fontsize=12)
    ax.set_ylabel("Frecuencia (cantidad de palabras)", fontsize=12)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    
    plt.tight_layout()
    plt.show()

def plot_word_frequencies(
    words: list[str] | list[dict],
    top_n: int = 20,
    case_sensitive: bool = False,
    title: str = None
) -> None:
    """Gráfico de barras con la frecuencia de las palabras extraídas del texto."""

    if not words:
        print("[plots] No hay palabras disponibles para graficar.")
        return

    # Extraer las palabras si se pasa una lista de diccionarios (dataset)
    if isinstance(words[0], dict) and "word" in words[0]:
        raw_words = [item["word"] for item in words]
    else:
        raw_words = words

    # Normalizar a minúsculas si no es sensible a mayúsculas
    processed_words = [w if case_sensitive else w.lower() for w in raw_words]

    # Conteo de frecuencias
    counts = Counter(processed_words)
    common = counts.most_common(top_n) if top_n else counts.most_common()

    if not common:
        print("[plots] No se encontraron palabras para graficar.")
        return

    labels = [item[0] for item in common]
    frequencies = [item[1] for item in common]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    ax = sns.barplot(x=labels, y=frequencies, color="#2b5c8f")
    ax.tick_params(axis="x", rotation=45, labelsize=10)

    # Mostrar la cantidad exacta sobre cada barra
    y_offset = max(frequencies) * 0.015
    for i, freq in enumerate(frequencies):
        ax.text(i, freq + y_offset, str(freq), ha="center", fontsize=9, fontweight="bold")

    plot_title = title or (f"Top {len(labels)} Palabras más frecuentes" if top_n else "Frecuencia de palabras")
    ax.set_title(plot_title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Palabra", fontsize=12)
    ax.set_ylabel("Frecuencia (Apariciones)", fontsize=12)

    plt.tight_layout()
    plt.show()
