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
    
    # Gráfico de barras con Seaborn
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
