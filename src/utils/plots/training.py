from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.utils.config import MODEL_CONFIGS, STAGE_2_REFERENCE
from src.utils.types import Stage2Reference, TrainResult

# Paleta fija por nombre de modelo para mantener consistencia visual entre gráficos y splits
_DEFAULT_PALETTE = sns.color_palette("tab10", n_colors=10)
DEFAULT_MODEL_COLORS: dict[str, tuple[float, float, float]] = {
    model_key: _DEFAULT_PALETTE[idx % len(_DEFAULT_PALETTE)]
    for idx, model_key in enumerate(MODEL_CONFIGS.keys())
}


def get_model_color_map(
    model_keys: list[str],
    color_map: dict[str, str | tuple[float, float, float]] | None = None,
) -> dict[str, str | tuple[float, float, float]]:
    """
    Devuelve un mapeo determinista etiqueta -> color para que cada modelo conserve
    siempre el mismo color aunque cambie el orden o el subconjunto graficado.
    """
    resolved: dict[str, str | tuple[float, float, float]] = {}
    custom = color_map or {}
    fallback_idx = len(DEFAULT_MODEL_COLORS)

    for key in model_keys:
        if key in custom:
            resolved[key] = custom[key]
        elif key in DEFAULT_MODEL_COLORS:
            resolved[key] = DEFAULT_MODEL_COLORS[key]
        else:
            resolved[key] = _DEFAULT_PALETTE[fallback_idx % len(_DEFAULT_PALETTE)]
            fallback_idx += 1

    return resolved


def plot_training_curves(
    results_dict: dict[str, TrainResult],
    split: Literal["val", "dev", "train"] = "val",
    color_map: dict[str, str | tuple[float, float, float]] | None = None,
) -> None:
    """
    Grafica las curvas de aprendizaje sobre un único conjunto de datos (`val`/`dev` o `train`),
    distinguiendo cada modelo por un color fijo asociado a su etiqueta:
      1. Pérdida (Cross-Entropy Loss)
      2. Perplejidad (PPL = exp(Loss))
      3. Accuracy (escala 0 a 1)
    """
    if not results_dict:
        print("[plots] No hay resultados de entrenamiento para graficar.")
        return

    normalized_split = "val" if split.lower() in {"val", "dev"} else "train"
    split_label = "Validación (Dev)" if normalized_split == "val" else "Entrenamiento (Train)"
    prefix = "val_" if normalized_split == "val" else ""

    colors = get_model_color_map(list(results_dict.keys()), color_map=color_map)

    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))

    for model_key, result in results_dict.items():
        history = result["history"]
        color = colors[model_key]

        loss_vals = history.get(f"{prefix}loss", [])
        ppl_vals = history.get(f"{prefix}perplexity", [float(np.exp(v)) for v in loss_vals])
        acc_vals = history.get(f"{prefix}accuracy", [])
        epochs_range = range(1, len(loss_vals) + 1)

        if loss_vals:
            axes[0].plot(epochs_range, loss_vals, linewidth=2, color=color, label=model_key)

        if ppl_vals:
            axes[1].plot(epochs_range, ppl_vals, linewidth=2, color=color, label=model_key)

        if acc_vals:
            axes[2].plot(epochs_range, acc_vals, linewidth=2, color=color, label=model_key)

    axes[0].set_title(f"Pérdida de {split_label}", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_ylim(bottom=0.0)
    axes[0].legend(fontsize=9)

    axes[1].set_title(f"Perplejidad de {split_label}", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Perplejidad (PPL)")
    axes[1].set_ylim(bottom=0.0)
    axes[1].legend(fontsize=9)

    axes[2].set_title(f"Accuracy de {split_label}", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Época")
    axes[2].set_ylabel("Accuracy (0 - 1)")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].legend(fontsize=9)

    sns.despine(fig=fig)
    fig.suptitle(f"Comparativa de Modelos sobre {split_label} — Etapa 3", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_perplexity_comparison(
    results_dict: dict[str, TrainResult],
    stage2_reference: Stage2Reference = STAGE_2_REFERENCE,
    color_map: dict[str, str | tuple[float, float, float]] | None = None,
) -> None:
    """
    Gráfico de barras comparando la mejor Perplejidad de Validación (métrica de número único)
    conservando el mismo color por etiqueta de modelo.
    """
    if not results_dict:
        print("[plots] No hay resultados para comparar perplejidad.")
        return

    names = list(results_dict.keys())
    ppls = [res["best_val_perplexity"] for res in results_dict.values()]
    colors = get_model_color_map(names, color_map=color_map)
    bar_colors = [colors[name] for name in names]

    sns.set_theme(style="ticks")
    plt.figure(figsize=(10, 5))

    ax = sns.barplot(x=names, y=ppls, palette=bar_colors, hue=names, legend=False)

    ref_ppl = float(stage2_reference["val_perplexity"])
    if ref_ppl > 0.0:
        ax.axhline(
            ref_ppl,
            color="#e74c3c",
            linestyle="--",
            linewidth=2,
            label=f"Referencia Etapa 2 ({stage2_reference['model_name']}): PPL = {ref_ppl:.2f}",
        )
        ax.legend(loc="upper right")

    y_offset = max(ppls) * 0.02 if ppls else 0.5
    for i, val in enumerate(ppls):
        ax.text(i, val + y_offset, f"{val:.2f}", ha="center", fontweight="bold", fontsize=10)

    ax.set_title(
        "Comparativa de Métrica de Número Único: Perplejidad en Validación (Dev)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Configuración del Modelo LSTM", fontsize=11)
    ax.set_ylabel("Perplejidad (PPL = exp(Loss))", fontsize=11)
    ax.set_ylim(bottom=0.0)
    ax.tick_params(axis="x", rotation=15)
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.show()
