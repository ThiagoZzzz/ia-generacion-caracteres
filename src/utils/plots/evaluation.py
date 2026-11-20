import pandas as pd
from src.utils.config import STAGE_2_REFERENCE
from src.utils.types import HardExample, Stage2Reference, TrainResult


def _is_jupyter_notebook() -> bool:
    """Detecta si el código se ejecuta dentro de un kernel interactivo de Jupyter/VS Code."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


def _render_dataframe(df: pd.DataFrame, title: str) -> None:
    """
    Renderiza un DataFrame de forma óptima según el entorno:
    - Jupyter Notebook / VS Code: tabla HTML nativa vía IPython.display.
    - Consola / Terminal: tabla de texto alineada por ancho de columna.
    """
    if _is_jupyter_notebook():
        from IPython.display import Markdown, display

        display(Markdown(f"#### {title}"))
        display(df)
        return

    table_str = df.to_string(index=False)
    sep_len = max(len(title), max((len(line) for line in table_str.splitlines()), default=60))
    print("\n" + "=" * sep_len)
    print(title)
    print("=" * sep_len)
    print(table_str)
    print("=" * sep_len + "\n")


def format_comparison_table(
    results_dict: dict[str, TrainResult],
    stage2_reference: Stage2Reference = STAGE_2_REFERENCE,
) -> pd.DataFrame:
    """
    Construye y renderiza la Tabla Comparativa Integral como DataFrame de pandas
    con configuraciones, estrategia de embeddings y métricas obtenidas.
    """
    if not results_dict:
        print("[evaluation] No hay resultados registrados para generar la tabla comparativa.")
        return pd.DataFrame()

    ref_ppl = float(stage2_reference["val_perplexity"])
    rows: list[dict] = []

    for key, res in results_dict.items():
        cfg = res["config"]
        if cfg["use_pretrained"]:
            emb_type = "Pre-entrenado (Fine-tuned)" if cfg["trainable_embedding"] else "Pre-entrenado (Congelado)"
        else:
            emb_type = "Desde cero (Scratch)"

        if ref_ppl > 0.0:
            delta_pct = ((ref_ppl - res["best_val_perplexity"]) / ref_ppl) * 100.0
            delta_str = f"{delta_pct:+.2f}%"
        else:
            delta_str = "N/A (Ref=0)"

        l1_str = f"{cfg['l1_reg']:.0e}" if cfg["l1_reg"] > 0 else "0"
        l2_str = f"{cfg['l2_reg']:.0e}" if cfg["l2_reg"] > 0 else "0"

        rows.append(
            {
                "Modelo": key,
                "Embeddings": emb_type,
                "Dim (E/L/D)": f"{cfg['embedding_dim']}/{cfg['rnn_units']}/{cfg['dense_units']}",
                "Dropout (In/Rec)": f"{cfg['dropout']:.1f}/{cfg['recurrent_dropout']:.1f}",
                "L1 / L2": f"{l1_str}/{l2_str}",
                "LR": f"{cfg['learning_rate']:.0e}",
                "Early Stop": "Sí" if cfg["use_early_stopping"] else "No",
                "Parámetros (Entr.)": f"{res['total_params']:,} ({res['trainable_params']:,})",
                "Época": f"{res['best_epoch']}/{res['epochs_trained']}",
                "Val Loss": round(res["best_val_loss"], 4),
                "Val PPL": round(res["best_val_perplexity"], 2),
                "Val BPC": round(res["best_val_bpc"], 3),
                "Val Acc": round(res["best_val_accuracy"], 4),
                "Mejora PPL vs E2": delta_str,
            }
        )

    df = pd.DataFrame(rows)
    _render_dataframe(df, "TABLA COMPARATIVA INTEGRAL — ETAPA 3 (MODELOS LSTM Y EMBEDDINGS)")
    return df


def format_hard_examples_table(hard_examples: list[HardExample]) -> pd.DataFrame:
    """
    Construye y renderiza la tabla de auditoría de errores con los >= 10 ejemplos
    concretos más difíciles para el modelo y su categoría morfosintáctica.
    """
    if not hard_examples:
        print("[evaluation] No se encontraron ejemplos difíciles para mostrar.")
        return pd.DataFrame()

    rows = [
        {
            "#": idx,
            "Categoría": ex["category"],
            "Contexto Previo": repr(ex["context"]),
            "Esperado": repr(ex["target_char"]),
            "Predicho": repr(ex["predicted_char"]),
            "P(Esp.)": round(ex["target_prob"], 4),
            "P(Pred.)": round(ex["predicted_prob"], 4),
            "Loss": round(ex["loss"], 4),
        }
        for idx, ex in enumerate(hard_examples, start=1)
    ]

    df = pd.DataFrame(rows)
    _render_dataframe(df, f"AUDITORÍA DE ERRORES — TOP {len(hard_examples)} EJEMPLOS DIFÍCILES")
    return df
