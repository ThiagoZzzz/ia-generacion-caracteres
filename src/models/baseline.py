import time
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import OneHotEncoder
from src.utils.config import SEED


def build_kgram_xy(ids_1d: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye pares (X, y) de ventana fija K (stride=1) a partir de una secuencia 1D de IDs.
    X tiene forma (N, k) e y tiene forma (N,).
    """
    arr = np.asarray(ids_1d, dtype=np.int64)
    n = len(arr) - k
    x_mat = np.lib.stride_tricks.sliding_window_view(arr[:-1], window_shape=k)
    y_vec = arr[k:]
    return x_mat[:n], y_vec[:n]


def train_evaluate_baseline(
    train_ids: np.ndarray,
    dev_ids: np.ndarray,
    vocab_size: int,
    k: int = 5,
    seed: int = SEED,
) -> dict:
    """
    Entrena y evalúa el modelo Baseline (Etapa 1): One-Hot sobre los últimos K caracteres
    seguido de Regresión Logística Multinomial sobre la partición 80/10/10.
    """
    x_train, y_train = build_kgram_xy(train_ids, k=k)
    x_dev, y_dev = build_kgram_xy(dev_ids, k=k)

    categories = [np.arange(vocab_size, dtype=np.int64) for _ in range(k)]
    enc = OneHotEncoder(categories=categories, handle_unknown="ignore", sparse_output=True)
    x_train_oh = enc.fit_transform(x_train)
    x_dev_oh = enc.transform(x_dev)

    clf = LogisticRegression(max_iter=500, random_state=seed)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(x_train_oh, y_train)
    elapsed = time.perf_counter() - t0

    train_acc = float(accuracy_score(y_train, clf.predict(x_train_oh)))
    dev_acc = float(accuracy_score(y_dev, clf.predict(x_dev_oh)))

    # Probabilidades sobre todas las clases conocidas por el clasificador
    dev_probs_present = clf.predict_proba(x_dev_oh)
    full_probs = np.full((len(y_dev), vocab_size), 1e-12, dtype=np.float64)
    full_probs[:, clf.classes_] = np.clip(dev_probs_present, 1e-12, 1.0)
    full_probs /= full_probs.sum(axis=1, keepdims=True)

    dev_loss = float(log_loss(y_dev, full_probs, labels=np.arange(vocab_size)))
    dev_ppl = float(np.exp(dev_loss))
    dev_bpc = float(dev_loss / np.log(2.0))
    trainable_params = int(clf.coef_.size + clf.intercept_.size)

    return {
        "model_name": f"baseline_logreg_k{k}",
        "k": k,
        "trainable_params": trainable_params,
        "training_time_s": float(elapsed),
        "train_accuracy": train_acc,
        "val_loss": dev_loss,
        "val_perplexity": dev_ppl,
        "val_bpc": dev_bpc,
        "val_accuracy": dev_acc,
    }
