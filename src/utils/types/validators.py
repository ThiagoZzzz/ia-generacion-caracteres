from typing import Any, get_args, get_origin, get_type_hints
import numpy as np
from src.utils.types.config_types import (
    EvaluationConfig,
    ModelConfig,
    Stage2Reference,
    TrainingConfig,
)


def _check_value_type(value: Any, expected_type: Any) -> bool:
    """
    Verifica en tiempo de ejecución si un valor coincide con la anotación de tipo esperada.
    Maneja la distinción estricta entre bool e int en Python, permite int donde se espera float,
    y valida recursivamente listas tipadas (ej. list[float]) o TypedDicts anidados.
    """
    origin = get_origin(expected_type)

    # Caso 1: Listas tipadas como list[float], list[int], etc.
    if origin is list:
        if not isinstance(value, list):
            return False
        args = get_args(expected_type)
        if not args:
            return True
        inner_type = args[0]
        return all(_check_value_type(item, inner_type) for item in value)

    # Caso 2: TypedDict anidado
    if hasattr(expected_type, "__required_keys__"):
        if not isinstance(value, dict):
            return False
        validate_schema(value, expected_type)
        return True

    # Caso 3: Tipos primitivos con reglas numéricas seguras
    if expected_type is bool:
        return isinstance(value, bool)

    if expected_type is int:
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    if expected_type is float:
        return isinstance(value, (int, float, np.number)) and not isinstance(value, bool)

    if expected_type is str:
        return isinstance(value, str)

    if isinstance(expected_type, type):
        return isinstance(value, expected_type)

    return True


def validate_schema(data: Any, schema: type, name: str = "config") -> None:
    """
    Validador general de esquemas: recibe un diccionario y una clase TypedDict (`schema`)
    y verifica en tiempo de ejecución:
      1. Que `data` sea efectivamente un diccionario (`dict`).
      2. Que estén presentes todas las claves requeridas por el esquema.
      3. Que no existan claves desconocidas o mal escritas.
      4. Que el tipo de dato de cada valor coincida con el declarado en el esquema.
    """
    if not isinstance(data, dict):
        raise TypeError(
            f"[{name}] Se esperaba un diccionario ({schema.__name__}), "
            f"pero se recibió '{type(data).__name__}'."
        )

    type_hints = get_type_hints(schema)
    required_keys: set[str] = set(getattr(schema, "__required_keys__", type_hints.keys()))
    allowed_keys: set[str] = set(type_hints.keys())
    provided_keys: set[str] = set(data.keys())

    # 1. Verificar claves obligatorias faltantes
    missing_keys = required_keys - provided_keys
    if missing_keys:
        raise KeyError(
            f"[{name}] Faltan claves obligatorias para {schema.__name__}: {sorted(missing_keys)}"
        )

    # 2. Verificar claves desconocidas / errores de tipeo
    unknown_keys = provided_keys - allowed_keys
    if unknown_keys:
        raise KeyError(
            f"[{name}] Claves no reconocidas en {schema.__name__}: {sorted(unknown_keys)}. "
            f"Claves permitidas: {sorted(allowed_keys)}"
        )

    # 3. Verificar tipos de cada clave presente
    for key, value in data.items():
        expected_type = type_hints[key]
        if not _check_value_type(value, expected_type):
            expected_name = getattr(expected_type, "__name__", str(expected_type))
            raise TypeError(
                f"[{name}] Tipo inválido en la clave '{key}': se esperaba '{expected_name}', "
                f"pero se recibió {repr(value)} ({type(value).__name__})."
            )


def validate_positive_int(value: Any, param_name: str, context: str = "") -> None:
    """Verifica que un parámetro escalar sea un entero estrictamente positivo (> 0)."""
    prefix = f"[{context}] " if context else ""
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or int(value) <= 0:
        raise ValueError(
            f"{prefix}El parámetro '{param_name}' debe ser un entero > 0. Se recibió: {repr(value)}."
        )


def validate_model_config(config: ModelConfig, name: str = "ModelConfig") -> None:
    """
    Valida la estructura y tipos de `ModelConfig` mediante `validate_schema`
    y comprueba que los hiperparámetros de arquitectura estén en rangos matemáticos válidos.
    """
    validate_schema(config, ModelConfig, name=name)

    if config["embedding_dim"] <= 0:
        raise ValueError(f"[{name}] 'embedding_dim' debe ser > 0 (recibido: {config['embedding_dim']}).")
    if config["rnn_units"] <= 0:
        raise ValueError(f"[{name}] 'rnn_units' debe ser > 0 (recibido: {config['rnn_units']}).")
    if config["dense_units"] < 0:
        raise ValueError(f"[{name}] 'dense_units' debe ser >= 0 (recibido: {config['dense_units']}).")
    if not (0.0 <= config["dropout"] < 1.0):
        raise ValueError(f"[{name}] 'dropout' debe estar en [0.0, 1.0) (recibido: {config['dropout']}).")
    if not (0.0 <= config["recurrent_dropout"] < 1.0):
        raise ValueError(
            f"[{name}] 'recurrent_dropout' debe estar en [0.0, 1.0) (recibido: {config['recurrent_dropout']})."
        )
    if config["learning_rate"] <= 0.0:
        raise ValueError(f"[{name}] 'learning_rate' debe ser > 0.0 (recibido: {config['learning_rate']}).")


def validate_training_config(config: TrainingConfig, name: str = "TrainingConfig") -> None:
    """
    Valida la estructura y tipos de `TrainingConfig` y verifica rangos de épocas,
    paciencia, factor de reducción de learning rate y ventana de co-ocurrencia.
    """
    validate_schema(config, TrainingConfig, name=name)

    if config["epochs"] <= 0:
        raise ValueError(f"[{name}] 'epochs' debe ser > 0 (recibido: {config['epochs']}).")
    if config["early_stopping_patience"] < 1:
        raise ValueError(
            f"[{name}] 'early_stopping_patience' debe ser >= 1 (recibido: {config['early_stopping_patience']})."
        )
    if config["early_stopping_min_delta"] < 0.0:
        raise ValueError(
            f"[{name}] 'early_stopping_min_delta' debe ser >= 0.0 (recibido: {config['early_stopping_min_delta']})."
        )
    if config["reduce_learning_rate_patience"] < 1:
        raise ValueError(
            f"[{name}] 'reduce_learning_rate_patience' debe ser >= 1 "
            f"(recibido: {config['reduce_learning_rate_patience']})."
        )
    if not (0.0 < config["reduce_learning_rate_factor"] < 1.0):
        raise ValueError(
            f"[{name}] 'reduce_learning_rate_factor' debe estar en (0.0, 1.0) "
            f"(recibido: {config['reduce_learning_rate_factor']})."
        )
    if config["min_learning_rate"] <= 0.0:
        raise ValueError(
            f"[{name}] 'min_learning_rate' debe ser > 0.0 (recibido: {config['min_learning_rate']})."
        )
    if config["pretrain_concurrence_window"] < 1:
        raise ValueError(
            f"[{name}] 'pretrain_concurrence_window' debe ser >= 1 "
            f"(recibido: {config['pretrain_concurrence_window']})."
        )


def validate_evaluation_config(config: EvaluationConfig, name: str = "EvaluationConfig") -> None:
    """Valida el diccionario de configuración de evaluación y generación."""
    validate_schema(config, EvaluationConfig, name=name)

    if config["hard_examples_count"] < 1:
        raise ValueError(f"[{name}] 'hard_examples_count' debe ser >= 1.")
    if config["hard_examples_context"] < 1:
        raise ValueError(f"[{name}] 'hard_examples_context' debe ser >= 1.")
    if config["generation_num_chars"] < 1:
        raise ValueError(f"[{name}] 'generation_num_chars' debe ser >= 1.")
    if not config["generation_seed_text"].strip():
        raise ValueError(f"[{name}] 'generation_seed_text' no puede ser una cadena vacía.")
    if not config["generation_temperatures"] or any(t <= 0.0 for t in config["generation_temperatures"]):
        raise ValueError(f"[{name}] Todas las temperaturas en 'generation_temperatures' deben ser > 0.0.")


def validate_stage2_reference(reference: Stage2Reference, name: str = "Stage2Reference") -> None:
    """Valida el diccionario de métricas de referencia de la Etapa 2."""
    validate_schema(reference, Stage2Reference, name=name)

    if reference["val_loss"] < 0.0:
        raise ValueError(f"[{name}] 'val_loss' no puede ser negativo (recibido: {reference['val_loss']}).")
    if reference["val_perplexity"] < 0.0:
        raise ValueError(
            f"[{name}] 'val_perplexity' no puede ser negativo (recibido: {reference['val_perplexity']})."
        )
    if not (0.0 <= reference["val_accuracy"] <= 1.0):
        raise ValueError(
            f"[{name}] 'val_accuracy' debe estar en [0.0, 1.0] (recibido: {reference['val_accuracy']})."
        )
