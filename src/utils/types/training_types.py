from typing import TypedDict
from src.utils.types.config_types import ModelConfig


class TrainingHistory(TypedDict, total=False):
    loss: list[float]
    accuracy: list[float]
    perplexity: list[float]
    val_loss: list[float]
    val_accuracy: list[float]
    val_perplexity: list[float]
    learning_rate: list[float]


class TrainResult(TypedDict):
    model_name: str
    config: ModelConfig
    epochs_trained: int
    best_epoch: int
    best_val_loss: float
    best_val_perplexity: float
    best_val_bpc: float
    best_val_accuracy: float
    trainable_params: int
    non_trainable_params: int
    total_params: int
    history: TrainingHistory


class EvaluationResult(TypedDict):
    loss: float
    perplexity: float
    bpc: float
    accuracy: float
    stage2_reference_perplexity: float
    perplexity_reduction_abs: float
    perplexity_reduction_pct: float
    loss_reduction_pct: float
    accuracy_gain_percentage_points: float


class HardExample(TypedDict):
    context: str
    target_char: str
    predicted_char: str
    target_prob: float
    predicted_prob: float
    loss: float
    category: str
