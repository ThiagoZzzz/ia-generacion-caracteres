from typing import TypedDict


class TrainingConfig(TypedDict):
    epochs: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    reduce_learning_rate_patience: int
    reduce_learning_rate_factor: float
    min_learning_rate: float
    pretrain_concurrence_window: int

class EvaluationConfig(TypedDict):
    hard_examples_count: int
    hard_examples_context: int
    generation_temperatures: list[float]
    generation_num_chars: int
    generation_seed_text: str

class Stage2Reference(TypedDict):
    model_name: str
    val_loss: float
    val_perplexity: float
    val_accuracy: float

class ModelConfig(TypedDict):
    description: str
    embedding_dim: int
    rnn_units: int
    dense_units: int
    dropout: float
    recurrent_dropout: float
    use_pretrained: bool
    trainable_embedding: bool
    learning_rate: float
