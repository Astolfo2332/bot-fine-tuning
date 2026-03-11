from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir: str = "./output"


@dataclass
class LoraHyperparameters:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingHyperparameters:
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    fp16: bool = True
    logging_steps: int = 10
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    max_seq_length: int = 512
