from dataclasses import dataclass, field


model_reference ="Ministral-3-8B-Instruct-2512-bnb-4bit"
# model_reference ="Llama-3.2-3B"
@dataclass
class ModelConfig:
    model_name: str = f"unsloth/{model_reference}"
    output_dir: str = f"./output/{model_reference}-multiturn"
    inference_dir: str = f"./output/{model_reference}-multiturn"
    max_seq_length: int = 256
    load_in_4bit: bool = True
    load_in_16bit: bool = False
    full_finetuning: bool = False
    chat_template: str = "native"


@dataclass
class LoraHyperparameters:
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407
    finetune_language_layers = True
    finetune_mlp_modules = True
    finetune_attention_layers = True
    finetune_vision_layers = False


@dataclass
class TrainingHyperparameters:
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10
    lr_scheduler_type: str = "cosine"
    bf16: bool = False
    logging_steps: int = 1
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    optim: str = "adamw_8bit"
    seed: int = 3407
    dataset_num_proc: int = 1
