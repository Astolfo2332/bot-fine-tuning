from unsloth import FastLanguageModel

from src.fine_tuning.config import ModelConfig, LoraHyperparameters


def load_model_and_tokenizer(config: ModelConfig):
    """Carga el modelo base con Unsloth (bf16 LoRA, sin QLoRA 4-bit)."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        load_in_16bit=config.load_in_16bit,
        full_finetuning=config.full_finetuning,
    )
    return model, tokenizer


def apply_lora(model, params: LoraHyperparameters, max_seq_length: int):
    """Aplica LoRA al modelo usando Unsloth."""
    model = FastLanguageModel.get_peft_model(
        model,
        r=params.r,
        lora_alpha=params.lora_alpha,
        lora_dropout=params.lora_dropout,
        target_modules=params.target_modules,
        bias=params.bias,
        use_gradient_checkpointing=params.use_gradient_checkpointing,
        random_state=params.random_state,
        max_seq_length=max_seq_length,
    )
    return model
