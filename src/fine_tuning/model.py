import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.fine_tuning.config import ModelConfig, LoraHyperparameters


def load_model_and_tokenizer(
    config: ModelConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Carga el modelo base con cuantización 4-bit (QLoRA) y su tokenizer."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def create_lora_config(params: LoraHyperparameters) -> LoraConfig:
    """Crea la configuración LoRA a partir de los hiperparámetros."""
    return LoraConfig(
        r=params.r,
        lora_alpha=params.lora_alpha,
        lora_dropout=params.lora_dropout,
        target_modules=params.target_modules,
        bias=params.bias,
        task_type=params.task_type,
    )


def apply_lora(model: AutoModelForCausalLM, lora_config: LoraConfig):
    """Aplica LoRA al modelo y retorna el PeftModel."""
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
