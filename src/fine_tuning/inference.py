import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_finetuned_model(
    base_model_name: str,
    adapter_path: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Carga el modelo base + adapter LoRA guardado para inferencia."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    return model, tokenizer


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    max_new_tokens: int = 256,
) -> str:
    """Genera una respuesta usando el chat template del modelo."""
    messages = [{"role": "user", "content": question}]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return response
