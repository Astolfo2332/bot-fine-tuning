import torch
from unsloth import FastLanguageModel


def load_finetuned_model(
    adapter_path: str,
    max_seq_length: int = 2048,
):
    """Carga el adapter LoRA guardado para inferencia con Unsloth."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 256,
) -> str:
    """Genera una respuesta usando el chat template del modelo (non-thinking mode)."""
    messages = [{"role": "user", "content": question}]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return response
