import torch
from unsloth import FastVisionModel
from src.prompts.prompts import system_prompt


def load_finetuned_model(
    base_model_name: str,
    adapter_path: str,
    max_seq_length: int = 2048,
):
    """Carga el adapter LoRA guardado para inferencia con Unsloth."""
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
    )
    model.load_adapter(adapter_path)

    model.to("cuda")

    FastVisionModel.for_inference(model)
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 256,
) -> str:
    """Genera una respuesta usando el chat template del modelo (non-thinking mode)."""
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": question}],
        }
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        # enable_thinking=False,
        return_tensors="pt"
    ).to(model.device)


    attention_mask = torch.ones_like(input_text).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_text,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
        )

    response = tokenizer.decode(
        outputs[0][input_text.shape[1]:], skip_special_tokens=True
    )
    return response
