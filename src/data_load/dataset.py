import json

from datasets import Dataset
from transformers import AutoTokenizer


def load_qa_json(path: str) -> Dataset:
    """Lee el JSON de pares Q&A y retorna un HF Dataset, filtrando entradas vacías."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = [
        item for item in data
        if item.get("question", "").strip() and item.get("answer", "").strip()
    ]

    return Dataset.from_list(filtered)


def format_to_text(example: dict, tokenizer: AutoTokenizer) -> dict:
    """Aplica el chat template con non-thinking mode y devuelve texto pre-formateado."""
    messages = [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    return {"text": text}


def prepare_dataset(
    path: str, tokenizer: AutoTokenizer, test_size: float = 0.1
) -> dict:
    """Carga el JSON, formatea con chat template (non-thinking) y separa en train/test.

    Returns:
        dict con claves "train" y "test", cada una un HF Dataset.
    """
    dataset = load_qa_json(path)
    dataset = dataset.map(lambda ex: format_to_text(ex, tokenizer))

    split = dataset.train_test_split(test_size=test_size, seed=42)
    return {"train": split["train"], "test": split["test"]}
