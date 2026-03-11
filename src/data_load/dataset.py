import json

from datasets import Dataset


def load_qa_json(path: str) -> Dataset:
    """Lee el JSON de pares Q&A y retorna un HF Dataset, filtrando entradas vacías."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = [
        item for item in data
        if item.get("question", "").strip() and item.get("answer", "").strip()
    ]

    return Dataset.from_list(filtered)


def format_chat_messages(example: dict) -> dict:
    """Transforma un par Q&A al formato de mensajes chat para SFTTrainer."""
    messages = [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    return {"messages": messages}


def prepare_dataset(path: str, test_size: float = 0.1) -> dict:
    """Carga el JSON, formatea a chat y separa en train/test.

    Returns:
        dict con claves "train" y "test", cada una un HF Dataset.
    """
    dataset = load_qa_json(path)
    dataset = dataset.map(format_chat_messages)

    split = dataset.train_test_split(test_size=test_size, seed=42)
    return {"train": split["train"], "test": split["test"]}
