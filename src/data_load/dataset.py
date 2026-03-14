import json

from datasets import Dataset
from transformers import AutoTokenizer
from src.prompts.prompts import system_prompt
from unsloth.chat_templates import get_chat_template


def load_qa_json(path: str) -> Dataset:
    """Lee el JSON de pares Q&A y retorna un HF Dataset, filtrando entradas vacías."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = [
        item for item in data
        if item.get("question", "").strip() and item.get("answer", "").strip()
    ]

    return Dataset.from_list(filtered)


def format_to_text_qwen(example: dict, tokenizer: AutoTokenizer) -> dict:
    """Aplica el chat template con non-thinking mode y devuelve texto pre-formateado."""
    messages = [
        # {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": example["question"]}]},
        {"role": "assistant", "content": [{"type": "text", "text": example["answer"]}]},
    ]

    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=False,
    #     # enable_thinking=False
    # )

    return {"messages": messages}

def format_to_text_llama(example: dict, tokenizer: AutoTokenizer) -> dict:
    messages = [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]

    return {"messages": messages}

def make_multi_turn_conversation_qwen(dataset: Dataset, batch_size:int=3) -> Dataset:
    """Se juntan los mensajes para obtener un sistema muilti conversacional en batches de 3 turnos de conversacion"""

    new_data = []
    # Iteramos sobre el dataset en saltos del tamaño del batch
    for i in range(0, len(dataset), batch_size):
        # Extraemos el subconjunto de ejemplos
        batch = dataset[i: i + batch_size]

        # Si el último batch es más pequeño que el batch_size,
        combined_messages = []
        for msg_list in batch["messages"]:
            combined_messages.extend(msg_list)

        system_message = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
        new_data.append({"messages": system_message + combined_messages})

    return Dataset.from_list(new_data)

def make_multi_turn_conversation_llama(dataset: Dataset, batch_size:int=3) -> Dataset:

    new_data = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i: i + batch_size]
        combined_messages = []
        for msg_list in batch["messages"]:
            combined_messages.extend(msg_list)

        # system_message = [{"role": "system", "content":system_prompt}]
        new_data.append({"messages": combined_messages})

    return Dataset.from_list(new_data)

def map_to_tokenizer_llama(example: dict, tokenizer) -> dict:
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
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
    # dataset = dataset.map(lambda ex: format_to_text_qwen(ex, tokenizer))

    dataset = dataset.map(lambda ex: format_to_text_llama(ex, tokenizer))
    dataset = make_multi_turn_conversation_llama(dataset, 5)

    chat_template = get_chat_template(
        tokenizer,
        chat_template="chatml"
    )

    dataset = dataset.map(lambda ex: map_to_tokenizer_llama(ex, chat_template))

    split = dataset.train_test_split(test_size=test_size, seed=23)

    return {"train": split["train"], "test": split["test"]}
