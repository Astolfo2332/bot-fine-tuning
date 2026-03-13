from dotenv import load_dotenv
load_dotenv()

import torch
import unsloth

from src.fine_tuning.inference import (
    load_finetuned_model,
    generate_response
)
from src.fine_tuning.config import ModelConfig

def main():

    model_config = ModelConfig()

    model, tokenizer = load_finetuned_model(
        model_config.model_name,
        model_config.output_dir,
        model_config.max_seq_length
    )

    question = "Tu me quieres?"

    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        question=question,
    )

    return response

if __name__ == "__main__":
    print(main())
