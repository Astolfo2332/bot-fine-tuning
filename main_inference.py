import os
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

    # print(os.getenv("HF_HOME"))

    model_config = ModelConfig()

    print("Cargando modelo...")
    model, tokenizer = load_finetuned_model(
        model_config.model_name,
        model_config.inference_dir,
        model_config.max_seq_length
    )

    question = """Que piensas de sherlyn?"""

    print("Ejecutando inferencia...")
    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        question=question,
    )


    print("Pregunta:", question, "\n\n")
    return response

if __name__ == "__main__":
    print("Respuesta:", main())
