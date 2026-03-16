import os
from dotenv import load_dotenv
from unsloth import FastLanguageModel
from src.prompts.ollama_prompt import ollama_modelfile

load_dotenv()

import torch
import unsloth


from src.fine_tuning.config import ModelConfig

def main():
    model_config = ModelConfig()

    print("Cargando modelo...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config.inference_dir,
        max_seq_length=256,
        load_in_4bit=True,
        load_in_16bit=False,
        local_files_only=True,
    )


    print("Convirtiendo modelo...")

    model.save_pretrained_merged(
        "ollama_models/miguel_bot",
        tokenizer,
        save_method="merged_16bit"
    )

    model_file = "./Ministral-3-8B-Instruct-2512.Q4_K_M.gguf"

    ollama_modelfile_format = ollama_modelfile.format(
        model_file=model_file,
        eos_token=tokenizer.eos_token
        )

    with open("ollama_models/miguel_bot/Modelfile", "w", encoding="utf-8") as f:
        f.write(ollama_modelfile_format)

    print("Conversión exitosa")

if __name__ == "__main__":
    main()
