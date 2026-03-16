from dotenv import load_dotenv

load_dotenv()

import os
import torch
import unsloth

from src.data_load.dataset import prepare_dataset
from src.fine_tuning.config import ModelConfig, LoraHyperparameters, TrainingHyperparameters
from src.fine_tuning.model import load_model_and_tokenizer, apply_lora
from src.fine_tuning.trainer import create_sft_config, create_trainer, train_and_save
from src.fine_tuning.mlflow_utils import initialize_mlflow_experiment, log_training_summary, end_mlflow_run, log_prompt
from src.prompts.prompts import system_prompt


def main():
    # Configuración
    model_config = ModelConfig()
    lora_params = LoraHyperparameters()
    training_params = TrainingHyperparameters()

    # Detecta automáticamente si existe un checkpoint para reanudar
    resume = (
        os.path.exists(model_config.output_dir)
        and any(
            d.startswith("checkpoint-")
            for d in os.listdir(model_config.output_dir)
            if os.path.isdir(os.path.join(model_config.output_dir, d))
        )
    )

    if resume:
        print("Checkpoint detectado, reanudando entrenamiento...")

    # Inicializar MLflow tracking
    experiment_name = "ministral-3-finetuning"
    initialize_mlflow_experiment(
        experiment_name=experiment_name,
        model_config=model_config,
        lora_params=lora_params,
        training_params=training_params,
        resume=resume,
    )

    # 1. Cargar modelo y tokenizer (bf16 con Unsloth)
    print(f"Cargando modelo {model_config.model_name}...")
    model, tokenizer = load_model_and_tokenizer(model_config)

    # 2. Cargar y preparar dataset (requiere tokenizer para chat template non-thinking)
    print("Cargando dataset...")
    datasets = prepare_dataset("data/qa_data.json", tokenizer, model_config.chat_template)
    print(f"Train: {len(datasets['train'])} ejemplos | Test: {len(datasets['test'])} ejemplos")
    print("Ejemplo de entrada formateada:", datasets["train"][0])

    # Log dataset info a MLflow
    log_training_summary(
        train_samples=len(datasets['train']),
        eval_samples=len(datasets['test']),
        total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    log_prompt(system_prompt, "system_prompt")

    # 3. Aplicar LoRA con Unsloth
    print("Aplicando LoRA...")
    model = apply_lora(model, lora_params, model_config.max_seq_length)

    # 4. Configurar y ejecutar entrenamiento
    sft_config = create_sft_config(model_config, training_params)
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        sft_config=sft_config,
    )

    train_and_save(trainer, model_config.output_dir, resume=resume)

    # Finalizar MLflow run exitosamente
    end_mlflow_run(status="FINISHED")
    print("\nExperimento completado y logueado en MLflow")



if __name__ == "__main__":
    main()
