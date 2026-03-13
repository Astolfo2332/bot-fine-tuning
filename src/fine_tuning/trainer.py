from trl import SFTTrainer, SFTConfig

from src.fine_tuning.config import TrainingHyperparameters, ModelConfig

from unsloth.trainer import UnslothVisionDataCollator


def create_sft_config(
    model_config: ModelConfig,
    params: TrainingHyperparameters,
) -> SFTConfig:
    """Crea el SFTConfig a partir de la configuración."""
    return SFTConfig(
        output_dir=model_config.output_dir,
        max_length=model_config.max_seq_length,
        dataset_text_field="",
        num_train_epochs=params.num_train_epochs,
        per_device_train_batch_size=params.per_device_train_batch_size,
        per_device_eval_batch_size=params.per_device_eval_batch_size,
        gradient_accumulation_steps=params.gradient_accumulation_steps,
        learning_rate=params.learning_rate,
        weight_decay=params.weight_decay,
        warmup_steps=params.warmup_steps,
        lr_scheduler_type=params.lr_scheduler_type,
        fp16=False,
        bf16=params.bf16,
        logging_steps=params.logging_steps,
        eval_strategy=params.eval_strategy,
        save_strategy=params.save_strategy,
        save_total_limit=params.save_total_limit,
        load_best_model_at_end=params.load_best_model_at_end,
        optim=params.optim,
        seed=params.seed,
        dataset_num_proc=params.dataset_num_proc,
        report_to="mlflow",
        # Para qwen3.5
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )


def create_trainer(model, tokenizer, train_dataset, eval_dataset, sft_config):
    """Configura el SFTTrainer de trl para supervised fine-tuning con texto pre-formateado."""
    return SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        data_collator=UnslothVisionDataCollator(model, tokenizer)
    )


def train_and_save(trainer: SFTTrainer, output_dir: str, resume: bool = False):
    """Ejecuta el entrenamiento y guarda el adapter LoRA + tokenizer."""
    print("Iniciando entrenamiento...")

    if resume:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    print(f"Guardando modelo en {output_dir}")
    trainer.save_model(output_dir)

    print("Entrenamiento completado.")
