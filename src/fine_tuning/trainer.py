from transformers import TrainingArguments
from trl import SFTTrainer

from src.fine_tuning.config import TrainingHyperparameters, ModelConfig


def create_training_args(
    model_config: ModelConfig,
    params: TrainingHyperparameters,
) -> TrainingArguments:
    """Crea los TrainingArguments a partir de la configuración."""
    return TrainingArguments(
        output_dir=model_config.output_dir,
        num_train_epochs=params.num_train_epochs,
        per_device_train_batch_size=params.per_device_train_batch_size,
        per_device_eval_batch_size=params.per_device_eval_batch_size,
        gradient_accumulation_steps=params.gradient_accumulation_steps,
        learning_rate=params.learning_rate,
        weight_decay=params.weight_decay,
        warmup_ratio=params.warmup_ratio,
        lr_scheduler_type=params.lr_scheduler_type,
        fp16=params.fp16,
        logging_steps=params.logging_steps,
        eval_strategy=params.eval_strategy,
        save_strategy=params.save_strategy,
        report_to="none",
    )


def create_trainer(model, tokenizer, train_dataset, eval_dataset, training_args, max_seq_length):
    """Configura el SFTTrainer de trl para supervised fine-tuning con chat template."""
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        max_seq_length=max_seq_length,
    )


def train_and_save(trainer: SFTTrainer, output_dir: str):
    """Ejecuta el entrenamiento y guarda el adapter LoRA + tokenizer."""
    print("Iniciando entrenamiento...")
    trainer.train()

    print(f"Guardando modelo en {output_dir}")
    trainer.save_model(output_dir)
    trainer.tokenizer.save_pretrained(output_dir)

    print("Entrenamiento completado.")
