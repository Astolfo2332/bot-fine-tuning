from src.data_load.dataset import prepare_dataset
from src.fine_tuning.config import ModelConfig, LoraHyperparameters, TrainingHyperparameters
from src.fine_tuning.model import load_model_and_tokenizer, create_lora_config, apply_lora
from src.fine_tuning.trainer import create_training_args, create_trainer, train_and_save


def main():
    # Configuración
    model_config = ModelConfig()
    lora_params = LoraHyperparameters()
    training_params = TrainingHyperparameters()

    # 1. Cargar y preparar dataset
    print("Cargando dataset...")
    datasets = prepare_dataset("data/qa_data.json")
    print(f"Train: {len(datasets['train'])} ejemplos | Test: {len(datasets['test'])} ejemplos")

    # 2. Cargar modelo y tokenizer (4-bit)
    print(f"Cargando modelo {model_config.model_name}...")
    model, tokenizer = load_model_and_tokenizer(model_config)

    # 3. Aplicar LoRA
    print("Aplicando LoRA...")
    lora_config = create_lora_config(lora_params)
    model = apply_lora(model, lora_config)

    # 4. Configurar y ejecutar entrenamiento
    training_args = create_training_args(model_config, training_params)
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        training_args=training_args,
        max_seq_length=training_params.max_seq_length,
    )

    train_and_save(trainer, model_config.output_dir)


if __name__ == "__main__":
    main()
