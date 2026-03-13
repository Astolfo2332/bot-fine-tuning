"""Utilidades para MLflow tracking de experimentos de fine-tuning."""

import mlflow
from dataclasses import asdict
from datetime import datetime

from src.fine_tuning.config import ModelConfig, LoraHyperparameters, TrainingHyperparameters


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Convierte un diccionario anidado en uno plano con keys separadas por puntos."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)):
            # Convertir listas a string para que MLflow las pueda loguear
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def initialize_mlflow_experiment(
    experiment_name: str,
    model_config: ModelConfig,
    lora_params: LoraHyperparameters,
    training_params: TrainingHyperparameters,
    resume: bool = False,
) -> str:
    """
    Inicializa un experimento MLflow y loguea todos los parámetros de configuración.
    
    Args:
        experiment_name: Nombre del experimento
        model_config: Configuración del modelo
        lora_params: Hiperparámetros de LoRA
        training_params: Hiperparámetros de entrenamiento
        resume: Si es True, es una reanudación de entrenamiento
    
    Returns:
        run_id del experimento creado
    """
    # Crear nombre del experimento si no existe
    try:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    except AttributeError:
        experiment_id = mlflow.create_experiment(experiment_name)
    
    mlflow.set_experiment(experiment_name)
    
    # Crear run name descriptivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short_name = model_config.output_dir.split("/")[-1].replace(".", "-")
    run_name = f"{model_short_name}_{timestamp}"
    if resume:
        run_name += "_resumed"
    
    # Iniciar nuevo run
    run = mlflow.start_run(run_name=run_name)
    
    # Loguear parámetros de ModelConfig
    model_config_dict = asdict(model_config)
    model_config_flat = flatten_dict(model_config_dict, parent_key="model")
    mlflow.log_params(model_config_flat)
    
    # Loguear parámetros de LoraHyperparameters
    lora_params_dict = asdict(lora_params)
    lora_params_flat = flatten_dict(lora_params_dict, parent_key="lora")
    mlflow.log_params(lora_params_flat)
    
    # Loguear parámetros de TrainingHyperparameters
    training_params_dict = asdict(training_params)
    training_params_flat = flatten_dict(training_params_dict, parent_key="training")
    mlflow.log_params(training_params_flat)
    
    # Loguear tags adicionales
    mlflow.set_tag("resume", resume)
    mlflow.set_tag("timestamp", timestamp)
    
    print(f"MLflow tracking iniciado - Experimento: {experiment_name}")
    print(f"  Run ID: {run.info.run_id}")
    print(f"  Run name: {run_name}")
    
    return run.info.run_id


def log_training_summary(
    train_samples: int,
    eval_samples: int,
    total_parameters: int,
    trainable_parameters: int,
):
    """Loguea información adicional del entrenamiento."""
    mlflow.log_param("train_samples", train_samples)
    mlflow.log_param("eval_samples", eval_samples)
    mlflow.log_param("total_parameters", total_parameters)
    mlflow.log_param("trainable_parameters", trainable_parameters)
    
    if total_parameters > 0:
        trainable_percent = (trainable_parameters / total_parameters) * 100
        mlflow.log_metric("trainable_params_percent", trainable_percent)
    
    print(f"Información del entrenamiento logueada en MLflow")

def log_prompt(prompt:str, key:str):
    mlflow.log_dict({"prompt": prompt}, f"prompts/{key}")

def end_mlflow_run(status: str = "FINISHED"):
    """Finaliza el run actual de MLflow."""
    mlflow.set_tag("status", status)
    mlflow.end_run()
    print(f"MLflow run finalizado - Status: {status}")


