# bot-fine-tuning

A simple set of scripts to fine tune a bot based on interactions trough Whatsapp.

## Usage

1. Clone the repository
2. Install the dependencies with uv
3. See the [data_exploration](./notebook/01_data_exploration.ipynb) for an example of how the data is prepared
4. Run the main_train.py script to fine tune the model
5. Use the main_inference.py script to test the model
6. Use the make_ollama_compatible.py to make the model compatible with Ollama and deploy it on your local machine

The scripts use mlflow so you can track the experiments and compare the results. You can also use mlflow to deploy the model on a server or on the cloud.

for getting the ui of mlflow:

```bash
mlflow ui
```