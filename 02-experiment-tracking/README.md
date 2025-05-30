# Experiment Tracking with MLflow

Create a conda environment:

```bash
conda create -n exp-tracking-env python=3.9
```

Then activate it:

```bash
conda activate exp-tracking-env
```

Install the requirements.txt

```bash
pip install -r ./requirements.txt
```

Then run mlflow cli, to get the version:

```bash
mlflow --version
```

Which is 2.22.0

## Run the MLflow UI

Run the ui, passing an option for the backend store URI. This example uses a SQLite database file called `mlflow.db`.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Pre-process the data

```bash
python preprocess_data.py --raw_data_path ./data --dest_path ./output
```

This creates 4 files in the `output` folder.


## Train the model

```bash
python train.py --data_path ./output
```

With `mlflow.autolog()` enabled, this will automatically log parameters, metrics, and artifacts to MLflow.

The `min_samples_split` is 2.

## Running MLFlow Tracking Server

You can run the MLflow tracking server with a specific backend store URI and an artifact location. For example:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts
```

## Tuning Hyperparameters with hyperopt

In hpo.py, `mlflow.log_params(params)` is used to capture all parameters used in the optimisation process. These are logged live to the MLflow tracking server.

```bash
python hpo.py --data_path ./output
```

The best (lowest) rsme is 5.335419588556921

## Promoting the best model to the registry

The script in register_model.py runs a set of runs with autolog enabled, and logs the val_rmse and test_rmse metrics. It then can search the MLflow experiment for the best run by the rmse metric, and register that model in the registry.

```bash
python register_model.py --data_path ./output
```

The run with the best (lowest) test_rmse is 5.567408012462019
