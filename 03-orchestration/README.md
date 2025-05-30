# Orchestrating a training pipeline

## Wokflow

1. Ingestion
   1. getting data
2. Transforming
   1. filtering
   2. removing outliers
3. Preparing for ML
   1. feature engineering
   2. creating matrices
4. Hyperparameter tuning
   1. Finding the best hyperparameters
5. Train the final model
6. Promote to registry

## Tools

Airflow
Prefect
Mage
Dagster
Luigi

Tools that are specific for ML orchestration:

MLflow
Kubeflow

## Setting up the environment

```bash
conda create -n orchestration python=3.12
```

Activate the environment:

```bash
conda activate orchestration
```


Install the requirements.txt

```bash
pip install -r ./requirements.txt
```

Also on mac, xgboost requires some additional steps:

```bash
brew install libomp
```

# Run the MLflow Tracking Server

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts
```

## Run the pipeline

```bash
python duration-prediction.py --year=2023 --month=3
```