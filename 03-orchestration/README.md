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

MLflow version is 2.22.0

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts
```

## Run the pipeline

```bash
python duration-prediction.py --year=2023 --month=3
```

The number of rows in the March 2023 dataset is 3403766.

After filtering the trips of less than 1 minute and more than 60 minutes, the number of rows is 3316216.

The intercept of the linear regression is 23.848428331734578.

The model model_size_bytes is 10340