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

