import pickle
from pathlib import Path
import pandas as pd
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

def read_dataframe(year: int, month: int):
    df = pd.read_parquet(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet')

    # Calculate the difference between pickup and dropoff times
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    # Convert duration to minutes
    df['duration'] = df['duration'].dt.total_seconds() / 60
    # remove less than 1 minute and more than 60 minutes
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    # set the categories
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    # Make pairs of PU and DO locations
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df

def create_X(df: pd.DataFrame, dv=None):
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    # Create a dictionary for each row
    dicts = df[categorical + numerical].to_dict(orient='records')
    
    # If DictVectorizer is not provided, create a new one
    if(dv is None):
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


def train_model(X_train, y_train, X_valid, y_valid, dv):
    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_valid, label=y_valid)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.0018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42,
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50,
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_valid, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open('models/preprocessor.b', 'wb') as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact('models/preprocessor.b', artifact_path='preprocessor')

        mlflow.xgboost.log_model(booster, artifact_path='models_mlflow')

def run(year, month):
    df_train = read_dataframe(year, month)

    # Read the next month for validation
    if month == 12:
        year += 1
        month = 1
    else:
        month += 1

    df_val = read_dataframe(year, month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)
    
    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    train_model(X_train, y_train, X_val, y_val, dv)


if __name__ == "__main__":
    #  use argparse to get year and month from command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()
    run(args.year, args.month)
