import pickle
import pandas as pd
# import uuid
import numpy as np
import argparse
import os

# Ensure the output directory exists
os.makedirs('output', exist_ok=True)
# Create subdirectory for taxi type if it doesn't exist
os.makedirs('output/yellow', exist_ok=True)
os.makedirs('output/green', exist_ok=True)

# Create the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, required=True)
parser.add_argument('--month', type=int, required=True)

args = parser.parse_args()
year = args.year
month = args.month

# taxi_type can be 'yellow' or 'green'
taxi_type = 'yellow' 

input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
output_file = f'output/{taxi_type}/_tripdata_{year}-{month:02d}.parquet'

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    # n = len(df)
    # ride_ids = [str(uuid.uuid4()) for _ in range(n)]
    # df['ride_id'] = ride_ids
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


    return df

def apply_model(input_file, model, output_file):
    print(f'Reading data from {input_file}')
    df = read_data(input_file)

    print('Applying model...')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    std_pred = np.std(y_pred)
    print(f"Standard deviation of predicted duration: {std_pred}")

    df_result = pd.DataFrame()

    df_result['ride_id'] = df['ride_id']
    # df_result['tpep_pickup_datetime'] = df['tpep_pickup_datetime']
    # df_result['PULocationID'] = df['PULocationID']
    # df_result['DOLocationID'] = df['DOLocationID']
    # df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    # df_result['diff'] = df_result['predicted_duration'] - df_result['actual_duration']

    print('Mean predicted duration:', df_result['predicted_duration'].mean())

    print(f'Saving results to {output_file}')
    df_result.to_parquet(
        output_file,
        engine='pyarrow', 
        compression=None,
        index=False
    )
    

if __name__ == '__main__':
    apply_model(input_file, model, output_file)
