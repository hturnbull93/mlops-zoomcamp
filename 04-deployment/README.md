# 04 Deployment

## Web service

### Setting up the environment

```bash
conda create -n web-service python=3.10.13
```

Activate the environment:

```bash
conda activate web-service
```

### Setting up the pipenv virtual environment

From the web-service directory:

```bash
pip install pipenv
```

```bash
pipenv install scikit-learn==1.5.0 flask --python=3.10.13
```

Enter the pipenv shell:

```bash
pipenv shell
```

And exit with:

```bash
exit
```

### Running the web service

To run the web service, navigate to the `web-service` directory and execute:

```bash
python predict.py
```

Install requests as a dev dependency:

```bash
pipenv install requests --dev
```
Test with:

```bash
python test.py
```

## Running the web service in production

Install gunicorn:

```bash
pip install gunicorn
```
Run the web service with gunicorn:

```bash
gunicorn --bind=0.0.0.0:9696 predict:app
```

## Packaging into a Docker container

The Dockerfile contains the instructions to run the app in a container.

Build the Docker image with:

```bash
docker build -t ride-duration-prediction-service:v1 .
```

Run the Docker container with:

```bash
docker run -it --rm -p 9696:9696 ride-duration-prediction-service:v1
```

## Batch

### Setting up the environment

```bash
conda create -n batch python=3.10.13
```

Activate the environment:

```bash
conda activate batch
```

Install the requirements.txt

```bash
pip install -r ./requirements.txt
```

Preparing the df_result dataframe with the columns `ride_id` and `predicted_duration` results in a parquet file of size around 66MB


The notebook can be converted into a script with:

```bash
jupyter nbconvert --to script score.ipynb
```

### Set up with pipenv

```bash
pip install pipenv
```

enter the pipenv shell:

```bash
pipenv shell
```

The script can be run with:

```bash
python score.py --year 2023 --month 3
```
<!-- 
## Running the notebook



# Running the notebook

Running through the notebook interactively, the standard deviation of the predicted duration is 6.247488852238704

The ride_id column is prepared, which is written with the predictions to a new dataframe.
 -->

