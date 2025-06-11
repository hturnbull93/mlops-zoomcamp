# Base image: Uses Python 3.10 slim version
# 'slim' is a minimal version of the Python image to keep container size small
FROM python:3.10-slim

# Upgrade pip to the latest version
# -U flag means upgrade to the latest version
RUN pip install -U pip

# Install pipenv for dependency management
RUN pip install pipenv

# Set the working directory inside the container to /app
# All subsequent commands will be run from this directory
WORKDIR /app

# Copy dependency files from host to container
# First dot: source path (host)
# Second dot: destination path (container's WORKDIR)
COPY Pipfile Pipfile.lock ./

# Install project dependencies using pipenv
# --system: Install packages into system python instead of creating a virtual environment
# --deploy: Ensure Pipfile.lock matches Pipfile
RUN pipenv install --system --deploy

# Copy application files to container
# predict.py: Main application file
# model.bin: Machine learning model file
COPY predict.py model.bin ./

# Declare that container will listen on port 9696
EXPOSE 9696

# Start the application using gunicorn server
# --bind=0.0.0.0:9696: Listen on all network interfaces on port 9696
# predict:app: The Flask application object (app) in predict.py
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]