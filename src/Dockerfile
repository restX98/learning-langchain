FROM python:3.10.11-slim-bullseye

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

### SYSTEM SETUP ###
RUN apt-get -y update && apt-get install -y curl build-essential fastjar libmagic-mgc libmagic1 mime-support && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

### PREPARE BUILD WITH NECESSARY FILES AND FOLDERS ###
COPY ./pyproject.toml /app/pyproject.toml

### INSTALL PYTHON DEPENDENCIES (Core) ###
WORKDIR /app
# update pip and install dependencies from pyproject.tml
RUN pip install -U pip && \
    pip install --no-cache-dir . 

### COPY CAT CODE INSIDE THE CONTAINER (so it can be run standalone) ###
COPY ./core /app/core
