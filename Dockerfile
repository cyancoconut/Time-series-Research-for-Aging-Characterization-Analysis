# METAbatt CLI pipeline image (headless).
# Runs src/main.py and the other `python -m ...` entry points.
# The customtkinter GUI (pipeline_ui.py) is intentionally NOT supported here
# (it needs X11); tensorflow/keras are excluded (the pipeline's HDBSCAN path
# never calls the autoencoder methods that import them).

FROM python:3.12-slim

# System libraries:
#  - build-essential: compiles hdbscan (Cython) if no wheel is available
#  - unixodbc-dev: required by pyodbc
#  - libgomp1: OpenMP runtime used by scikit-learn / hdbscan at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        unixodbc-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY requirements_docker.txt ./
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy the application code. This is the only layer that re-runs on a code edit;
# during active development you can override it with a bind mount:
#   -v $(pwd)/src:/app/src
COPY src/ ./src/

# Imports are relative to src/, and config.json is resolved as ../config.json
# from here (mount it at /app/config.json).
WORKDIR /app/src

ENTRYPOINT ["python", "main.py"]
# Default arg: a battery config mounted into the container. Override at runtime,
# e.g.  docker run ... metabatt /app/battery_config.json --cells VTC_cell01
CMD ["/app/battery_config.json"]
