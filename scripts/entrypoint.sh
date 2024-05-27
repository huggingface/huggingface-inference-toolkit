#!/bin/bash

# Define the default port
PORT=5000

# Check if AIP_MODE is set and adjust the port for Vertex AI
if [[ ! -z "${AIP_MODE}" ]]; then
  PORT=${AIP_HTTP_PORT}
fi

# Check if HF_MODEL_DIR is set and if not skip installing custom dependencies
if [[ ! -z "${HF_MODEL_DIR}" ]]; then
  # Check if requirements.txt exists and if so install dependencies
  if [ -f "${HF_MODEL_DIR}/requirements.txt" ]; then
    echo "Installing custom dependencies from ${HF_MODEL_DIR}/requirements.txt"
    pip install -r ${HF_MODEL_DIR}/requirements.txt --no-cache-dir;
  fi
fi

# Start the server
uvicorn webservice_starlette:app --host 0.0.0.0 --port ${PORT}
