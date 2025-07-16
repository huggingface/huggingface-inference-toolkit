#!/bin/bash

# Set the default port
PORT=5000

# Check if AIP_MODE is set and adjust the port for Vertex AI
if [[ ! -z "${AIP_MODE}" ]]; then
    PORT=${AIP_HTTP_PORT}
fi

# Check that only one of HF_MODEL_ID or HF_MODEL_DIR is provided
if [[ ! -z "${HF_MODEL_ID}" && ! -z "${HF_MODEL_DIR}" ]]; then
    echo "Error: Both HF_MODEL_ID and HF_MODEL_DIR are set. Please provide only one."
    exit 1
elif [[ -z "${HF_MODEL_ID}" && -z "${HF_MODEL_DIR}" ]]; then
    echo "Error: Neither HF_MODEL_ID nor HF_MODEL_DIR is set. Please provide one of them."
    exit 1
fi

# If HF_MODEL_ID is provided, download handler.py and requirements.txt if available
if [[ ! -z "${HF_MODEL_ID}" ]]; then
    filename=${HF_DEFAULT_PIPELINE_NAME:-handler.py}
    revision=${HF_REVISION:-main}

    if [[ ! -z "${HF_HUB_TOKEN}" ]]; then
        echo "Attempting HuggingFace CLI login"
        huggingface-cli login --token $HF_HUB_TOKEN
    else
        echo "HF_HUB_TOKEN not provided"
    fi

    echo "Downloading $filename for model ${HF_MODEL_ID}"
    huggingface-cli download ${HF_MODEL_ID} "$filename" --revision "$revision" --local-dir /tmp

    # Check if handler.py was downloaded successfully
    if [ -f "/tmp/$filename" ]; then
        echo "$filename downloaded successfully, checking if there's a requirements.txt file..."
        rm /tmp/$filename

        # Attempt to download requirements.txt
        echo "Downloading requirements.txt for model ${HF_MODEL_ID}"
        huggingface-cli download "${HF_MODEL_ID}" requirements.txt --revision "$revision" --local-dir /tmp

        # Check if requirements.txt was downloaded successfully
        if [ -f "/tmp/requirements.txt" ]; then
            echo "requirements.txt downloaded successfully, now installing the dependencies..."
            
            # Install dependencies
            pip install -r /tmp/requirements.txt --no-cache-dir
            rm /tmp/requirements.txt
        else
            echo "${HF_MODEL_ID} with revision $revision contains a custom handler at $filename but doesn't contain a requirements.txt file, so skipping downloading and installing extra requirements from it."
        fi
    else
        echo "${HF_MODEL_ID} with revision $revision doesn't contain a $filename file, so skipping download."
    fi
fi

# If HF_MODEL_DIR is provided, check for requirements.txt and install dependencies if available
if [[ ! -z "${HF_MODEL_DIR}" ]]; then
    # Check if requirements.txt exists and if so install dependencies
    if [ -f "${HF_MODEL_DIR}/requirements.txt" ]; then
        echo "Installing custom dependencies from ${HF_MODEL_DIR}/requirements.txt"
        pip install -r ${HF_MODEL_DIR}/requirements.txt --no-cache-dir
    fi
fi

# Start the server
exec uvicorn webservice_starlette:app --host 0.0.0.0 --port ${PORT}
