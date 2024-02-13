# /bin/bash

# check if HF_MODEL_DIR is set and if not skip installing custom dependencies
if [[ ! -z "${HF_MODEL_DIR}" ]]; then
  # check if requirements.txt exists and if so install dependencies
  if [ -f "${HF_MODEL_DIR}/requirements.txt" ]; then
    echo "Installing custom dependencies from ${HF_MODEL_DIR}/requirements.txt"
    pip install -r ${HF_MODEL_DIR}/requirements.txt --no-cache-dir;
  fi
fi

# start the server
python3 -m uvicorn webservice_starlette:app --host 0.0.0.0 --port 5000