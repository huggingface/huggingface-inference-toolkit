#!/bin/bash

# The runtime image carries no compiler: build-essential, cmake, the Python headers and, on the
# CUDA variant, nvcc all live in the builder stage. Dropping them is most of what took the CPU
# image from 3.1G to 0.7G and the GPU one from 6.8G to 4.4G, and the majority of model
# repositories never needed them -- their requirements.txt resolves entirely to wheels.
#
# So install the toolchain on demand, and only for the repositories that turn out to need it.
# The trigger is a wheels-only pip pass that failed, not merely the presence of a requirements.txt:
# gating on the file existing would make the wheel-only majority pay the apt download too, while
# gating on a failed ordinary install would first sit through a build that was doomed anyway.
install_build_toolchain() {
    echo "Installing a build toolchain to satisfy the requirements from source"
    if ! apt-get update; then
        return 1
    fi

    # Ask the interpreter rather than restating the version the Dockerfile installs: the headers
    # have to match whichever Python /opt/venv was built on, and BASE_IMAGE is a build arg.
    local python_version
    python_version=$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])') || return 1

    local packages=(build-essential cmake pkg-config "python${python_version}-dev")

    # A CUDA extension (flash-attn, deepspeed, apex, ...) needs nvcc to compile and the libcuda
    # stub to link against. Neither is in the CUDA *runtime* base, and the stub is what
    # LIBRARY_PATH pointed at back when this image was built on the devel one.
    if [[ -n "${CUDA_VERSION:-}" ]]; then
        local cuda_apt_version="${CUDA_VERSION%.*}"   # 12.1.0 -> 12.1
        cuda_apt_version="${cuda_apt_version/./-}"    # 12.1   -> 12-1
        packages+=("cuda-nvcc-${cuda_apt_version}" "cuda-cudart-dev-${cuda_apt_version}")
        export CUDA_HOME=/usr/local/cuda
        export PATH="${CUDA_HOME}/bin:${PATH}"
        # Unset on the runtime base, where the devel one used to define it.
        export LIBRARY_PATH="${CUDA_HOME}/lib64/stubs${LIBRARY_PATH:+:${LIBRARY_PATH}}"
    fi

    apt-get install -y --no-install-recommends "${packages[@]}"
}

install_requirements() {
    local requirements=$1

    # --only-binary is both the fast path and the probe: pip either resolves the whole set from
    # wheels or refuses without attempting a single build, so nothing is half-installed before
    # the retry below. A pip failure here is expected and not fatal -- read the second pass.
    echo "Installing custom dependencies from ${requirements} (wheels only)"
    if pip install -r "${requirements}" --no-cache-dir --only-binary :all:; then
        return 0
    fi

    echo "Wheels alone did not satisfy ${requirements}, so something has to be built from source"
    if ! install_build_toolchain; then
        # Let pip run regardless: its own error names the package that cannot be satisfied, which
        # is more useful to the repository owner than an apt failure.
        echo "Warning: could not install a build toolchain, retrying the requirements anyway"
    fi

    pip install -r "${requirements}" --no-cache-dir
}

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

    echo "Downloading $filename for model ${HF_MODEL_ID}"
    hf download ${HF_MODEL_ID} "$filename" --revision "$revision" --local-dir /tmp

    # Check if handler.py was downloaded successfully
    if [ -f "/tmp/$filename" ]; then
        echo "$filename downloaded successfully, checking if there's a requirements.txt file..."
        rm /tmp/$filename

        # Attempt to download requirements.txt
        echo "Downloading requirements.txt for model ${HF_MODEL_ID}"
        hf download "${HF_MODEL_ID}" requirements.txt --revision "$revision" --local-dir /tmp

        # Check if requirements.txt was downloaded successfully
        if [ -f "/tmp/requirements.txt" ]; then
            echo "requirements.txt downloaded successfully, now installing the dependencies..."

            install_requirements /tmp/requirements.txt
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
        install_requirements "${HF_MODEL_DIR}/requirements.txt"
    fi
fi

# Start the server.
# gunicorn rather than uvicorn directly: WORKERS lets a node host several workers per GPU, which is
# what makes idle unloading worth it, and --graceful-timeout gives a worker time to finish the
# inference it is running instead of being killed mid-request (gunicorn's own default is 30s).
exec gunicorn webservice_starlette:app \
  -k uvicorn.workers.UvicornWorker \
  --workers ${WORKERS:-1} \
  --bind 0.0.0.0:${PORT} \
  --timeout ${TIMEOUT:-30} \
  --graceful-timeout ${GRACEFUL_TIMEOUT:-300}
