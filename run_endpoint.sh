#!/usr/bin/env bash

export KMP_HW_SUBSET=${OMP_NUM_THREADS}c,1t
export MKL_NUM_THREADS=${OMP_NUM_THREADS}

echo "Hardware setup:"
echo "| - NUM_THREADS:       ${MKL_NUM_THREADS}"
echo "| - HARDWARE_AFFINITY: ${KMP_HW_SUBSET}"
echo "| - AFFINITY:          ${KMP_AFFINITY}"

uvicorn huggingface_inference_toolkit.webservice_starlette:app --host ${ENDPOINT_IF} --port ${ENDPOINT_PORT}