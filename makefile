.PHONY: quality style unit-test integ-test

check_dirs := src 

# run tests

unit-test:
	python3 -m pytest -s -v ./tests/unit

integ-test:
	python3 -m pytest -s -v ./tests/integ/

# Check that source code meets quality standards

quality:
	ruff $(check_dirs) 

# Format source code automatically

style: 
	ruff $(check_dirs) --fix

torch-gpu:
	docker build --no-cache -f dockerfiles/pytorch/gpu/Dockerfile -t starlette-transformers:gpu .

torch-cpu:
	docker build -f dockerfiles/pytorch/cpu/Dockerfile -t starlette-transformers:cpu .

run-classification:
	docker run -e HF_MODEL="hf-internal-testing/tiny-random-distilbert" -e HF_MODEL_DIR="/tmp2" -e HF_TASK="text-classification" --gpus all starlette-transformers:gpu