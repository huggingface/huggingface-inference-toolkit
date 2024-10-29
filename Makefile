.PHONY: quality style unit-test integ-test inference-pytorch-gpu inference-pytorch-cpu inference-pytorch-inf2 stop-all

check_dirs := src tests

# Check that source code meets quality standards
quality:
	ruff check $(check_dirs)

# Format source code automatically
style:
	ruff check $(check_dirs) --fix

# Run unit tests
unit-test:
	RUN_SLOW=True python3 -m pytest -s -v tests/unit -n 10 --log-cli-level='ERROR'

# Run integration tests
integ-test:
	python3 -m pytest -s -v tests/integ/

# Build Docker image for PyTorch on GPU
inference-pytorch-gpu:
	docker build -f dockerfiles/pytorch/Dockerfile -t integration-test-pytorch:gpu .

# Build Docker image for PyTorch on CPU
inference-pytorch-cpu:
	docker build --build-arg="BASE_IMAGE=ubuntu:22.04" -f dockerfiles/pytorch/Dockerfile -t integration-test-pytorch:cpu .

# Build Docker image for PyTorch on AWS Inferentia2
inference-pytorch-inf2:
	docker build -f dockerfiles/pytorch/Dockerfile.inf2 -t integration-test-pytorch:inf2 .

# Stop all and prune/clean the Docker Containers 
stop-all:
	docker stop $$(docker ps -a -q) && docker container prune --force
