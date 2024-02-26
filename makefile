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

inference-pytorch-gpu:
	docker build -f dockerfiles/pytorch/gpu/Dockerfile -t integration-test-pytorch:gpu .

inference-pytorch-cpu:
	docker build -f dockerfiles/pytorch/cpu/Dockerfile -t integration-test-pytorch:cpu .

stop-all:
	docker stop $$(docker ps -a -q) && docker container prune --force