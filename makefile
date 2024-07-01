.PHONY: quality style unit-test integ-test

check_dirs := src tests 

# run tests

unit-test:
	RUN_SLOW=True python3 -m pytest -s -v tests/unit -n 10 --log-cli-level='ERROR'

integ-test:
	python3 -m pytest -s -v tests/integ/

# Check that source code meets quality standards

quality:
	ruff $(check_dirs) 

# Format source code automatically

style: 
	ruff $(check_dirs) --fix

inference-pytorch-gpu:
	docker build -f dockerfiles/pytorch/Dockerfile -t integration-test-pytorch:gpu .

inference-pytorch-cpu:
	docker build --build-arg="BASE_IMAGE=ubuntu:22.04" -f dockerfiles/pytorch/Dockerfile -t integration-test-pytorch:cpu .

inference-pytorch-inf2:
	docker build -f dockerfiles/pytorch/Dockerfile.inf2 -t integration-test-pytorch:inf2 .

vertex-pytorch-gpu:
	docker build -t vertex -f dockerfiles/pytorch/Dockerfile -t integration-test-pytorch:gpu .

vertex-pytorch-cpu:
	docker build  -t vertex --build-arg="BASE_IMAGE=ubuntu:22.04" -f dockerfiles/pytorch/Dockerfile -t integration-test-pytorch:cpu .

stop-all:
	docker stop $$(docker ps -a -q) && docker container prune --force