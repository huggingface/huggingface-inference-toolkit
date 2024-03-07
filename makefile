.PHONY: quality style unit-test integ-test

check_dirs := src 

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

inference-pytorch-neuron:
	docker build --build-arg=BASE_IMAGE=ubuntu:22.04 --build-arg=NEURONX=1 -f dockerfiles/pytorch/Dockerfile  -t integration-test-pytorch:neuron .

stop-all:
	docker stop $$(docker ps -a -q) && docker container prune --force
