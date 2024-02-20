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

inference-tensorflow-gpu:
	docker build --no-cache -f dockerfiles/tensorflow/gpu/Dockerfile -t integration-test-tensorflow:gpu .

inference-tensorflow-cpu:
	docker build -f dockerfiles/tensorflow/cpu/Dockerfile -t integration-test-tensorflow:cpu .

stop-all:
	docker stop $$(docker ps -a -q) && docker container prune --force

run-tensorflow-remote-gpu:
	docker run -e HF_TASK=text-classification -e HF_MODEL_ID=distilbert/distilbert-base-uncased integration-test-tensorflow:gpu

run-tensorflow-local-gpu:
	rm -rf /tmp/distilbert && \
	huggingface-cli download hf-internal-testing/tiny-random-distilbert --local-dir /tmp/distilbert && \
	docker run --gpus all \
		-v /tmp/distilbert:/opt/huggingface/model \
		-e HF_MODEL_DIR=/opt/huggingface/model \
		-e HF_TASK=text-classification \
		integration-test-tensorflow:gpu