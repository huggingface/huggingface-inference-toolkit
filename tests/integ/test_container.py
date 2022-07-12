import json
import os
import re
import tempfile
import time

import numpy as np
import pytest
from integ.config import task2input, task2model, task2output, task2validation
import docker
import requests
from docker.client import DockerClient

client = docker.from_env()


def make_sure_other_containers_are_stopped(client: DockerClient, container_name: str):
    try:
        previous = client.containers.get(container_name)
        previous.stop()
        previous.remove()
    except Exception:
        return None


def verify_task(container: DockerClient, task: str, framework: str = "pytorch"):
    BASE_URL = "http://localhost:5000"
    model = task2model[task][framework]
    input = task2input[task]
    # health check
    t = 0
    while t < 10:
        try:
            response = requests.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                break
        except Exception:
            pass
        finally:
            t += 1
            time.sleep(2)

    prediction = requests.post(f"{BASE_URL}/predict", json=input).json()
    assert task2validation[task](result=prediction, snapshot=task2output[task]) == True


@pytest.mark.parametrize(
    "task",
    [
        "text-classification",
        # "zero-shot-classification",
        # "ner",
        # "question-answering",
        # "fill-mask",
        # "summarization",
        # "translation_xx_to_yy",
        # "text2text-generation",
        # "text-generation",
        # "feature-extraction",
        # "image-classification",
        # "automatic-speech-recognition",
    ],
)
def test_cpu_container(task) -> None:
    container_name = "integration-test"
    container_image = "starlette-transformers:cpu"
    make_sure_other_containers_are_stopped(client, container_name)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # https://github.com/huggingface/infinity/blob/test-ovh/test/integ/utils.py
        container = client.containers.run(
            container_image,
            name=container_name,
            ports={"5000": "5000"},
            environment={"HF_MODEL_ID": "/opt/huggingface/model", "HF_TASK": task},
            volumes={tmpdirname: {"bind": "/opt/huggingface/model", "mode": "rw"}},
            detach=True,
            # GPU
            # device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]
        )
        # time.sleep(5)
        verify_task(container, task)
        container.stop()
        container.remove()
