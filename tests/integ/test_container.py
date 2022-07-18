import tempfile
import time

import docker
import pytest
import requests
from docker.client import DockerClient
from huggingface_inference_toolkit.utils import _load_repository_from_hf
from integ.config import task2input, task2model, task2output, task2validation


client = docker.from_env()


def make_sure_other_containers_are_stopped(client: DockerClient, container_name: str):
    try:
        previous = client.containers.get(container_name)
        previous.stop()
        previous.remove()
    except Exception:
        return None


def wait_for_container_to_be_ready(base_url):
    t = 0
    while t < 10:
        try:
            response = requests.get(f"{base_url}/health")
            if response.status_code == 200:
                break
        except Exception:
            pass
        finally:
            t += 1
            time.sleep(2)
    return True


def verify_task(container: DockerClient, task: str, framework: str = "pytorch"):
    BASE_URL = "http://localhost:5000"
    input = task2input[task]
    # health check
    wait_for_container_to_be_ready(BASE_URL)

    prediction = requests.post(f"{BASE_URL}/predict", json=input).json()
    assert task2validation[task](result=prediction, snapshot=task2output[task]) is True


@pytest.mark.parametrize(
    "task",
    [
        "text-classification",
        "zero-shot-classification",
        "ner",
        "question-answering",
        "fill-mask",
        "summarization",
        "translation_xx_to_yy",
        "text2text-generation",
        "text-generation",
        "feature-extraction",
        "image-classification",
        "automatic-speech-recognition",
    ],
)
def test_cpu_container_remote_model(task) -> None:
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


def test_cpu_container_local_model() -> None:
    container_name = "integration-test"
    container_image = "starlette-transformers:cpu"
    make_sure_other_containers_are_stopped(client, container_name)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # https://github.com/huggingface/infinity/blob/test-ovh/test/integ/utils.py
        storage_dir = _load_repository_from_hf("distilbert-base-uncased-finetuned-sst-2-english", tmpdirname)
        container = client.containers.run(
            container_image,
            name=container_name,
            ports={"5000": "5000"},
            environment={"HF_MODEL_DIR": tmpdirname, "HF_TASK": "text-classification"},
            volumes={tmpdirname: {"bind": tmpdirname, "mode": "rw"}},
            detach=True,
            # GPU
            # device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]
        )
        # time.sleep(5)
        verify_task(container, "text-classification")

        container.stop()
        container.remove()


@pytest.mark.parametrize(
    "repository_id",
    ["philschmid/custom-pipeline-text-classification"],
)
def test_cpu_container_custom_pipeline(repository_id) -> None:
    container_name = "integration-test"
    container_image = "starlette-transformers:cpu"
    make_sure_other_containers_are_stopped(client, container_name)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # https://github.com/huggingface/infinity/blob/test-ovh/test/integ/utils.py
        storage_dir = _load_repository_from_hf("philschmid/custom-pipeline-text-classification", tmpdirname)
        container = client.containers.run(
            container_image,
            name=container_name,
            ports={"5000": "5000"},
            environment={
                "HF_MODEL_DIR": tmpdirname,
            },
            volumes={tmpdirname: {"bind": tmpdirname, "mode": "ro"}},
            detach=True,
            # GPU
            # device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]
        )
        BASE_URL = "http://localhost:5000"
        wait_for_container_to_be_ready(BASE_URL)
        payload = {"inputs": "this is a test"}
        prediction = requests.post(f"{BASE_URL}/predict", json=payload).json()
        assert prediction == payload
        # time.sleep(5)
        container.stop()
        container.remove()
