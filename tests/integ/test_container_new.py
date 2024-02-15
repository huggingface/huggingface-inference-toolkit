import random
import tempfile
import time

import docker
import pytest
import requests
from docker.client import DockerClient
from huggingface_inference_toolkit.utils import _is_gpu_available, _load_repository_from_hf
from integ.config import task2input, task2model, task2output, task2validation
from transformers.testing_utils import require_torch, slow, require_tf, _run_slow_tests

IS_GPU = _run_slow_tests
DEVICE = "gpu" if IS_GPU else "cpu"

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


def verify_task(task: str, port: int = 5000, framework: str = "pytorch"):
    BASE_URL = f"http://localhost:{port}"
    input = task2input[task]

    if (
        task == "image-classification"
        or task == "object-detection"
        or task == "image-segmentation"
        or task == "zero-shot-image-classification"
    ):
        prediction = requests.post(
            f"{BASE_URL}", data=task2input[task], headers={"content-type": "image/x-image"}
        ).json()
    elif task == "automatic-speech-recognition" or task == "audio-classification":
        prediction = requests.post(
            f"{BASE_URL}", data=task2input[task], headers={"content-type": "audio/x-audio"}
        ).json()
    elif task == "text-to-image":
        prediction = requests.post(f"{BASE_URL}", json=input, headers={"accept": "image/png"}).content
    else:
        prediction = requests.post(f"{BASE_URL}", json=input).json()
    assert task2validation[task](result=prediction, snapshot=task2output[task]) is True


@require_torch
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
        "audio-classification",
        "object-detection",
        "image-segmentation",
        "table-question-answering",
        "conversational",
        # TODO currently not supported due to multimodality input
        # "visual-question-answering",
        # "zero-shot-image-classification",
        "sentence-similarity",
        "sentence-embeddings",
        "sentence-ranking",
        # diffusers
        "text-to-image",
    ],
)
def test_pt_container_remote_model(task) -> None:

    framework = "pytorch"
    port = 5000 #random.randint(5000, 6000)

    verify_task(task, port, framework)