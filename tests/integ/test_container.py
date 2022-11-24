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


def verify_task(container: DockerClient, task: str, port: int = 5000, framework: str = "pytorch"):
    BASE_URL = f"http://localhost:{port}"
    input = task2input[task]
    # health check
    wait_for_container_to_be_ready(BASE_URL)
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
    container_name = f"integration-test-{task}"
    container_image = f"starlette-transformers:{DEVICE}"
    framework = "pytorch"
    model = task2model[task][framework]
    port = random.randint(5000, 6000)
    device_request = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])] if IS_GPU else []

    make_sure_other_containers_are_stopped(client, container_name)
    container = client.containers.run(
        container_image,
        name=container_name,
        ports={"5000": port},
        environment={"HF_MODEL_ID": model, "HF_TASK": task},
        detach=True,
        # GPU
        device_requests=device_request,
    )
    # time.sleep(5)

    verify_task(container, task, port)
    container.stop()
    container.remove()


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
def test_pt_container_local_model(task) -> None:
    container_name = f"integration-test-{task}"
    container_image = f"starlette-transformers:{DEVICE}"
    framework = "pytorch"
    model = task2model[task][framework]
    port = random.randint(5000, 6000)
    device_request = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])] if IS_GPU else []
    make_sure_other_containers_are_stopped(client, container_name)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # https://github.com/huggingface/infinity/blob/test-ovh/test/integ/utils.py
        storage_dir = _load_repository_from_hf(model, tmpdirname, framework="pytorch")
        container = client.containers.run(
            container_image,
            name=container_name,
            ports={"5000": port},
            environment={"HF_MODEL_DIR": "/opt/huggingface/model", "HF_TASK": task},
            volumes={tmpdirname: {"bind": "/opt/huggingface/model", "mode": "ro"}},
            detach=True,
            # GPU
            device_requests=device_request,
        )
        # time.sleep(5)
        verify_task(container, task, port)
        container.stop()
        container.remove()


@require_torch
@pytest.mark.parametrize(
    "repository_id",
    ["philschmid/custom-handler-test", "philschmid/custom-handler-distilbert"],
)
def test_pt_container_custom_handler(repository_id) -> None:
    container_name = "integration-test-custom"
    container_image = f"starlette-transformers:{DEVICE}"
    device_request = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])] if IS_GPU else []
    port = random.randint(5000, 6000)

    make_sure_other_containers_are_stopped(client, container_name)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # https://github.com/huggingface/infinity/blob/test-ovh/test/integ/utils.py
        storage_dir = _load_repository_from_hf(repository_id, tmpdirname)
        container = client.containers.run(
            container_image,
            name=container_name,
            ports={"5000": port},
            environment={
                "HF_MODEL_DIR": tmpdirname,
            },
            volumes={tmpdirname: {"bind": tmpdirname, "mode": "ro"}},
            detach=True,
            # GPU
            device_requests=device_request,
        )
        BASE_URL = f"http://localhost:{port}"
        wait_for_container_to_be_ready(BASE_URL)
        payload = {"inputs": "this is a test"}
        prediction = requests.post(f"{BASE_URL}", json=payload).json()
        assert prediction == payload
        # time.sleep(5)
        container.stop()
        container.remove()


@require_torch
@pytest.mark.parametrize(
    "repository_id",
    ["philschmid/custom-pipeline-text-classification"],
)
def test_pt_container_legacy_custom_pipeline(repository_id) -> None:
    container_name = "integration-test-custom"
    container_image = f"starlette-transformers:{DEVICE}"
    device_request = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])] if IS_GPU else []
    port = random.randint(5000, 6000)

    make_sure_other_containers_are_stopped(client, container_name)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # https://github.com/huggingface/infinity/blob/test-ovh/test/integ/utils.py
        storage_dir = _load_repository_from_hf(repository_id, tmpdirname)
        container = client.containers.run(
            container_image,
            name=container_name,
            ports={"5000": port},
            environment={
                "HF_MODEL_DIR": tmpdirname,
            },
            volumes={tmpdirname: {"bind": tmpdirname, "mode": "ro"}},
            detach=True,
            # GPU
            device_requests=device_request,
        )
        BASE_URL = f"http://localhost:{port}"
        wait_for_container_to_be_ready(BASE_URL)
        payload = {"inputs": "this is a test"}
        prediction = requests.post(f"{BASE_URL}", json=payload).json()
        assert prediction == payload
        # time.sleep(5)
        container.stop()
        container.remove()


@require_tf
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
    ],
)
def test_tf_container_remote_model(task) -> None:
    container_name = f"integration-test-{task}"
    container_image = f"starlette-transformers:{DEVICE}"
    framework = "tensorflow"
    model = task2model[task][framework]
    device_request = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])] if IS_GPU else []
    if model is None:
        pytest.skip("no supported TF model")
    port = random.randint(5000, 6000)
    make_sure_other_containers_are_stopped(client, container_name)
    container = client.containers.run(
        container_image,
        name=container_name,
        ports={"5000": port},
        environment={"HF_MODEL_ID": model, "HF_TASK": task},
        detach=True,
        # GPU
        device_requests=device_request,
    )
    # time.sleep(5)
    verify_task(container, task, port)
    container.stop()
    container.remove()


@require_tf
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
    ],
)
def test_tf_container_local_model(task) -> None:
    container_name = f"integration-test-{task}"
    container_image = f"starlette-transformers:{DEVICE}"
    framework = "tensorflow"
    model = task2model[task][framework]
    device_request = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])] if IS_GPU else []
    if model is None:
        pytest.skip("no supported TF model")
    port = random.randint(5000, 6000)
    make_sure_other_containers_are_stopped(client, container_name)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # https://github.com/huggingface/infinity/blob/test-ovh/test/integ/utils.py
        storage_dir = _load_repository_from_hf(model, tmpdirname, framework=framework)
        container = client.containers.run(
            container_image,
            name=container_name,
            ports={"5000": port},
            environment={"HF_MODEL_DIR": "/opt/huggingface/model", "HF_TASK": task},
            volumes={tmpdirname: {"bind": "/opt/huggingface/model", "mode": "ro"}},
            detach=True,
            # GPU
            device_requests=device_request,
        )
        # time.sleep(5)
        verify_task(container, task, port)
        container.stop()
        container.remove()


# @require_tf
# @pytest.mark.parametrize(
#     "repository_id",
#     ["philschmid/custom-pipeline-text-classification"],
# )
# def test_tf_cpu_container_custom_pipeline(repository_id) -> None:
#     container_name = "integration-test-custom"
#     container_image = "starlette-transformers:cpu"
#     make_sure_other_containers_are_stopped(client, container_name)
#     with tempfile.TemporaryDirectory() as tmpdirname:
#         # https://github.com/huggingface/infinity/blob/test-ovh/test/integ/utils.py
#         storage_dir = _load_repository_from_hf("philschmid/custom-pipeline-text-classification", tmpdirname)
#         container = client.containers.run(
#             container_image,
#             name=container_name,
#             ports={"5000": "5000"},
#             environment={
#                 "HF_MODEL_DIR": tmpdirname,
#             },
#             volumes={tmpdirname: {"bind": tmpdirname, "mode": "ro"}},
#             detach=True,
#             # GPU
#             # device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]
#         )
#         BASE_URL = "http://localhost:5000"
#         wait_for_container_to_be_ready(BASE_URL)
#         payload = {"inputs": "this is a test"}
#         prediction = requests.post(f"{BASE_URL}", json=payload).json()
#         assert prediction == payload
#         # time.sleep(5)
#         container.stop()
#         container.remove()
