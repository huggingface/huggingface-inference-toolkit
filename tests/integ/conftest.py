import docker
import pytest
import random
import logging
from tests.integ.config import task2model


@pytest.fixture(scope = "function")
def start_container(
    device,
    task,
    framework
):
    client = docker.DockerClient(base_url='unix://var/run/docker.sock')
    container_name = f"integration-test-{framework}-{task}-{device}"
    container_image = f"integration-test-{framework}:{device}"
    port = random.randint(5000, 6000)
    model = task2model[task][framework]

    logging.debug(f"Image: {container_image}")
    logging.debug(f"Port: {port}")

    device_request = [
        docker.types.DeviceRequest(
            count=-1,
            capabilities=[["gpu"]])
    ] if device == "gpu" else []

    yield client.containers.run(
        image = container_image,
        name=container_name,
        ports={"5000": port},
        environment={
            "HF_MODEL_ID": model,
            "HF_TASK": task,
            "CUDA_LAUNCH_BLOCKING": 1
        },
        detach=True,
        # GPU
        device_requests=device_request,
    ), port

    #Teardown
    previous = client.containers.get(container_name)
    previous.stop()
    previous.remove()

