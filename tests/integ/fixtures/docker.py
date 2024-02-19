import docker
import pytest
import random
import time
import logging


@pytest.fixture(scope = "module")
def start_container(
    device,
    task,
    model,
    framework
):
    client = docker.DockerClient(base_url='unix://var/run/docker.sock')
    container_name = f"integration-test-{framework}-{task}-{device}"
    container_image = f"integration-test-{framework}:{device}"
    port = random.randint(5000, 6000)

    logging.debug(f"Image: {container_image}")
    logging.debug(f"Port: {port}")

    previous = client.containers.get(container_name)
    if previous:
        previous.stop()
        previous.remove()

    device_request = [
        docker.types.DeviceRequest(
            count=-1,
            capabilities=[["gpu"]])
    ] if device == "gpu" else []

    container = client.containers.run(
        image = container_image,
        name=container_name,
        ports={"5000": port},
        environment={"HF_MODEL_ID": model, "HF_TASK": task},
        detach=True,
        # GPU
        device_requests=device_request,
    )

    return container_name, port

def stop_container(container_name):

    client = docker.DockerClient(base_url='unix://var/run/docker.sock')
    previous = client.containers.get(container_name)
    previous.stop()
    previous.remove()

