from tests.integ.fixtures.docker import start_container, stop_container
from tests.integ.helpers import verify_task
from tests.integ.config import (
    task2input,
    task2model,
    task2output,
    task2validation
)
import pytest
import time
import tenacity

class TestTextClassification:

    @pytest.mark.parametrize(
        "device",
        ["gpu"]
    )
    @pytest.mark.parametrize(
        "task",
        ["text-classification"]
    )
    @pytest.mark.parametrize(
        "model",
        [task2model["text-classification"]["pytorch"]]
    )
    @pytest.mark.parametrize(
        "framework",
        ["pytorch"]
    )
    def test_classification(start_container):

        time.sleep(5)
        verify_task(
            task = "text-classification",
            port = start_container[1]
        )

