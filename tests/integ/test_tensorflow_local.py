import tempfile
from tests.integ.helpers import verify_task
from tests.integ.config import (
    task2input,
    task2model,
    task2output,
    task2validation
)
from transformers.testing_utils import (
    require_tf,
    slow,
    _run_slow_tests
)
import pytest


class TestTensorflowLocal:

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
            "conversational",
        ],
    )
    @pytest.mark.parametrize(
        "device",
        ["gpu", "cpu"]
    )
    @pytest.mark.parametrize(
        "framework",
        ["tensorflow"]
    )
    @pytest.mark.parametrize(
        "repository_id",
        [""]
    )
    @pytest.mark.usefixtures('local_container')
    def test_tf_container_local_model(
        self,
        local_container,
        task,
        framework,
        device
    ) -> None:

        verify_task(
            task = task,
            port = local_container[1],
            framework = framework
        )
