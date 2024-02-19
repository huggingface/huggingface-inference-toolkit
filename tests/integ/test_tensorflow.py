from tests.integ.fixtures.docker import start_container, stop_container
from tests.integ.helpers import verify_task
from tests.integ.config import (
    task2input,
    task2model,
    task2output,
    task2validation
)
import pytest

class TestTensorflowInference:

    @pytest.mark.parametrize(
        "device",
        ["gpu", "cpu"]
    )
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
            "sentence-similarity",
            "sentence-embeddings",
            "sentence-ranking"
        ]
    )
    @pytest.mark.parametrize(
        "framework",
        ["tensorflow"]
    )
    def test_classification(self, start_container, task):

        verify_task(
            task = task,
            port = start_container[1]
        )

