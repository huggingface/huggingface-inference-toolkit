from tests.integ.helpers import verify_task
from tests.integ.config import (
    task2input,
    task2model,
    task2output,
    task2validation
)
import pytest
import tenacity
import docker

class TestTensorflowInference:

    @tenacity.retry(
        retry = tenacity.retry_if_exception(docker.errors.APIError),
        stop = tenacity.stop_after_attempt(3)
    )
    @pytest.mark.parametrize(
        "device",
        ["gpu"]
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
    @pytest.mark.usefixtures('start_container')
    def test_inference_gpu(self, start_container, task, framework, device):

        verify_task(task = task, port = start_container[1])

"""
    @pytest.mark.parametrize(
        "device",
        ["cpu"]
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
    @pytest.mark.usefixtures('start_container')
    def test_inference_cpu(self, start_container, task, framework, device):

        verify_task(task = task, port = start_container[1])
"""