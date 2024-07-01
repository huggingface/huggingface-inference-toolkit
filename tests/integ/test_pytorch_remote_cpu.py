import docker
import pytest
import tenacity

from tests.integ.helpers import verify_task


class TestPytorchRemote:

    @tenacity.retry(
        retry=tenacity.retry_if_exception(docker.errors.APIError),
        stop=tenacity.stop_after_attempt(5),
        reraise=True,
    )
    @pytest.mark.parametrize("device", ["cpu"])
    @pytest.mark.parametrize(
        "task",
        [
            "text-classification",
            "zero-shot-classification",
            "question-answering",
            "fill-mask",
            "summarization",
            "ner",
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
            "sentence-ranking",
            "text-to-image",
        ],
    )
    @pytest.mark.parametrize("framework", ["pytorch"])
    @pytest.mark.usefixtures("remote_container")
    def test_inference_remote(self, remote_container, task, framework, device):

        verify_task(task=task, port=remote_container[1])
