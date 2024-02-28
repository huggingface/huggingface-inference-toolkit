import tempfile
from tests.integ.helpers import verify_task
from tests.integ.config import (
    task2input,
    task2model,
    task2output,
    task2validation
)
from transformers.testing_utils import (
    require_torch,
    slow,
    _run_slow_tests
)
import pytest

class TestPytorchLocal:

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
            "sentence-similarity",
            "sentence-embeddings",
            "sentence-ranking",
            "text-to-image",
        ],
    )
    @pytest.mark.parametrize(
        "device",
        ["gpu"]
    )
    @pytest.mark.parametrize(
        "framework",
        ["pytorch"]
    )
    @pytest.mark.parametrize(
        "repository_id",
        [""]
    )
    @pytest.mark.usefixtures('local_container')
    def test_pt_container_local_model(
        self,
        local_container,
        task,
        framework,
        device
    ) -> None:

        verify_task(task = task, port = local_container[1])


    @require_torch
    @pytest.mark.parametrize(
        "repository_id",
        ["philschmid/custom-handler-test", "philschmid/custom-handler-distilbert"],
    )
    @pytest.mark.parametrize(
        "device",
        ["gpu"]
    )
    @pytest.mark.parametrize(
        "framework",
        ["pytorch"]
    )
    @pytest.mark.parametrize(
        "task",
        ["custom"]
    )
    @pytest.mark.usefixtures('local_container')
    def test_pt_container_custom_handler(
        self,
        local_container,
        task,
        device,
        repository_id
    ) -> None:
        
        verify_task(
            task = task,
            port = local_container[1],
        )


    @require_torch
    @pytest.mark.parametrize(
        "repository_id",
        ["philschmid/custom-pipeline-text-classification"],
    )
    @pytest.mark.parametrize(
        "device",
        ["gpu"]
    )
    @pytest.mark.parametrize(
        "framework",
        ["pytorch"]
    )
    @pytest.mark.parametrize(
        "task",
        ["custom"]
    )
    @pytest.mark.usefixtures('local_container')
    def test_pt_container_legacy_custom_pipeline(
        self,
        local_container,
        repository_id,
        device,
        task
    ) -> None:

        verify_task(task = task, port = local_container[1])
