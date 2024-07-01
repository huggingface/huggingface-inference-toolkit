import pytest
from huggingface_inference_toolkit.optimum_utils import is_optimum_neuron_available
from transformers.testing_utils import require_torch

from tests.integ.helpers import verify_task

require_inferentia = pytest.mark.skipif(
    not is_optimum_neuron_available(),
    reason="Skipping tests, since optimum neuron is not available or not running on inf2 instances.",
)


class TestPytorchLocal:
    @require_torch
    @require_inferentia
    @pytest.mark.parametrize(
        "task",
        [
            "feature-extraction",
            "fill-mask",
            "question-answering",
            "text-classification",
            "token-classification",
            "image-classification",
            "image-segmentation",
            "object-detection",
            "sentence-embeddings",
        ],
    )
    @pytest.mark.parametrize("device", ["inf2"])
    @pytest.mark.parametrize("framework", ["pytorch"])
    @pytest.mark.parametrize("repository_id", [""])
    @pytest.mark.usefixtures("local_container")
    def test_pt_container_local_model(self, local_container, task) -> None:

        verify_task(task=task, port=local_container[1])
