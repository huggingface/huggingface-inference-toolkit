import tempfile
from transformers.testing_utils import require_torch, slow

from huggingface_inference_toolkit.utils import _load_repository_from_hf, get_pipeline
from huggingface_inference_toolkit.accelerate_utils import check_support_for_model_parallelism


@require_torch
def test_model_parallelism_check():
    # supported architecture
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_repository_from_hf(
            "hf-internal-testing/tiny-random-GPT2LMHeadModel", tmpdirname, framework="pytorch"
        )
        is_supported = check_support_for_model_parallelism(storage_dir.as_posix())
        assert is_supported == True

    # not supported architecture
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_repository_from_hf(
            "hf-internal-testing/tiny-random-DistilBertModel", tmpdirname, framework="pytorch"
        )
        is_supported = check_support_for_model_parallelism(storage_dir.as_posix())
        assert is_supported == False
