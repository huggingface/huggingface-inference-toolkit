import logging
import tempfile

from PIL import Image
from transformers.testing_utils import require_torch, slow

from huggingface_inference_toolkit.diffusers_utils import IEAutoPipelineForText2Image
from huggingface_inference_toolkit.heavy_utils import get_pipeline, load_repository_from_hf

logging.basicConfig(level="DEBUG")

@require_torch
def test_get_diffusers_pipeline():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = load_repository_from_hf(
            "echarlaix/tiny-random-stable-diffusion-xl",
            tmpdirname,
            framework="pytorch"
        )
        pipe = get_pipeline("text-to-image", storage_dir.as_posix())
        assert isinstance(pipe, IEAutoPipelineForText2Image)


@slow
@require_torch
def test_pipe_on_gpu():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = load_repository_from_hf(
            "echarlaix/tiny-random-stable-diffusion-xl",
            tmpdirname,
            framework="pytorch"
        )
        pipe = get_pipeline(
            "text-to-image",
            storage_dir.as_posix()
        )
        logging.error(f"Pipe: {pipe.pipeline}")
        assert pipe.pipeline.device.type == "cuda"


@require_torch
def test_text_to_image_task():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = load_repository_from_hf(
            "echarlaix/tiny-random-stable-diffusion-xl",
            tmpdirname,
            framework="pytorch"
        )
        pipe = get_pipeline("text-to-image", storage_dir.as_posix())
        res = pipe("Lets create an embedding")
        assert isinstance(res, Image.Image)
