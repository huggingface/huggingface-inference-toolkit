import logging
import pathlib
import subprocess
import sys
import tempfile
import textwrap

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


def test_importing_diffusers_utils_does_not_resolve_diffusers():
    """
    Importing this module must not touch `diffusers`.

    `from diffusers import AutoPipelineForText2Image` resolves through diffusers' lazy
    module and imports every pipeline, so a single unimportable pipeline module used to
    take down whatever imported us -- including worker startup, since `heavy_utils`
    imports this module during `download_model`. Run in a subprocess with a deliberately
    broken `diffusers` in place: the import has to survive it.
    """
    program = textwrap.dedent(
        """
        import sys, types
        from importlib.machinery import ModuleSpec

        broken = types.ModuleType("diffusers")
        broken.__spec__ = ModuleSpec("diffusers", loader=None)

        def _explode(name):
            # Dunders have to behave: torch's op registration walks sys.modules probing
            # for __file__, and a module that raises on that breaks unrelated imports.
            if name.startswith("__"):
                raise AttributeError(name)
            raise RuntimeError(
                "Failed to import diffusers.pipelines.auto_pipeline because of the following "
                "error: No module named 'transformers.masking_utils'"
            )

        broken.__getattr__ = _explode
        sys.modules["diffusers"] = broken

        import huggingface_inference_toolkit.diffusers_utils as d

        # The names must not have been pulled in at import time ...
        assert not hasattr(d, "AutoPipelineForText2Image"), "diffusers resolved at import"
        # ... and the module must still be usable for everything that is not a pipeline build.
        assert d.is_diffusers_available() is True
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=str(pathlib.Path(__file__).resolve().parents[2] / "src"),
    )
    assert result.returncode == 0, f"import broke on an unimportable diffusers:\n{result.stderr}"
    assert "OK" in result.stdout
