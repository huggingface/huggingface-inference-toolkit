import importlib.util
import os
from typing import Union

from transformers.utils.import_utils import is_torch_bf16_gpu_available

from huggingface_inference_toolkit.logging import logger

_diffusers = importlib.util.find_spec("diffusers") is not None


def is_diffusers_available():
    return _diffusers


if is_diffusers_available():
    import torch
    from diffusers import (
        AutoPipelineForText2Image,
        DPMSolverMultistepScheduler,
        StableDiffusionPipeline,
    )


class IEAutoPipelineForText2Image:
    def __init__(self, model_dir: str, device: Union[str, None] = None, **kwargs):  # needs "cuda" for GPU
        dtype = torch.float32
        if device == "cuda":
            dtype = torch.bfloat16 if is_torch_bf16_gpu_available() else torch.float16
        device_map = "balanced" if device == "cuda" else None

        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            model_dir, torch_dtype=dtype, device_map=device_map, **kwargs
        )
        # try to use DPMSolverMultistepScheduler
        if isinstance(self.pipeline, StableDiffusionPipeline):
            try:
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
            except Exception:
                pass

    def __call__(
        self,
        prompt,
        **kwargs,
    ):
        if "prompt" in kwargs:
            logger.warning(
                "prompt has been provided twice, both via arg and kwargs, so the `prompt` arg will be used "
                "instead, and the `prompt` in kwargs will be discarded."
            )
            kwargs.pop("prompt")

        # diffusers doesn't support seed but rather the generator kwarg
        # see: https://github.com/huggingface/api-inference-community/blob/8e577e2d60957959ba02f474b2913d84a9086b82/docker_images/diffusers/app/pipelines/text_to_image.py#L172-L176
        if "seed" in kwargs:
            seed = int(kwargs["seed"])
            generator = torch.Generator().manual_seed(seed)
            kwargs["generator"] = generator
            kwargs.pop("seed")

        # TODO: add support for more images (Reason is correct output)
        if "num_images_per_prompt" in kwargs:
            kwargs.pop("num_images_per_prompt")
            logger.warning("Sending num_images_per_prompt > 1 to pipeline is not supported. Using default value 1.")

        if "num_inference_steps" not in kwargs:
            kwargs["num_inference_steps"] = int(os.environ.get("DEFAULT_NUM_INFERENCE_STEPS", 50))

        if "target_size" in kwargs:
            kwargs["height"] = kwargs["target_size"].pop("height", None)
            kwargs["width"] = kwargs["target_size"].pop("width", None)
            kwargs.pop("target_size")

        if "output_type" in kwargs and kwargs["output_type"] != "pil":
            kwargs.pop("output_type")
            logger.warning("The `output_type` cannot be modified, and PIL will be used by default instead.")

        # Call pipeline with parameters
        out = self.pipeline(prompt, num_images_per_prompt=1, **kwargs)

        return out.images[0]


DIFFUSERS_TASKS = {
    "text-to-image": IEAutoPipelineForText2Image,
}


def get_diffusers_pipeline(task=None, model_dir=None, device=-1, **kwargs):
    """Get a pipeline for Diffusers models."""
    device = "cuda" if device == 0 else "cpu"
    pipeline = DIFFUSERS_TASKS[task](model_dir=model_dir, device=device, **kwargs)
    return pipeline
