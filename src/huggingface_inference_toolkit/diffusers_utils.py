import importlib.util
import json
import os

_diffusers = importlib.util.find_spec("diffusers") is not None


def is_diffusers_available():
    return _diffusers


if is_diffusers_available():
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline


def check_supported_pipeline(model_dir):
    try:
        with open(os.path.join(model_dir, "model_index.json")) as json_file:
            data = json.load(json_file)
            if data["_class_name"] == "StableDiffusionPipeline":
                return True
            else:
                return False
    except Exception:
        return False


class DiffusersPipelineImageToText:
    def __init__(self, model_dir: str, device: str = None):  # needs "cuda" for GPU
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch.float16)
        # try to use DPMSolverMultistepScheduler
        try:
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        except Exception:
            pass
        self.pipeline.to(device)

    def __call__(
        self,
        prompt,
        num_inference_steps=25,
        guidance_scale=7.5,
        num_images_per_prompt=1,
        height=None,
        width=None,
        negative_prompt=None,
    ):
        # TODO: add support for more images (Reason is correct output)
        num_images_per_prompt = 1

        # Call pipeline with parameters
        out = self.pipeline(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
        )

        return out.images[0]


DIFFUSERS_TASKS = {
    "text-to-image": DiffusersPipelineImageToText,
}


def get_diffusers_pipeline(task=None, model_dir=None, device=-1, **kwargs):
    """Get a pipeline for Diffusers models."""
    device = "cuda" if device == 0 else "cpu"
    pipeline = DIFFUSERS_TASKS[task](model_dir=model_dir, device=device)
    return pipeline
