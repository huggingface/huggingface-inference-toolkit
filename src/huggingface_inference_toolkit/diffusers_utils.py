import importlib.util
import logging
import os

from transformers.utils.import_utils import is_torch_bf16_gpu_available
from optimum import neuron

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)

_diffusers = importlib.util.find_spec("diffusers") is not None


def is_diffusers_available():
    return _diffusers


if is_diffusers_available():
    import torch
    from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler, StableDiffusionPipeline


class IEAutoPipelineForText2Image:
    def __init__(self, model_dir: str, device: str = None):  # needs "cuda" for GPU
        dtype = torch.float32
        if device == "cuda":
            dtype = torch.bfloat16 if is_torch_bf16_gpu_available() else torch.float16
        device_map = "auto" if device == "cuda" else None

        self.pipeline = AutoPipelineForText2Image.from_pretrained(model_dir, torch_dtype=dtype, device_map=device_map)
        # try to use DPMSolverMultistepScheduler
        if isinstance(self.pipeline, StableDiffusionPipeline):
            try:
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
            except Exception:
                pass

        self.pipeline.to(device)

    def __call__(
        self,
        prompt,
        **kwargs,
    ):
        # TODO: add support for more images (Reason is correct output)
        if "num_images_per_prompt" in kwargs:
            kwargs.pop("num_images_per_prompt")
            logger.warning("Sending num_images_per_prompt > 1 to pipeline is not supported. Using default value 1.")

        # Call pipeline with parameters
        out = self.pipeline(prompt, num_images_per_prompt=1, **kwargs)
        return out.images[0]

#
# DIFFUSERS_TASKS = {
#     "text-to-image": [NeuronStableDiffusionXLPipeline],
# }


def load_optimum_diffusion_pipeline(task, model_dir):

    # Step 1: load config and look for _class_name
    try:
        config = StableDiffusionPipeline.load_config(pretrained_model_name_or_path=model_dir)
    except OSError as e:
        logger.error("Unable to load config file for repository %s", model_dir)
        logger.exception(e)
        raise

    pipeline_class_name = config['_class_name']

    logger.debug("Repository pipeline class name %s", pipeline_class_name)
    if pipeline_class_name.contains("Diffusion") and pipeline_class_name.contains("XL"):
        if task == "image-to-image":
            pipeline_class = neuron.NeuronStableDiffusionXLImg2ImgPipeline
        else:
            pipeline_class = neuron.NeuronStableDiffusionXLPipeline
    else:
        if task == "image-to-image":
            pipeline_class = neuron.NeuronStableDiffusionImg2ImgPipeline
        else:
            pipeline_class = neuron.NeuronStableDiffusionPipeline

    logger.debug("Pipeline class %s", pipeline_class.__class__)

    # if is neuron model, no need for additional kwargs
    if pipeline_class_name.contains("Neuron"):
        kwargs = {}
    else:
        # Model will be compiled and exported on the flight as the cached models cause a performance drop
        # for diffusion models, unless otherwise specified through an explicit env variable

        # Image shapes need to be frozen at loading/compilation time
        compiler_args = {
            "auto_cast": "matmul",
            "auto_cast_type": "bf16",
            "inline_weights_to_neff": os.environ.get("INLINE_WEIGHTS_TO_NEFF",
                                                     "false").lower() in ["false", "no", "0"],
            "data_parallel_mode": os.environ.get("DATA_PARALLEL_MODE", "unet")
        }
        input_shapes = {"batch_size": 1,
                        "height": int(os.environ("IMAGE_HEIGHT", 512)),
                        "width": int(os.environ("IMAGE_WIDTH", 512))}
        kwargs = {**compiler_args, **input_shapes, "export": True}

    # In the second case, exporting can take a huge amount of time, which makes endpoints not a really suited solution
    # at least as long as the cache is not really an option for diffusion
    return pipeline_class(kwargs)


def get_diffusers_pipeline(task=None, model_dir=None, **kwargs):
    """Get a pipeline for Diffusers models."""
    pipeline = load_optimum_diffusion_pipeline(task=task, model_dir=model_dir)
    return pipeline
