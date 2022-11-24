import importlib.util
import json
import os

_diffusers = importlib.util.find_spec("diffusers") is not None


def is_diffusers_available():
    return _diffusers


if is_diffusers_available():
    from diffusers import StableDiffusionPipeline


def check_supported_pipeline(model_dir):
    try: 
      with open(os.path.join(model_dir), "model_index.json") as json_file:
          data = json.load(json_file)
          if data["_class_name"] == "StableDiffusionPipeline":
              return True
          else: 
              return False
    except: 
      return False      


class DiffusersPipelineImageToText:
    def __init__(self, model_dir: str, device: str = None):  # needs "cuda" for GPU
        # try:
        #   pipeline = DIFFUSERS_TASKS[task].from_pretrained(model_dir, torch_dtype=torch.float16,)
        # except:
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_dir)
        self.pipeline.to(device)

    def __call__(self, prompt, **kwargs):

        if kwargs:
            out = self.pipeline(prompt, **kwargs)
        else:
            out = self.pipeline(prompt)

        # TODO: only return 1 image currently
        return out.images[0]


DIFFUSERS_TASKS = {
    "text-to-image": DiffusersPipelineImageToText,
}


def get_diffusers_pipeline(task=None, model_dir=None, device=-1, **kwargs):
    device = "cuda" if device == 0 else "cpu"
    pipeline = DIFFUSERS_TASKS[task](model_dir=model_dir, device=device)
    return pipeline
