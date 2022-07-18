import os
from pathlib import Path


HF_MODEL_DIR = os.environ.get("HF_MODEL_DIR", "/opt/huggingface/model")
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", None)
HF_TASK = os.environ.get("HF_TASK", None)
HF_FRAMEWORK = os.environ.get("HF_FRAMEWORK", "pytorch")
HF_REVISION = os.environ.get("HF_REVISION", None)
HF_HUB_TOKEN = os.environ.get("HF_HUB_TOKEN", None)
# custom handler consts
HF_DEFAULT_PIPELINE_NAME = os.environ.get("HF_DEFAULT_PIPELINE_NAME", "pipeline.py")
# default is pipeline.PreTrainedPipeline
HF_MODULE_NAME = os.environ.get("HF_MODULE_NAME", f"{Path(HF_DEFAULT_PIPELINE_NAME).stem}.PreTrainedPipeline")