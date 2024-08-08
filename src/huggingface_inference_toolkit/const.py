import os
from pathlib import Path

from huggingface_inference_toolkit.env_utils import strtobool

HF_MODEL_DIR = os.environ.get("HF_MODEL_DIR", "/opt/huggingface/model")
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", None)
HF_TASK = os.environ.get("HF_TASK", None)
HF_FRAMEWORK = os.environ.get("HF_FRAMEWORK", None)
HF_REVISION = os.environ.get("HF_REVISION", None)
HF_HUB_TOKEN = os.environ.get("HF_HUB_TOKEN", None)
HF_TRUST_REMOTE_CODE = strtobool(os.environ.get("HF_TRUST_REMOTE_CODE", "0"))
# custom handler consts
HF_DEFAULT_PIPELINE_NAME = os.environ.get("HF_DEFAULT_PIPELINE_NAME", "handler.py")
# default is pipeline.PreTrainedPipeline
HF_MODULE_NAME = os.environ.get(
    "HF_MODULE_NAME", f"{Path(HF_DEFAULT_PIPELINE_NAME).stem}.EndpointHandler"
)
