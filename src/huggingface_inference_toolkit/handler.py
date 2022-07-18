import logging
from abc import ABC
from pathlib import Path
from typing import Optional, Union

from huggingface_inference_toolkit.utils import check_and_register_custom_pipeline_from_directory, get_pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(name)s | %(levelname)s | %(message)s", level=logging.INFO)


class HuggingFaceHandler(ABC):
    """
    A Default Hugging Face Inference Handler which works with all transformers pipelines, Sentence Transformers and Optimum.
    """

    def __init__(self, model_dir: Union[str, Path], task=None):
        self.pipeline = get_pipeline(model_dir=model_dir, task=task)

    def __call__(self, data):
        """
        Handles an inference request with input data and makes a prediction.
        Args:
            :data: (obj): the raw request body data.
        :return: prediction output
        """
        logger.info(data)
        inputs = data.pop("inputs", data)
        parameters = data.pop("parameters", None)

        # pass inputs with all kwargs in data
        if parameters is not None:
            prediction = self.pipeline(inputs, **parameters)
        else:
            prediction = self.pipeline(inputs)
        # postprocess the prediction
        return prediction



def get_inference_handler_either_custom_or_default_handler(
    model_dir: Path, task: Optional[str] = None
):
    """
    get inference handler either custom or default Handler
    """
    custom_pipeline = check_and_register_custom_pipeline_from_directory(model_dir)
    if custom_pipeline:
        return custom_pipeline
    else:
        return HuggingFaceHandler(model_dir=model_dir, task=task)
