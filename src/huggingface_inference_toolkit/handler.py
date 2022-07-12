import logging
from abc import ABC
from pathlib import Path
from typing import Union

from huggingface_inference_toolkit.utils import _is_gpu_available, get_pipeline


logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(name)s | %(levelname)s | %(message)s", level=logging.INFO)


class HuggingFaceHandler(ABC):
    """
    A Default Hugging Face Inference Handler which works with all transformers pipelines, Sentence Transformers and Optimum.
    """

    def __init__(self, model_dir: Union[str, Path]):
        self.pipeline = get_pipeline(model_dir=model_dir)

    def handle(self, data):
        """
        Handles an inference request with input data and makes a prediction.
        Args:
            :data: (obj): the raw request body data.
        :return: prediction output
        """
        # pass inputs with all kwargs in data
        if data.parameters is not None:
            prediction = self.pipeline(data.inputs, **data.parameters)
        else:
            prediction = self.pipeline(data.inputs)
        # postprocess the prediction
        return prediction
