import logging
import os
from pathlib import Path
from typing import Optional, Union

from huggingface_inference_toolkit.utils import check_and_register_custom_pipeline_from_directory, get_pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)


class HuggingFaceHandler:
    """
    A Default Hugging Face Inference Handler which works with all
    transformers pipelines, Sentence Transformers and Optimum.
    """

    def __init__(self, model_dir: Union[str, Path], task=None, framework="pt"):
        self.pipeline = get_pipeline(
            model_dir=model_dir,
            task=task,
            framework=framework
        )

    def __call__(self, data):
        """
        Handles an inference request with input data and makes a prediction.
        Args:
            :data: (obj): the raw request body data.
        :return: prediction output
        """
        inputs = data.pop("inputs", data)
        parameters = data.pop("parameters", None)

        # pass inputs with all kwargs in data
        if parameters is not None:
            prediction = self.pipeline(inputs, **parameters)
        else:
            prediction = self.pipeline(inputs)
        # postprocess the prediction
        return prediction


class VertexAIHandler(HuggingFaceHandler):
    """
    A Default Vertex AI Hugging Face Inference Handler which abstracts the
    Vertex AI specific logic for inference.
    """
    def __init__(self, model_dir: Union[str, Path], task=None, framework="pt"):
        super().__init__(model_dir, task, framework)
    
    def __call__(self, data):
        """
        Handles an inference request with input data and makes a prediction.
        Args:
            :data: (obj): the raw request body data.
        :return: prediction output
        """
        if "instances" not in data:
            raise ValueError("The request body must contain a key 'instances' with a list of instances.")
        parameters = data.pop("parameters", None)
        
        predictions = []
        # iterate over all instances and make predictions
        for inputs in data["instances"]:
            payload = {"inputs": inputs, "parameters": parameters}
            predictions.append(super().__call__(payload))
        
        # reutrn predictions
        return {"predictions": predictions}

def get_inference_handler_either_custom_or_default_handler(
    model_dir: Path,
    task: Optional[str] = None
):
    """
    Returns the appropriate inference handler based on the given model directory and task.
    
    Args:
        model_dir (Path): The directory path where the model is stored.
        task (Optional[str]): The task for which the inference handler is required. Defaults to None.
    
    Returns:
        InferenceHandler: The appropriate inference handler based on the given model directory and task.
    """
    custom_pipeline = check_and_register_custom_pipeline_from_directory(model_dir)
    if custom_pipeline:
        return custom_pipeline
    elif os.environ.get("AIP_MODE", None) == "PREDICTION": 
        return VertexAIHandler(model_dir=model_dir, task=task)
    else:
        return HuggingFaceHandler(model_dir=model_dir, task=task)
