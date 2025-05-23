import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

from huggingface_inference_toolkit.const import HF_TRUST_REMOTE_CODE
from huggingface_inference_toolkit.sentence_transformers_utils import SENTENCE_TRANSFORMERS_TASKS
from huggingface_inference_toolkit.utils import (
    check_and_register_custom_pipeline_from_directory,
    get_pipeline,
)


class HuggingFaceHandler:
    """
    A Default Hugging Face Inference Handler which works with all
    Transformers, Diffusers, Sentence Transformers and Optimum pipelines.
    """

    def __init__(
        self, model_dir: Union[str, Path], task: Union[str, None] = None, framework: Literal["pt"] = "pt"
    ) -> None:
        self.pipeline = get_pipeline(
            model_dir=model_dir,  # type: ignore
            task=task,  # type: ignore
            framework=framework,
            trust_remote_code=HF_TRUST_REMOTE_CODE,
        )

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles an inference request with input data and makes a prediction.
        Args:
            :data: (obj): the raw request body data.
        :return: prediction output
        """
        inputs = data.pop("inputs", data)
        parameters = data.pop("parameters", {})

        # diffusers and sentence transformers pipelines do not have the `task` arg
        if not hasattr(self.pipeline, "task"):
            # sentence transformers parameters not supported yet
            if any(isinstance(self.pipeline, v) for v in SENTENCE_TRANSFORMERS_TASKS.values()):
                return (  # type: ignore
                    self.pipeline(**inputs) if isinstance(inputs, dict) else self.pipeline(inputs)
                )
            # diffusers does support kwargs
            return (  # type: ignore
                self.pipeline(**inputs, **parameters)
                if isinstance(inputs, dict)
                else self.pipeline(inputs, **parameters)
            )

        if self.pipeline.task == "question-answering":
            if not isinstance(inputs, dict):
                raise ValueError(f"inputs must be a dict, but a `{type(inputs)}` was provided instead.")
            if not all(k in inputs for k in {"question", "context"}):
                raise ValueError(
                    f"{self.pipeline.task} expects `inputs` to be a dict containing both `question` and "
                    "`context` as the keys, both of them being either a `str` or a `List[str]`."
                )

        if self.pipeline.task == "table-question-answering":
            if not isinstance(inputs, dict):
                raise ValueError(f"inputs must be a dict, but a `{type(inputs)}` was provided instead.")
            if "question" in inputs:
                inputs["query"] = inputs.pop("question")
            if not all(k in inputs for k in {"table", "query"}):
                raise ValueError(
                    f"{self.pipeline.task} expects `inputs` to be a dict containing the keys `table` and "
                    "either `question` or `query`."
                )

        if self.pipeline.task.__contains__("translation") or self.pipeline.task in {
            "text-generation",
            "image-to-text",
            "automatic-speech-recognition",
            "text-to-audio",
            "text-to-speech",
        }:
            # `generate_kwargs` needs to be a dict, `generation_parameters` is here for forward compatibility
            if "generation_parameters" in parameters:
                parameters["generate_kwargs"] = parameters.pop("generation_parameters")

        if self.pipeline.task.__contains__("translation") or self.pipeline.task in {"text-generation"}:
            # flatten the values of `generate_kwargs` as it's not supported as is, but via top-level parameters
            generate_kwargs = parameters.pop("generate_kwargs", {})
            for key, value in generate_kwargs.items():
                parameters[key] = value

        if self.pipeline.task.__contains__("zero-shot-classification"):
            if "candidateLabels" in parameters:
                parameters["candidate_labels"] = parameters.pop("candidateLabels")
            if not isinstance(inputs, dict):
                inputs = {"sequences": inputs}
            if "text" in inputs:
                inputs["sequences"] = inputs.pop("text")
            if not all(k in inputs for k in {"sequences"}) or not all(k in parameters for k in {"candidate_labels"}):
                raise ValueError(
                    f"{self.pipeline.task} expects `inputs` to be either a string or a dict containing the "
                    "key `text` or `sequences`, and `parameters` to be a dict containing either `candidate_labels` "
                    "or `candidateLabels`."
                )

        return (
            self.pipeline(**inputs, **parameters) if isinstance(inputs, dict) else self.pipeline(inputs, **parameters)  # type: ignore
        )


class VertexAIHandler(HuggingFaceHandler):
    """
    A Default Vertex AI Hugging Face Inference Handler which abstracts the
    Vertex AI specific logic for inference.
    """

    def __init__(
        self, model_dir: Union[str, Path], task: Union[str, None] = None, framework: Literal["pt"] = "pt"
    ) -> None:
        super().__init__(model_dir=model_dir, task=task, framework=framework)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles an inference request with input data and makes a prediction.
        Args:
            :data: (obj): the raw request body data.
        :return: prediction output
        """
        if "instances" not in data:
            raise ValueError("The request body must contain a key 'instances' with a list of instances.")
        parameters = data.pop("parameters", {})

        predictions = []
        # iterate over all instances and make predictions
        for inputs in data["instances"]:
            payload = {"inputs": inputs, "parameters": parameters}
            predictions.append(super().__call__(payload))

        # return predictions
        return {"predictions": predictions}


def get_inference_handler_either_custom_or_default_handler(model_dir: Path, task: Optional[str] = None) -> Any:
    """
    Returns the appropriate inference handler based on the given model directory and task.

    Args:
        model_dir (Path): The directory path where the model is stored.
        task (Optional[str]): The task for which the inference handler is required. Defaults to None.

    Returns:
        InferenceHandler: The appropriate inference handler based on the given model directory and task.
    """
    custom_pipeline = check_and_register_custom_pipeline_from_directory(model_dir)
    if custom_pipeline is not None:
        return custom_pipeline

    if os.environ.get("AIP_MODE", None) == "PREDICTION":
        return VertexAIHandler(model_dir=model_dir, task=task)

    return HuggingFaceHandler(model_dir=model_dir, task=task)
