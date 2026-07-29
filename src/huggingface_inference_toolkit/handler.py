import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

from huggingface_inference_toolkit.const import HF_TRUST_REMOTE_CODE
from huggingface_inference_toolkit.env_utils import api_inference_compat, ignore_custom_handler
from huggingface_inference_toolkit.logging import logger
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

        if api_inference_compat():
            if self.pipeline.task == "text-classification" and isinstance(inputs, str):
                # A single string is sent as a batch of one, with the whole ranking asked for
                # rather than the top label alone. Left as is for a list of inputs: defaulting
                # top_k there would change how many labels each one comes back with.
                inputs = [inputs]
                parameters.setdefault("top_k", int(os.environ.get("DEFAULT_TOP_K", 5)))
            if self.pipeline.task == "token-classification":
                parameters.setdefault(
                    "aggregation_strategy", os.environ.get("DEFAULT_AGGREGATION_STRATEGY", "simple")
                )

        resp = (
            self.pipeline(**inputs, **parameters) if isinstance(inputs, dict) else self.pipeline(inputs, **parameters)  # type: ignore
        )

        if api_inference_compat():
            resp = self._reshape_for_api_inference_compat(inputs, resp)

        return resp

    def _reshape_for_api_inference_compat(self, inputs: Any, resp: Any) -> Any:
        """
        Adapt a pipeline's output to the shape the Inference API / Hub widgets expect.

        Every branch degrades to the untouched response and a warning when the output does not
        look the way it should: a display concern must never turn a successful inference into an
        error.
        """
        if self.pipeline.task == "text-classification":
            # Always one ranking per input. A flat response means one entry per input, which is the
            # same thing as a single ranking only when there was exactly one input — so the number
            # of inputs in the request decides, not the shape of the response. Reading the shape
            # alone would turn N top-1 results into what looks like one N-label ranking, losing
            # which result belongs to which input.
            if not isinstance(resp, list) or (resp and isinstance(resp[0], list)):
                return resp  # already grouped per input
            if isinstance(inputs, str):
                n_inputs = 1
            elif isinstance(inputs, list):
                n_inputs = len(inputs)
            else:
                logger.warning("Cannot group text-classification scores, unexpected inputs type %s", type(inputs))
                return resp
            if n_inputs == 1:
                return [resp]
            if len(resp) != n_inputs:
                logger.warning("Inputs and resp len differ, %d != %d", n_inputs, len(resp))
                return resp
            # One entry per input: group each on its own so the shape stays list[list[dict]]
            # whatever the input count, without changing how many labels came back.
            return [[entry] for entry in resp]

        if self.pipeline.task == "feature-extraction":
            # Transformers returns the headless encoder outputs, shaped
            # [n_inputs, batch_size = 1, n_tokens, n_hidden]. The batch dim is always 1 here, so
            # drop it and hand back a 2D/3D array.
            # https://github.com/huggingface/transformers/blob/5c47d08b/src/transformers/pipelines/feature_extraction.py#L27
            if isinstance(inputs, list):
                if not isinstance(resp, list) or len(resp) != len(inputs):
                    logger.warning(
                        "Inputs and resp len differ (or resp is not a list, type %s)", type(resp)
                    )
                    return resp
                squeezed = []
                for embeddings in resp:
                    if not isinstance(embeddings, list) or len(embeddings) != 1:
                        logger.warning("One of the output batch size differs from 1: %d", len(embeddings))
                        return resp
                    squeezed.append(embeddings[0])
                return squeezed
            if isinstance(inputs, str):
                if isinstance(resp, list) and len(resp) == 1:
                    return resp[0]
                logger.warning("The output batch size differs from 1: %d", len(resp))
                return resp
            logger.warning("Output unexpected type %s", type(resp))
            return resp

        if self.pipeline.task == "image-segmentation":
            # Semantic segmentation reports no per-mask score, but the widget needs one
            if isinstance(resp, list):
                for element in resp:
                    if isinstance(element, dict) and element.get("score") is None:
                        element["score"] = 1
            return resp

        if self.pipeline.task == "zero-shot-classification":
            # Parallel labels/scores arrays -> the [{label, score}] the widget renders
            if not isinstance(resp, dict) or "labels" not in resp or "scores" not in resp:
                logger.warning("Unable to remap response for api inference compat, unexpected shape")
                return resp
            labels, scores = resp["labels"], resp["scores"]
            if len(labels) != len(scores):
                logger.warning(
                    "Unable to remap response for api inference compat, "
                    "labels and scores do not have the same len, %d != %d",
                    len(labels),
                    len(scores),
                )
                return resp
            return [
                {"label": label, "score": score}
                for label, score in zip(labels, scores, strict=True)  # lengths checked above
            ]

        return resp


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
    custom_pipeline = (
        None if ignore_custom_handler() else check_and_register_custom_pipeline_from_directory(model_dir)
    )
    if custom_pipeline is not None:
        return custom_pipeline

    if os.environ.get("AIP_MODE", None) == "PREDICTION":
        return VertexAIHandler(model_dir=model_dir, task=task)

    return HuggingFaceHandler(model_dir=model_dir, task=task)
