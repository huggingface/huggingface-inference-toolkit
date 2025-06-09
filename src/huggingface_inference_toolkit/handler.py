import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

from huggingface_inference_toolkit import logging
from huggingface_inference_toolkit.const import HF_TRUST_REMOTE_CODE
from huggingface_inference_toolkit.env_utils import api_inference_compat
from huggingface_inference_toolkit.utils import check_and_register_custom_pipeline_from_directory


class HuggingFaceHandler:
    """
    A Default Hugging Face Inference Handler which works with all
    Transformers, Diffusers, Sentence Transformers and Optimum pipelines.
    """

    def __init__(
        self, model_dir: Union[str, Path], task: Union[str, None] = None, framework: Literal["pt"] = "pt"
    ) -> None:
        from huggingface_inference_toolkit.heavy_utils import get_pipeline
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

        # import as late as possible to reduce the footprint
        from huggingface_inference_toolkit.sentence_transformers_utils import SENTENCE_TRANSFORMERS_TASKS

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
                inputs = [inputs]
                parameters.setdefault("top_k", os.environ.get("DEFAULT_TOP_K", 5))
            if self.pipeline.task == "token-classification":
                parameters.setdefault("aggregation_strategy", os.environ.get("DEFAULT_AGGREGATION_STRATEGY", "simple"))

        resp = self.pipeline(**inputs, **parameters) if isinstance(inputs, dict) else \
            self.pipeline(inputs, **parameters)

        if api_inference_compat():
            if self.pipeline.task == "text-classification":
                # We don't want to return {} but [{}] in any case
                if isinstance(resp, list) and len(resp) > 0:
                    if not isinstance(resp[0], list):
                        return [resp]
                return resp
            if self.pipeline.task == "feature-extraction":
                # If the library used is Transformers then the feature-extraction is returning the headless encoder
                # outputs as embeddings. The shape is a 3D or 4D array
                # [n_inputs, batch_size = 1, n_sentence_tokens, num_hidden_dim].
                # Let's just discard the batch size dim that always seems to be 1 and return a 2D/3D array
                # https://github.com/huggingface/transformers/blob/5c47d08b0d6835b8d8fc1c06d9a1bc71f6e78ace/src/transformers/pipelines/feature_extraction.py#L27
                # for api inference (reason: mainly display)
                new_resp = []
                if isinstance(inputs, list):
                    if isinstance(resp, list) and len(resp) == len(inputs):
                        for it in resp:
                            # Batch size dim is the first it level, dicard it
                            if isinstance(it, list) and len(it) == 1:
                                new_resp.append(it[0])
                            else:
                                logging.logger.warning("One of the output batch size differs from 1: %d", len(it))
                                return resp
                        return new_resp
                    else:
                        logging.logger.warning("Inputs and resp len differ (or resp is not a list, type %s)",
                                               type(resp))
                        return resp
                elif isinstance(inputs, str):
                    if isinstance(resp, list) and len(resp) == 1:
                        return resp[0]
                    else:
                        logging.logger.warning("The output batch size differs from 1: %d", len(resp))
                        return resp
                else:
                    logging.logger.warning("Output unexpected type %s", type(resp))
                    return resp
            if self.pipeline.task == "image-segmentation":
                if isinstance(resp, list):
                    new_resp = []
                    for el in resp:
                        if isinstance(el, dict) and el.get("score") is None:
                            el["score"] = 1
                        new_resp.append(el)
                    resp = new_resp
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
    custom_pipeline = check_and_register_custom_pipeline_from_directory(model_dir)
    if custom_pipeline is not None:
        return custom_pipeline

    if os.environ.get("AIP_MODE", None) == "PREDICTION":
        return VertexAIHandler(model_dir=model_dir, task=task)

    return HuggingFaceHandler(model_dir=model_dir, task=task)
