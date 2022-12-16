import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union

from huggingface_hub import HfApi, login
from huggingface_hub.file_download import cached_download, hf_hub_url
from huggingface_hub.utils import filter_repo_objects
from transformers import WhisperForConditionalGeneration,AutoConfig ,pipeline
from transformers.file_utils import is_tf_available, is_torch_available
from transformers.pipelines import Conversation, Pipeline

from huggingface_inference_toolkit.const import HF_DEFAULT_PIPELINE_NAME, HF_MODULE_NAME
from huggingface_inference_toolkit.diffusers_utils import (
    check_supported_pipeline,
    get_diffusers_pipeline,
    is_diffusers_available,
)
from huggingface_inference_toolkit.sentence_transformers_utils import (
    get_sentence_transformers_pipeline,
    is_sentence_transformers_available,
)


logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)

_optimum_available = importlib.util.find_spec("optimum") is not None
_accelerate_available = importlib.util.find_spec("accelerate") is not None

if is_tf_available():
    import tensorflow as tf

if is_torch_available():
    import torch


def is_optimum_available():
    return False
    # TODO: change when supported
    # return _optimum_available


def is_accelerate_available():
    return _accelerate_available


framework2weight = {
    "pytorch": "pytorch*",
    "tensorflow": "tf*",
    "tf": "tf*",
    "pt": "pytorch*",
    "flax": "flax*",
    "rust": "rust*",
    "onnx": "*onnx",
}


def create_artifact_filter(framework):
    """
    Returns a list of regex pattern based on the DL Framework. which will be to used to ignore files when downloading
    """
    ignore_regex_list = ["pytorch*", "tf*", "flax*", "rust*", "*onnx"]
    pattern = framework2weight.get(framework, None)
    if pattern in ignore_regex_list:
        ignore_regex_list.remove(pattern)
        return ignore_regex_list
    else:
        return []


def wrap_conversation_pipeline(pipeline):
    """
    Wrap a Conversation with a helper for better UX when using REST API
    """

    def wrapped_pipeline(inputs, *args, **kwargs):
        converted_input = Conversation(
            inputs["text"],
            past_user_inputs=inputs.get("past_user_inputs", []),
            generated_responses=inputs.get("generated_responses", []),
        )
        prediction = pipeline(converted_input, *args, **kwargs)
        return {
            "generated_text": prediction.generated_responses[-1],
            "conversation": {
                "past_user_inputs": prediction.past_user_inputs,
                "generated_responses": prediction.generated_responses,
            },
        }

    return wrapped_pipeline


def _is_gpu_available():
    """
    checks if a gpu is available.
    """
    if is_tf_available():
        return True if len(tf.config.list_physical_devices("GPU")) > 0 else False
    elif is_torch_available():
        return torch.cuda.is_available()
    else:
        raise RuntimeError(
            "At least one of TensorFlow 2.0 or PyTorch should be installed. "
            "To install TensorFlow 2.0, read the instructions at https://www.tensorflow.org/install/ "
            "To install PyTorch, read the instructions at https://pytorch.org/."
        )

def _get_model_config(path_to_model_artifacts: Union[str, Path]):
    """
    extracts the model config from the model artifacts
    """
    return AutoConfig.from_pretrained(path_to_model_artifacts)

def _get_framework():
    """
    extracts which DL framework is used for inference, if both are installed use pytorch
    """
    if is_torch_available():
        return "pytorch"
    elif is_tf_available():
        return "tensorflow"
    else:
        raise RuntimeError(
            "At least one of TensorFlow 2.0 or PyTorch should be installed. "
            "To install TensorFlow 2.0, read the instructions at https://www.tensorflow.org/install/ "
            "To install PyTorch, read the instructions at https://pytorch.org/."
        )


def _load_repository_from_hf(
    repository_id: Optional[str] = None,
    target_dir: Optional[Union[str, Path]] = None,
    framework: Optional[str] = None,
    revision: Optional[str] = None,
    hf_hub_token: Optional[str] = None,
):
    """
    Load a model from huggingface hub.
    """
    if hf_hub_token is not None:
        login(token=hf_hub_token)

    if framework is None:
        framework = _get_framework()

    if isinstance(target_dir, str):
        target_dir = Path(target_dir)

    # create workdir
    if not target_dir.exists():
        target_dir.mkdir(parents=True)

    # create regex to only include the framework specific weights
    ignore_regex = create_artifact_filter(framework)

    # get image artifact files
    _api = HfApi()
    repo_info = _api.repo_info(
        repo_id=repository_id,
        repo_type="model",
        revision=revision,
    )
    # apply regex to filter out non-framework specific weights if args.framework is set
    filtered_repo_files = filter_repo_objects(
        items=[f.rfilename for f in repo_info.siblings],
        ignore_patterns=ignore_regex,
    )

    # iterate over all files and download them
    for repo_file in filtered_repo_files:
        url = hf_hub_url(repo_id=repository_id, filename=repo_file, revision=revision)

        # define values if repo has nested strucutre
        if isinstance(repo_file, str):
            repo_file = Path(repo_file)

        repo_file_is_dir = repo_file.parent != Path(".")
        real_target_dir = target_dir / repo_file.parent if repo_file_is_dir else target_dir
        real_file_name = str(repo_file.name) if repo_file_is_dir else repo_file

        # download files
        path = cached_download(
            url,
            cache_dir=real_target_dir.as_posix(),
            force_filename=real_file_name,
            use_auth_token=hf_hub_token,
        )
        if os.path.exists(path + ".lock"):
            os.remove(path + ".lock")

    # create requirements.txt if not exists
    if not (target_dir / "requirements.txt").exists():
        target_dir.joinpath("requirements.txt").touch()

    return target_dir


def check_and_register_custom_pipeline_from_directory(model_dir):
    """
    Checks if a custom pipeline is available and registers it if so.
    """
    # path to custom handler
    custom_module = Path(model_dir).joinpath(HF_DEFAULT_PIPELINE_NAME)
    legacy_module = Path(model_dir).joinpath("pipeline.py")
    if custom_module.is_file():
        logger.info(f"Found custom pipeline at {custom_module}")
        spec = importlib.util.spec_from_file_location(HF_MODULE_NAME, custom_module)
        if spec:
            # add the whole directory to path for submodlues
            sys.path.insert(0, model_dir)
            # import custom handler
            handler = importlib.util.module_from_spec(spec)
            sys.modules[HF_MODULE_NAME] = handler
            spec.loader.exec_module(handler)
            # init custom handler with model_dir
            custom_pipeline = handler.EndpointHandler(model_dir)
    elif legacy_module.is_file():
        logger.warning(
            "You are using a legacy custom pipeline with. Please update to the new format. See documentation for more information."
        )
        spec = importlib.util.spec_from_file_location("pipeline.PreTrainedPipeline", legacy_module)
        if spec:
            # add the whole directory to path for submodlues
            sys.path.insert(0, model_dir)
            # import custom handler
            pipeline = importlib.util.module_from_spec(spec)
            sys.modules["pipeline.PreTrainedPipeline"] = pipeline
            spec.loader.exec_module(pipeline)
            # init custom handler with model_dir
            custom_pipeline = pipeline.PreTrainedPipeline(model_dir)
    else:
        logger.info(f"No custom pipeline found at {custom_module}")
        custom_pipeline = None
    return custom_pipeline


def get_device():
    """
    The get device function will return the device for the DL Framework.
    """
    if _is_gpu_available():
        return 0
    else:
        return -1


def get_pipeline(task: str, model_dir: Path, **kwargs) -> Pipeline:
    """
    create pipeline class for a specific task based on local saved model
    """
    device = get_device()
    logger.info(f"Using device { 'GPU' if device == 0 else 'CPU'}")

    if task is None:
        raise EnvironmentError(
            "The task for this model is not set: Please set one: https://huggingface.co/docs#how-is-a-models-type-of-inference-api-and-widget-determined"
        )
    # define tokenizer or feature extractor as kwargs to load it the pipeline correctly
    if task in {
        "automatic-speech-recognition",
        "image-segmentation",
        "image-classification",
        "audio-classification",
        "object-detection",
        "zero-shot-image-classification",
    }:
        kwargs["feature_extractor"] = model_dir
    else:
        kwargs["tokenizer"] = model_dir

    # set model kwargs for low cpu usage load and device map to load lager models with accelerate
    if is_accelerate_available() and _get_framework() == "pytorch":
        model_kwargs = kwargs.get("model_kwargs", {})
        kwargs["model_kwargs"] = {
            "low_cpu_mem_usage": True,
            **model_kwargs,
        }
        
    # add check for optimum accelerated pipeline
    if is_optimum_available():
        # TODO: add check for optimum accelerated pipeline
        logger.info("Optimum is not implement yet using default pipeline.")
        hf_pipeline = pipeline(task=task, model=model_dir, device=device, **kwargs)
    elif is_sentence_transformers_available() and task in [
        "sentence-similarity",
        "sentence-embeddings",
        "sentence-ranking",
    ]:
        hf_pipeline = get_sentence_transformers_pipeline(task=task, model_dir=model_dir, device=device, **kwargs)
    elif is_diffusers_available() and check_supported_pipeline(model_dir) and task == "text-to-image":
        hf_pipeline = get_diffusers_pipeline(task=task, model_dir=model_dir, device=device, **kwargs)
    else:
        hf_pipeline = pipeline(task=task, model=model_dir,device=device, **kwargs)

    # wrapp specific pipeline to support better ux
    if task == "conversational":
        hf_pipeline = wrap_conversation_pipeline(hf_pipeline)
    elif task == "automatic-speech-recognition" and isinstance(hf_pipeline.model, WhisperForConditionalGeneration):
        # set chunk length to 30s for whisper to enable long audio files
        hf_pipeline._preprocess_params["chunk_length_s"] = 30
        hf_pipeline._preprocess_params["ignore_warning"] = True
        # set decoder to english by default
        # TODO: replace when transformers 4.26.0 is release with
        # hf_pipeline.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")
        hf_pipeline.tokenizer.language = "english"
        hf_pipeline.tokenizer.task = "transcribe"
        hf_pipeline.model.config.forced_decoder_ids = [
            (rank + 1, token) for rank, token in enumerate(hf_pipeline.tokenizer.prefix_tokens[1:])
        ]
    return hf_pipeline
