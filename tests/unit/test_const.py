import os


def test_if_provided():
    os.environ["HF_MODEL_DIR"] = "provided"
    os.environ["HF_MODEL_ID"] = "mymodel/test"
    os.environ["HF_TASK"] = "text-classification"
    os.environ["HF_DEFAULT_PIPELINE_NAME"] = "endpoint.py"
    os.environ["HF_FRAMEWORK"] = "tf"
    os.environ["HF_REVISION"] = "12312"
    os.environ["HF_HUB_TOKEN"] = "hf_x"
    from huggingface_inference_toolkit.const import (
        HF_MODEL_DIR,
        HF_MODEL_ID,
        HF_TASK,
        HF_DEFAULT_PIPELINE_NAME,
        HF_MODULE_NAME,
        HF_FRAMEWORK,
        HF_REVISION,
        HF_HUB_TOKEN,
    )

    assert HF_MODEL_DIR == "provided"
    assert HF_MODEL_ID == "mymodel/test"
    assert HF_TASK == "text-classification"
    assert HF_DEFAULT_PIPELINE_NAME == "endpoint.py"
    assert HF_MODULE_NAME == "endpoint.PreTrainedPipeline"
    assert HF_FRAMEWORK == "tf"
    assert HF_REVISION == "12312"
    assert HF_HUB_TOKEN == "hf_x"


def test_default():
    from huggingface_inference_toolkit.const import (
        HF_MODEL_DIR,
        HF_MODEL_ID,
        HF_TASK,
        HF_DEFAULT_PIPELINE_NAME,
        HF_MODULE_NAME,
        HF_FRAMEWORK,
        HF_REVISION,
        HF_HUB_TOKEN,
    )

    assert HF_MODEL_DIR == "/opt/huggingface/model"
    assert HF_MODEL_ID is None
    assert HF_TASK is None
    assert HF_DEFAULT_PIPELINE_NAME == "pipeline.py"
    assert HF_MODULE_NAME == "pipeline.PreTrainedPipeline"
    assert HF_FRAMEWORK == "pytorch"
    assert HF_REVISION is None
    assert HF_HUB_TOKEN is None
