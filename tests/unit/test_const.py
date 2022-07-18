import os


def test_if_provided():
    os.environ["HF_MODEL_DIR"] = "provided"
    os.environ["HF_MODEL_ID"] = "mymodel/test"
    os.environ["HF_TASK"] = "text-classification"
    os.environ["HF_DEFAULT_PIPELINE_NAME"] = "endpoint.py"
    from huggingface_inference_toolkit.const import (
        HF_MODEL_DIR,
        HF_MODEL_ID,
        HF_TASK,
        HF_DEFAULT_PIPELINE_NAME,
        HF_MODULE_NAME,
    )

    assert HF_MODEL_DIR == "provided"
    assert HF_MODEL_ID == "mymodel/test"
    assert HF_TASK == "text-classification"
    assert HF_DEFAULT_PIPELINE_NAME == "endpoint.py"
    assert HF_MODULE_NAME == "endpoint.Pipeline"


def test_default():
    from huggingface_inference_toolkit.const import (
        HF_MODEL_DIR,
        HF_MODEL_ID,
        HF_TASK,
        HF_DEFAULT_PIPELINE_NAME,
        HF_MODULE_NAME,
    )

    assert HF_MODEL_DIR == "/opt/huggingface/model"
    assert HF_MODEL_ID is None
    assert HF_TASK is None
    assert HF_DEFAULT_PIPELINE_NAME == "pipeline.py"
    assert HF_MODULE_NAME == "pipeline.Pipeline"
