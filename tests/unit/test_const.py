import os
from unittest import mock


# @mock.patch.dict(
#     os.environ,
#     {
#         "HF_MODEL_DIR": "provided",
#         "HF_MODEL_ID": "mymodel/test",
#         "HF_TASK": "text-classification",
#         "HF_DEFAULT_PIPELINE_NAME": "endpoint.py",
#         "HF_FRAMEWORK": "tf",
#         "HF_REVISION": "12312",
#         "HF_HUB_TOKEN": "hf_x",
#     },
#     clear=True,
# )
def test_if_provided():
    with mock.patch.dict(
        os.environ,
        {
            "HF_MODEL_DIR": "provided",
            "HF_MODEL_ID": "mymodel/test",
            "HF_TASK": "text-classification",
            "HF_DEFAULT_PIPELINE_NAME": "endpoint.py",
            "HF_FRAMEWORK": "tf",
            "HF_REVISION": "12312",
            "HF_HUB_TOKEN": "hf_x",
        },
        clear=True,
    ):

        from huggingface_inference_toolkit.const import (
            HF_DEFAULT_PIPELINE_NAME,
            HF_FRAMEWORK,
            HF_HUB_TOKEN,
            HF_MODEL_DIR,
            HF_MODEL_ID,
            HF_MODULE_NAME,
            HF_REVISION,
            HF_TASK,
        )

        assert HF_MODEL_DIR == "provided"
        assert HF_MODEL_ID == "mymodel/test"
        assert HF_TASK == "text-classification"
        assert HF_DEFAULT_PIPELINE_NAME == "endpoint.py"
        assert HF_MODULE_NAME == "endpoint.PreTrainedPipeline"
        assert HF_FRAMEWORK == "tf"
        assert HF_REVISION == "12312"
        assert HF_HUB_TOKEN == "hf_x"


# def test_default():
#     os.environ = {}
#     from huggingface_inference_toolkit.const import (
#         HF_DEFAULT_PIPELINE_NAME,
#         HF_FRAMEWORK,
#         HF_HUB_TOKEN,
#         HF_MODEL_DIR,
#         HF_MODEL_ID,
#         HF_MODULE_NAME,
#         HF_REVISION,
#         HF_TASK,
#     )

#     assert os.environ == {}
#     assert HF_MODEL_DIR == "/opt/huggingface/model"
#     assert HF_MODEL_ID is None
#     assert HF_TASK is None
#     assert HF_DEFAULT_PIPELINE_NAME == "pipeline.py"
#     assert HF_MODULE_NAME == "pipeline.PreTrainedPipeline"
#     assert HF_FRAMEWORK == "pytorch"
#     assert HF_REVISION is None
#     assert HF_HUB_TOKEN is None
