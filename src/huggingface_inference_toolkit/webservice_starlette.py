from cmath import log
import os
import logging
from pathlib import Path
from huggingface_inference_toolkit.const import (
    HF_FRAMEWORK,
    HF_HUB_TOKEN,
    HF_MODEL_DIR,
    HF_MODEL_ID,
    HF_REVISION,
    HF_TASK,
)
from huggingface_inference_toolkit.serialization.json_utils import Jsoner

from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route
from starlette.requests import Request

from huggingface_inference_toolkit.handler import (
    HuggingFaceHandler,
    get_inference_handler_either_custom_or_default_handler,
)
from huggingface_inference_toolkit.serialization.base import ContentType
from huggingface_inference_toolkit.utils import _load_repository_from_hf

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(name)s | %(levelname)s | %(message)s", level=logging.INFO)

# @app.startup_handler
async def some_startup_task():
    global inference_handler
    # 1. check if model artifacts available in HF_MODEL_DIR
    if len(list(Path(HF_MODEL_DIR).glob("**/*"))) <= 0:
        if HF_MODEL_ID is not None:
            _load_repository_from_hf(
                repository_id=HF_MODEL_ID,
                target_dir=HF_MODEL_DIR,
                framework=HF_FRAMEWORK,
                revision=HF_REVISION,
                hf_hub_token=HF_HUB_TOKEN,
            )
        else:
            raise ValueError(
                f"Can't initialize model. Please set env HF_MODEL_DIR or provider a HF_MODEL_ID. Provided values are HF_MODEL_DIR:{HF_MODEL_DIR} and HF_MODEL_ID:{HF_MODEL_ID}"
            )

    logger.info(f"Initializing model from directory:{HF_MODEL_DIR}")
    # 2. determine correct inference handler
    inference_handler = get_inference_handler_either_custom_or_default_handler(HF_MODEL_DIR, task=HF_TASK)
    logger.info(f"Model initialized successfully")


async def health(request):
    return PlainTextResponse("Ok")


async def predict(request):
    try:
        content_type = request.headers.get("content-Type", None)
        logger.info(await request.body())
        deserialized_body = ContentType.get_deserializer(content_type).deserialize(await request.body())
        if "inputs" not in deserialized_body:
            raise ValueError("Body needs to provide a inputs key")

        pred = inference_handler(deserialized_body)
        return Response(Jsoner.serialize(pred))
    except Exception as e:
        logger.error(e)
        return Response(Jsoner.serialize({"error": str(e)}))


app = Starlette(
    debug=True,
    routes=[
        Route("/", health, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        Route("/predict", predict, methods=["POST"]),
    ],
    on_startup=[some_startup_task],
)
