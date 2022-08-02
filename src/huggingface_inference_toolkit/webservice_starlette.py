import logging
import os
from pathlib import Path
from time import perf_counter

import orjson
from huggingface_inference_toolkit.const import (
    HF_FRAMEWORK,
    HF_HUB_TOKEN,
    HF_MODEL_DIR,
    HF_MODEL_ID,
    HF_REVISION,
    HF_TASK,
)
from huggingface_inference_toolkit.handler import get_inference_handler_either_custom_or_default_handler
from huggingface_inference_toolkit.serialization.base import ContentType
from huggingface_inference_toolkit.serialization.json_utils import Jsoner
from huggingface_inference_toolkit.utils import _load_repository_from_hf
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route


logger = logging.getLogger(__name__)
if os.environ.get("HF_ENDPOINT", None):
    logging.basicConfig(format="| %(levelname)s | %(message)s", level=logging.INFO)
else:
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
uvicorn_error = logging.getLogger("uvicorn.error")
uvicorn_error.disabled = True
uvicorn_access = logging.getLogger("uvicorn.access")
uvicorn_access.disabled = True


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
    logger.info("Model initialized successfully")


async def health(request):
    return PlainTextResponse("Ok")


async def predict(request):
    try:
        # tracks request time
        start_time = perf_counter()

        # extracts content from request
        content_type = request.headers.get("content-Type", None)
        # try to deserialize payload
        deserialized_body = ContentType.get_deserializer(content_type).deserialize(await request.body())
        # checks if input schema is correct
        if "inputs" not in deserialized_body:
            raise ValueError(f"Body needs to provide a inputs key, recieved: {orjson.dumps(deserialized_body)}")

        # runs inference
        pred = inference_handler(deserialized_body)
        # log request time
        # TODO: repalce with middleware
        logger.info(f"POST {request.url.path} |  Duration: {(perf_counter()-start_time) *1000:.2f} ms")
        # deserialized and resonds with json
        return Response(Jsoner.serialize(pred))
    except Exception as e:
        logger.error(e)
        return Response(Jsoner.serialize({"error": str(e)}), status_code=400)


app = Starlette(
    debug=True,
    routes=[
        Route("/", health, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        Route("/", predict, methods=["POST"]),
        Route("/predict", predict, methods=["POST"]),
    ],
    on_startup=[some_startup_task],
)
