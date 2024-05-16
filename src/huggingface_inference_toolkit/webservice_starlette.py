import logging
import os
from pathlib import Path
from time import perf_counter

import orjson
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route

from huggingface_inference_toolkit.async_utils import async_handler_call
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
from huggingface_inference_toolkit.utils import _load_repository_from_hf, convert_params_to_int_or_bool


def config_logging(level=logging.INFO):
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", datefmt="", level=level, force=True)
    # disable uvicorn access logs to hide /health
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.disabled = True
    # remove double logs for errors
    logging.getLogger("uvicorn").removeHandler(logging.getLogger("uvicorn").handlers[0])


config_logging(os.environ.get("LOG_LEVEL", logging.getLevelName(logging.INFO)))
logger = logging.getLogger(__name__)


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
                f"""Can't initialize model.
                Please set env HF_MODEL_DIR or provide a HF_MODEL_ID.
                Provided values are:
                HF_MODEL_DIR: {HF_MODEL_DIR} and HF_MODEL_ID:{HF_MODEL_ID}"""
            )

    logger.info(f"Initializing model from directory:{HF_MODEL_DIR}")
    # 2. determine correct inference handler
    inference_handler = get_inference_handler_either_custom_or_default_handler(HF_MODEL_DIR, task=HF_TASK)
    logger.info("Model initialized successfully")


async def health(request):
    return PlainTextResponse("Ok")


async def predict(request):
    try:
        # extracts content from request
        content_type = request.headers.get("content-Type", None)
        # try to deserialize payload
        deserialized_body = ContentType.get_deserializer(content_type).deserialize(await request.body())
        # checks if input schema is correct
        if "inputs" not in deserialized_body:
            raise ValueError(f"Body needs to provide a inputs key, recieved: {orjson.dumps(deserialized_body)}")

        # check for query parameter and add them to the body
        if request.query_params and "parameters" not in deserialized_body:
            deserialized_body["parameters"] = convert_params_to_int_or_bool(dict(request.query_params))

        # tracks request time
        start_time = perf_counter()
        # run async not blocking call
        pred = await async_handler_call(inference_handler, deserialized_body)
        # log request time
        logger.info(f"POST {request.url.path} | Duration: {(perf_counter()-start_time) *1000:.2f} ms")

        # response extracts content from request
        accept = request.headers.get("accept", None)
        if accept is None or accept == "*/*":
            accept = "application/json"
        # deserialized and resonds with json
        serialized_response_body = ContentType.get_serializer(accept).serialize(pred, accept)
        return Response(serialized_response_body, media_type=accept)
    except Exception as e:
        logger.error(e)
        return Response(Jsoner.serialize({"error": str(e)}), status_code=400, media_type="application/json")


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
