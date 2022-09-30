import logging
from pathlib import Path
from time import perf_counter

import orjson
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
from huggingface_inference_toolkit.utils import _load_repository_from_hf
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route


def config_logging(level=logging.INFO):
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", datefmt="", level=level)
    # disable uvicorn access logs to hide /health
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.disabled = True
    # remove double logs for errors
    logging.getLogger("uvicorn").removeHandler(logging.getLogger("uvicorn").handlers[0])


config_logging()
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

        # run async not blocking call
        pred = await async_handler_call(inference_handler, deserialized_body)
        # run sync blocking call -> slighty faster for < 200ms prediction time
        # pred = inference_handler(deserialized_body)

        # log request time
        # TODO: repalce with middleware
        logger.info(f"POST {request.url.path} | Duration: {(perf_counter()-start_time) *1000:.2f} ms")
        # deserialized and resonds with json
        return Response(Jsoner.serialize(pred), media_type="application/json")
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


# for pegasus it was async
# 1.2rps at 20 with 17s latency
# 1rps at 1 user with 930ms latency

# for pegasus it was sync
# 1.2rps at 20 with 17s latency
# 1rps at 1 user with 980ms latency
# health is blocking with 17s latency


# for tiny it was async
# 107.7rps at 500 with 4.7s latency
# 8.5rps at 1 user with 120ms latency

# for tiny it was sync
# 109rps at 500 with 4.6s latency
# 8.5rps at 1 user with 120ms latency
