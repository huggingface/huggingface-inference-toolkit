import os
from pathlib import Path
from time import perf_counter

import orjson
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route

from huggingface_inference_toolkit.async_utils import async_handler_call
from huggingface_inference_toolkit.logging import logger
from huggingface_inference_toolkit.serialization.base import ContentType
from huggingface_inference_toolkit.serialization.json_utils import Jsoner
from huggingface_inference_toolkit.utils import (
    convert_params_to_int_or_bool,
)

import sys
sys.path.append('speech-to-speech')
from s2s_handler import EndpointHandler

async def prepare_handler():
    global inference_handler
    inference_handler = EndpointHandler()
    logger.info("Model initialized successfully")

async def health(request):
    return PlainTextResponse("Ok")


async def predict(request):
    try:
        # extracts content from request
        content_type = request.headers.get("content-Type", None)
        # try to deserialize payload
        deserialized_body = ContentType.get_deserializer(content_type).deserialize(
            await request.body()
        )
        # checks if input schema is correct
        if "inputs" not in deserialized_body and "instances" not in deserialized_body:
            raise ValueError(
                f"Body needs to provide a inputs key, received: {orjson.dumps(deserialized_body)}"
            )

        # check for query parameter and add them to the body
        if request.query_params and "parameters" not in deserialized_body:
            deserialized_body["parameters"] = convert_params_to_int_or_bool(
                dict(request.query_params)
            )

        # tracks request time
        start_time = perf_counter()
        # run async not blocking call
        pred = await async_handler_call(inference_handler, deserialized_body)
        # log request time
        logger.info(
            f"POST {request.url.path} | Duration: {(perf_counter()-start_time) *1000:.2f} ms"
        )

        # response extracts content from request
        accept = request.headers.get("accept", None)
        if accept is None or accept == "*/*":
            accept = "application/json"
        # deserialized and resonds with json
        serialized_response_body = ContentType.get_serializer(accept).serialize(
            pred, accept
        )
        return Response(serialized_response_body, media_type=accept)
    except Exception as e:
        logger.error(e)
        return Response(
            Jsoner.serialize({"error": str(e)}),
            status_code=400,
            media_type="application/json",
        )


# Create app based on which cloud environment is used
if os.getenv("AIP_MODE", None) == "PREDICTION":
    logger.info("Running in Vertex AI environment")
    # extract routes from environment variables
    _predict_route = os.getenv("AIP_PREDICT_ROUTE", None)
    _health_route = os.getenv("AIP_HEALTH_ROUTE", None)
    if _predict_route is None or _health_route is None:
        raise ValueError(
            "AIP_PREDICT_ROUTE and AIP_HEALTH_ROUTE need to be set in Vertex AI environment"
        )

    app = Starlette(
        debug=False,
        routes=[
            Route(_health_route, health, methods=["GET"]),
            Route(_predict_route, predict, methods=["POST"]),
        ],
        on_startup=[prepare_handler],
    )
else:
    app = Starlette(
        debug=False,
        routes=[
            Route("/", health, methods=["GET"]),
            Route("/health", health, methods=["GET"]),
            Route("/", predict, methods=["POST"]),
            Route("/predict", predict, methods=["POST"]),
        ],
        on_startup=[prepare_handler],
    )
