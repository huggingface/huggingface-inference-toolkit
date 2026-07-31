import asyncio
import base64
import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

import orjson
from anyio import Semaphore
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route

from huggingface_inference_toolkit import idle
from huggingface_inference_toolkit.async_utils import (
    MAX_CONCURRENT_THREADS,
    MAX_THREADS_GUARD,
    async_call,
    offload,
)
from huggingface_inference_toolkit.const import (
    HF_FRAMEWORK,
    HF_HUB_TOKEN,
    HF_MODEL_DIR,
    HF_MODEL_ID,
    HF_REVISION,
    HF_TASK,
)
from huggingface_inference_toolkit.env_utils import task_route_enabled
from huggingface_inference_toolkit.handler import (
    get_inference_handler_either_custom_or_default_handler,
)
from huggingface_inference_toolkit.latency_guard import latency_guard
from huggingface_inference_toolkit.logging import logger
from huggingface_inference_toolkit.serialization.base import ContentType
from huggingface_inference_toolkit.serialization.json_utils import Jsoner
from huggingface_inference_toolkit.utils import (
    convert_params_to_int_or_bool,
    should_discard_left,
)
from huggingface_inference_toolkit.vertex_ai_utils import _load_repository_from_gcs

INFERENCE_HANDLERS = {}
INFERENCE_HANDLERS_SEMAPHORE = Semaphore(1)
MODEL_DOWNLOADED = False
MODEL_DL_SEMAPHORE = Semaphore(1)


def download_model():
    """Fetch the model artifacts if they are not on disk yet. Blocking, so run it in a thread."""
    global MODEL_DOWNLOADED
    # imported late: pulling in the hub client drags transformers with it
    from huggingface_inference_toolkit.heavy_utils import load_repository_from_hf

    # 1. check if model artifacts available in HF_MODEL_DIR
    if next(Path(HF_MODEL_DIR).glob("**/*"), None) is None:
        # 2. if not available, try to load from HF_MODEL_ID
        if HF_MODEL_ID is not None:
            load_repository_from_hf(
                repository_id=HF_MODEL_ID,
                target_dir=HF_MODEL_DIR,
                framework=HF_FRAMEWORK,
                revision=HF_REVISION,
                hf_hub_token=HF_HUB_TOKEN,
            )
        # 3. check if in Vertex AI environment and load from GCS
        # If artifactUri not on Model Creation not set returns an empty string
        elif len(os.environ.get("AIP_STORAGE_URI", "")) > 0:
            _load_repository_from_gcs(
                os.environ["AIP_STORAGE_URI"], target_dir=HF_MODEL_DIR
            )
        # 4. if not available, raise error
        else:
            raise ValueError(
                f"""Can't initialize model.
                Please set env HF_MODEL_DIR or provider a HF_MODEL_ID.
                Provided values are:
                HF_MODEL_DIR: {HF_MODEL_DIR} and HF_MODEL_ID:{HF_MODEL_ID}"""
            )
    else:
        logger.info("Model already present in %s", HF_MODEL_DIR)
    MODEL_DOWNLOADED = True


async def ensure_model_downloaded():
    if not MODEL_DOWNLOADED:
        async with MODEL_DL_SEMAPHORE:
            if not MODEL_DOWNLOADED:
                await async_call(download_model)


def resolve_task(requested_task):
    """
    The task to serve a request with.

    A repository serves one model, but not always through one pipeline: the sentence-transformers
    tasks all answer `feature-extraction` requests with embeddings, which is what a caller asking
    that route for such a repository means.
    """
    if requested_task == "feature-extraction" and HF_TASK in {
        "sentence-similarity",
        "sentence-embeddings",
        "sentence-ranking",
    }:
        return "sentence-embeddings"
    return requested_task


async def ensure_handler_loaded(task):
    """
    Load the pipeline for `task` on first use, once, even if several requests race for it.

    Handlers are kept per task rather than one per worker: the same model directory can be served
    through more than one pipeline, and each is built on demand the first time it is asked for.
    """
    handler = INFERENCE_HANDLERS.get(task)
    if handler is None:
        async with INFERENCE_HANDLERS_SEMAPHORE:
            handler = INFERENCE_HANDLERS.get(task)
            if handler is None:
                logger.info("Initializing %s from directory: %s", task, HF_MODEL_DIR)
                handler = await get_inference_handler_either_custom_or_default_handler(
                    HF_MODEL_DIR, task=task
                )
                INFERENCE_HANDLERS[task] = handler
                logger.info("Model initialized successfully")
    return handler


async def prepare_model_artifacts():
    if idle.UNLOAD_IDLE:
        # Nothing is loaded up front: the worker stays cheap until a request arrives, and the idle
        # checker terminates it once it goes quiet again, releasing the memory for good.
        asyncio.create_task(idle.live_check_loop(), name="live_check_loop")
        logger.info("Idle unloading enabled, deferring model load to the first request")
        return
    await ensure_model_downloaded()
    await ensure_handler_loaded(HF_TASK)


@asynccontextmanager
async def lifespan(app):
    # Starlette 1.0 removed `on_startup` / `on_shutdown` in favor of the ASGI lifespan protocol.
    #
    # There is no shutdown work to do: uvicorn stops accepting, waits for in-flight request
    # handlers, and only then runs this hook — a request queued on the inference semaphore is
    # inside its handler too, so it is waited on as well. What keeps a long inference from being
    # cut off is gunicorn's --graceful-timeout, set in scripts/entrypoint.sh.
    await prepare_model_artifacts()
    yield


async def health(request):
    return PlainTextResponse("Ok")


# Report Prometheus metrics
# inf_batch_current_size: Current number of requests being processed
# inf_queue_size: Number of requests waiting in the queue
# inf_accepting: Whether new requests are being accepted (0 while shedding load)
# inf_auto_frozen: Whether the latency guard considers this worker overloaded
async def metrics(request):
    batch_current_size = MAX_CONCURRENT_THREADS - MAX_THREADS_GUARD.value
    queue_size = MAX_THREADS_GUARD.statistics().tasks_waiting
    return PlainTextResponse(
        f"inf_batch_current_size {batch_current_size}\n" +
        f"inf_queue_size {queue_size}\n"
        f"inf_accepting {int(latency_guard.accepting)}\n"
        f"inf_auto_frozen {int(latency_guard.auto_frozen)}\n"
    )


async def predict(request):
    # Shed load before reading the body: a worker whose latency has drifted far above its baseline
    # is better off refusing quickly than queueing work its callers will time out on. Deliberately
    # outside the idle witness below: a request we refuse is not activity, and must not keep an
    # otherwise idle worker alive.
    if not latency_guard.accepting:
        return Response(
            Jsoner.serialize({"error": "Service temporarily unavailable, overload detected"}),
            status_code=503,
            media_type="application/json",
        )

    # The idle checker must see this request: it counts what is in flight, so it cannot terminate
    # the worker between the moment we start and the moment we answer.
    async with idle.request_witnesses():
        return await _predict(request)


async def _predict(request):
    try:
        # The route may name the task (/pipeline/{task}); otherwise it is the one this repository
        # was deployed for.
        task = resolve_task(request.path_params.get("task", HF_TASK))

        # With idle unloading the model is not loaded at startup, so the first request pays for it.
        # No-ops once loaded.
        await ensure_model_downloaded()
        inference_handler = await ensure_handler_loaded(task)

        # extracts content from request
        content_type = request.headers.get("content-Type", os.environ.get("DEFAULT_CONTENT_TYPE", ""))
        # try to deserialize payload, off the loop: this is where a PNG body reaches PIL
        deserializer = ContentType.get_deserializer(content_type, task)
        deserialized_body = await offload(deserializer.deserialize, await request.body())
        # checks if input schema is correct
        if "inputs" not in deserialized_body and "instances" not in deserialized_body:
            raise ValueError(
                f"Body needs to provide a inputs key, received: {orjson.dumps(deserialized_body)}"
            )

        # Decode base64 audio inputs before running inference
        if "parameters" in deserialized_body and task in {
            "automatic-speech-recognition",
            "audio-classification",
        }:
            # Be more strict on base64 decoding, the provided string should valid base64 encoded data
            deserialized_body["inputs"] = await offload(
                base64.b64decode, deserialized_body["inputs"], validate=True
            )

        # check for query parameter and add them to the body
        if request.query_params and "parameters" not in deserialized_body:
            deserialized_body["parameters"] = convert_params_to_int_or_bool(
                dict(request.query_params)
            )

        # tracks request time
        start_time = perf_counter()
        # run async not blocking call, skipping it if the caller is gone by the time a slot frees
        pred = await async_call(
            inference_handler, deserialized_body, request=request if should_discard_left() else None
        )
        # log request time
        logger.info(
            f"POST {request.url.path} | Duration: {(perf_counter()-start_time) *1000:.2f} ms"
        )

        if pred is None:
            logger.info("No content returned as caller already left")
            return Response(status_code=204)

        # response extracts content from request
        accept = request.headers.get("accept", None)
        if accept is None or accept == "*/*":
            accept = os.environ.get("DEFAULT_ACCEPT", "application/json")
        # An Accept header may list several types and carry parameters: resolve it to the single
        # media type we answer with, and serialize / label the response with that one.
        accept = ContentType.resolve_accept(accept)
        # serialize off the loop too: the JSON path PNG-encodes and base64s any PIL image it
        # finds, so a generated image is encoded here even when the caller asked for JSON.
        serialized_response_body = await offload(
            ContentType.get_serializer(accept).serialize, pred, accept
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
        lifespan=lifespan,
    )
else:
    routes = [
        Route("/", health, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        Route("/", predict, methods=["POST"]),
        Route("/predict", predict, methods=["POST"]),
        Route("/metrics", metrics, methods=["GET"]),
    ]
    if task_route_enabled():
        # Lets a caller ask this repository for a specific pipeline, which is how the Inference API
        # addresses a model that serves more than one task. Opt-in: each task asked for builds its
        # own pipeline, with its own copy of the weights, kept for the life of the worker.
        routes.append(Route("/pipeline/{task:path}", predict, methods=["POST"]))
    app = Starlette(
        debug=False,
        routes=routes,
        lifespan=lifespan,
    )
