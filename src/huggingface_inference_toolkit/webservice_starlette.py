from cmath import log
import os
import logging
from huggingface_inference_toolkit.serialization.json_utils import Jsoner

from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route
from starlette.requests import Request

from huggingface_inference_toolkit.handler import HuggingFaceHandler
from huggingface_inference_toolkit.serialization.base import ContentType

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(name)s | %(levelname)s | %(message)s", level=logging.INFO)


# HF_MODEL_DIR = os.environ.get("HF_MODEL_DIR", "/opt/huggingface/model")
# HF_TASK = os.environ.get("HF_TASK", None)


HF_MODEL_DIR = os.environ.get("HF_MODEL_DIR", "distilbert-base-uncased-finetuned-sst-2-english")
HF_TASK = os.environ.get("HF_TASK", "text-classification")

# @app.startup_handler
async def some_startup_task():
    global inference_handler

    # if empty_directory_or_not_hf_remote_id is None or task is None:
    #     raise ValueError(
    #         f"Can't initialize model. Please set correct model id and task. provided values are model_id:{model_id_or_path} and task:{task}"
    #     )

    logger.info(f"Initializing model from direcotry:{HF_MODEL_DIR} with task:{HF_TASK}")
    # create inference handler
    inference_handler = HuggingFaceHandler(HF_MODEL_DIR, task=HF_TASK)
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
