import logging
import os

from robyn import Robyn

from huggingface_inference_toolkit.serialization.base import ContentType
from huggingface_inference_toolkit.serialization.json_utils import Jsoner

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)


app = Robyn(__file__)

HF_MODEL_DIR = os.environ.get("HF_MODEL_DIR", "/opt/huggingface/model")
HF_TASK = os.environ.get("HF_TASK", None)

# @app.startup_handler
# async def startup_event():
# global inference_handler

# if empty_directory_or_not_hf_remote_id is None or task is None:
#     raise ValueError(
#         f"Can't initialize model. Please set correct model id and task. provided values are model_id:{model_id_or_path} and task:{task}"
#     )

# logger.info(f"Initializing model with model_id:{model_id_or_path} and task:{task}")
# # create inference handler
# inference_handler = HuggingFaceHandler(HF_MODEL_ID)
# logger.info(f"Model initialized successfully on device: {inference_handler.model.device}")
# return inference_handler


@app.get("/health")
async def health():
    return "OK"


@app.post("/predict")
async def predict(request):
    try:
        logger.info(request)
        content_type = request.headers.get("Content-Type", None)
        body = ContentType.get_deserializer(content_type).deserialize(request["body"])
        logger.info(body)

        # pred = inference_handler(body["inputs"])
        return Jsoner.serialize(body)
    except Exception as e:
        logger.error(e)
        return Jsoner.serialize({"error": str(e)})


app.start(port=5000)
