import logging
import sys

from rich.console import Console
from rich.logging import RichHandler


def setup_logging():
    # # Remove all existing handlers
    # for handler in logging.root.handlers[:]:
    #     logging.root.removeHandler(handler)

    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                tracebacks_show_locals=True,
                console=Console(stderr=False, file=sys.stdout),
            )
        ],
    )
    # Remove `datasets` logger to only log on `critical` mode
    # as it produces `PyTorch` messages to update on `info`
    logging.getLogger("datasets").setLevel(logging.CRITICAL)

    # # Remove Uvicorn loggers
    # logging.getLogger("uvicorn").handlers.clear()
    # logging.getLogger("uvicorn.access").handlers.clear()
    # logging.getLogger("uvicorn.error").handlers.clear()

    logger = logging.getLogger("huggingface_inference_toolkit")
    logger.setLevel(logging.INFO)
    return logger


# Create and configure the logger
logger = setup_logging()
