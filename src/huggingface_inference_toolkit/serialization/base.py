from huggingface_inference_toolkit.serialization.audio_utils import Audioer
from huggingface_inference_toolkit.serialization.image_utils import Imager
from huggingface_inference_toolkit.serialization.json_utils import Jsoner


content_type_mapping = {
    "application/json": Jsoner,
    "text/csv": None,
    "text/plain": None,
    # image types
    "image/png": Imager,
    "image/jpeg": Imager,
    "image/tiff": Imager,
    "image/bmp": Imager,
    "image/gif": Imager,
    "image/webp": Imager,
    "image/x-image": Imager,
    # audio types
    "audio/x-flac": Audioer,
    "audio/mpeg": Audioer,
    "audio/wave": Audioer,
    "audio/ogg": Audioer,
    "audio/x-audio": Audioer,
}


class ContentType:
    @staticmethod
    def get_deserializer(content_type):
        if content_type in content_type_mapping:
            return content_type_mapping[content_type].get
        else:
            raise Exception(
                f'Content type "{content_type}" not supported. Supported content types are: {list(content_type_mapping.keys()).split(", ")}'
            )
