from huggingface_inference_toolkit.serialization.audio_utils import Audioer
from huggingface_inference_toolkit.serialization.image_utils import Imager
from huggingface_inference_toolkit.serialization.json_utils import Jsoner

content_type_mapping = {
    "application/json": Jsoner,
    # image types
    "image/png": Imager,
    "image/jpeg": Imager,
    "image/jpg": Imager,
    "image/tiff": Imager,
    "image/bmp": Imager,
    "image/gif": Imager,
    "image/webp": Imager,
    "image/x-image": Imager,
    # audio types
    "audio/x-flac": Audioer,
    "audio/flac": Audioer,
    "audio/mpeg": Audioer,
    "audio/x-mpeg-3": Audioer,
    "audio/wave": Audioer,
    "audio/wav": Audioer,
    "audio/x-wav": Audioer,
    "audio/ogg": Audioer,
    "audio/x-audio": Audioer,
    "audio/webm": Audioer,
    "audio/webm;codecs=opus": Audioer,
    "audio/AMR": Audioer,
    "audio/amr": Audioer,
    "audio/AMR-WB": Audioer,
    "audio/AMR-WB+": Audioer,
    "audio/m4a": Audioer,
    "audio/x-m4a": Audioer,
}


class ContentType:
    @staticmethod
    def get_deserializer(content_type: str, task: str):
        if content_type.lower().startswith("application/octet-stream"):
            if "audio" in task or "speech" in task:
                return Audioer
            elif "image" in task:
                return Imager
            message = f"""
                Content type "{content_type}" not supported for task {task}.
                Supported content types are:
                {", ".join(list(content_type_mapping.keys()))}
            """
            raise Exception(message)

        # Extract media type from content type
        extracted = content_type.split(";")[0]
        if extracted in content_type_mapping:
            return content_type_mapping[extracted]
        else:
            message = f"""
                Content type "{content_type}" not supported.
                Supported content types are:
                {", ".join(list(content_type_mapping.keys()))}
            """
            raise Exception(message)

    @staticmethod
    def get_serializer(accept: str):
        extracts = accept.split(",")
        for extract in extracts:
            extracted = extract.split(";")[0]
            if extracted in content_type_mapping:
                return content_type_mapping[extracted]
        message = f"""
            Accept type "{accept}" not supported.
            Supported accept types are:
            {", ".join(list(content_type_mapping.keys()))}
        """
        raise Exception(message)
