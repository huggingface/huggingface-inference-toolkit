from typing import Optional

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


def _normalize(media_type: str) -> str:
    """Strip any parameter (charset, codecs, ...) and fold case: `image/PNG; foo=bar` -> `image/png`."""
    return media_type.split(";")[0].strip().lower()


# Media types are matched case-insensitively and without their parameters, hence a lookup keyed on
# the normalized form of every supported type. `content_type_mapping` stays the documented list
# and is what error messages report.
_normalized_mapping = {_normalize(media_type): serializer for media_type, serializer in content_type_mapping.items()}


class ContentType:
    @staticmethod
    def get_deserializer(content_type: str, task: Optional[str]):
        if not content_type:
            message = "No content type provided and no default one configured."
            raise Exception(message)

        media_type = _normalize(content_type)

        if media_type == "application/octet-stream":
            # Raw bytes announce nothing about their own type, so the task has to decide how to
            # read them.
            if task:
                if "audio" in task or "speech" in task:
                    return Audioer
                if "image" in task:
                    return Imager
            message = f"""
                Content type "{content_type}" not supported for task {task}.
                Supported content types are:
                {", ".join(list(content_type_mapping.keys()))}
            """
            raise Exception(message)

        if media_type in _normalized_mapping:
            return _normalized_mapping[media_type]
        else:
            message = f"""
                Content type "{content_type}" not supported.
                Supported content types are:
                {", ".join(list(content_type_mapping.keys()))}
            """
            raise Exception(message)

    @staticmethod
    def resolve_accept(accept: str) -> str:
        """
        Resolve an Accept header to the media type we will answer with: the first entry we can
        actually produce, normalized. Callers must serialize and label the response with this
        rather than with the raw header, which may list several types and carry parameters.
        """
        for candidate in accept.split(","):
            media_type = _normalize(candidate)
            if media_type in _normalized_mapping:
                return media_type
        message = f"""
            Accept type "{accept}" not supported.
            Supported accept types are:
            {", ".join(list(content_type_mapping.keys()))}
        """
        raise Exception(message)

    @staticmethod
    def get_serializer(accept: str):
        return _normalized_mapping[ContentType.resolve_accept(accept)]
