import base64
import binascii
from typing import Optional

from huggingface_inference_toolkit.serialization.audio_utils import Audioer
from huggingface_inference_toolkit.serialization.image_utils import Imager
from huggingface_inference_toolkit.serialization.json_utils import Jsoner

# Tasks whose top-level `inputs` is a single media item. For these, a JSON string is the media
# content itself, base64-encoded — never a path or URL for the server to resolve. See
# `decode_media_string_input`.
AUDIO_INPUT_TASKS = frozenset(
    {
        "automatic-speech-recognition",
        "audio-classification",
        "zero-shot-audio-classification",
    }
)
IMAGE_INPUT_TASKS = frozenset(
    {
        "image-classification",
        "image-segmentation",
        "image-to-text",
        "image-to-image",
        "object-detection",
        "depth-estimation",
        "image-feature-extraction",
        "mask-generation",
        "zero-shot-image-classification",
        "zero-shot-object-detection",
        # First positional argument is the image, so a bare string is resolved here too. The
        # dict form these three usually arrive in is handled by `MEDIA_INPUT_KEYS` below.
        "visual-question-answering",
        "document-question-answering",
        "image-text-to-text",
    }
)

# Tasks whose `inputs` is a dict carrying the media under a key instead of being the media. The
# handler splats such a dict into the pipeline as keyword arguments, so the nested string never
# passes through the single-value path above and has to be decoded here as well.
MEDIA_INPUT_KEYS = {
    "visual-question-answering": ("image",),
    "document-question-answering": ("image",),
    # Only the plain `images` / `image` argument. A conversational payload nests images inside a
    # messages list; hf-inference does not serve that shape, so it is deliberately not walked.
    "image-text-to-text": ("images", "image"),
}

# Media tasks the toolkit registers no media type for: nothing in `content_type_mapping` can carry
# a video, so there is no encoding a caller could legitimately send instead. transformers would
# fetch an `http(s)` string and open anything else as a local file, so a string is refused.
UNSUPPORTED_MEDIA_TASKS = frozenset({"video-classification"})

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


def _decode_base64_media(task: Optional[str], deserializer, value: str, where: str):
    """Decode one base64 media string, or reject it. `where` names the field for the error."""
    contract = (
        f"{where} for task '{task}' must be the media content itself: either raw bytes sent with "
        "a matching audio/* or image/* Content-Type, or a base64-encoded string in a JSON body."
    )
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"{contract} The provided string could not be decoded as base64 ({e}).") from e
    try:
        return deserializer.deserialize(raw)["inputs"]
    except Exception as e:
        # Decoding succeeding does not mean the caller sent media: a path is often valid base64 in
        # its own right, `/var/lib/nonexistent` being one. Restate the contract rather than
        # surfacing a deserializer complaint about a buffer the caller never knowingly sent.
        raise ValueError(f"{contract} The decoded bytes are not valid media ({e}).") from e


def decode_media_string_input(task: Optional[str], value):
    """
    Resolve a JSON `inputs` for a media task into the decoded media the pipeline expects.

    For audio and image tasks a string input is the media content itself, base64-encoded, and is
    decoded here into the bytes / PIL image the binary-body path already produces. This is
    deliberately the only way a string reaches such a pipeline: passed through untouched, the
    pipeline would resolve it as a local path or a remote reference rather than as content. A
    string that is not valid base64 is rejected rather than passed through.

    Three shapes are handled, because transformers resolves a string in all three:

    - `inputs` is the media (`IMAGE_INPUT_TASKS` / `AUDIO_INPUT_TASKS`), decoded in place;
    - `inputs` is a dict carrying the media under a key (`MEDIA_INPUT_KEYS`) -- the handler splats
      it into the pipeline as keyword arguments, so each media key is decoded and the rest of the
      dict is left alone;
    - the task has no media type the toolkit can accept (`UNSUPPORTED_MEDIA_TASKS`), so a string
      cannot be anything but a path or a URL and is refused outright.

    Anything else is returned unchanged: a binary body is already decoded, and a text task's
    string is literal text.
    """
    if isinstance(value, str) and task in UNSUPPORTED_MEDIA_TASKS:
        raise ValueError(
            f"'inputs' for task '{task}' cannot be a string: transformers would fetch it as a URL "
            "or open it as a local file, and the toolkit registers no media type this task's "
            "content could be sent as instead."
        )

    if isinstance(value, dict):
        keys = MEDIA_INPUT_KEYS.get(task or "")
        if not keys:
            return value
        decoded = dict(value)
        for key in keys:
            if isinstance(decoded.get(key), str):
                decoded[key] = _decode_base64_media(task, Imager, decoded[key], f"'inputs[\"{key}\"]'")
        return decoded

    if not isinstance(value, str):
        return value
    if task in AUDIO_INPUT_TASKS:
        deserializer = Audioer
    elif task in IMAGE_INPUT_TASKS:
        deserializer = Imager
    else:
        return value
    return _decode_base64_media(task, deserializer, value, "'inputs'")
