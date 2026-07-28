import pytest

from huggingface_inference_toolkit.serialization.audio_utils import Audioer
from huggingface_inference_toolkit.serialization.base import ContentType
from huggingface_inference_toolkit.serialization.image_utils import Imager
from huggingface_inference_toolkit.serialization.json_utils import Jsoner


@pytest.mark.parametrize(
    "content_type,expected",
    [
        ("application/json", Jsoner),
        # parameters are not part of the media type
        ("application/json; charset=UTF-8", Jsoner),
        ("application/json;charset=utf-8", Jsoner),
        ("image/png", Imager),
        ("audio/wav", Audioer),
        # both the mapping and the header may carry case
        ("APPLICATION/JSON", Jsoner),
        ("audio/AMR-WB", Audioer),
        ("audio/amr-wb", Audioer),
        ("audio/webm;codecs=opus", Audioer),
    ],
)
def test_get_deserializer_media_types(content_type, expected):
    assert ContentType.get_deserializer(content_type, "text-classification") is expected


@pytest.mark.parametrize(
    "task,expected",
    [
        ("automatic-speech-recognition", Audioer),
        ("audio-classification", Audioer),
        ("image-classification", Imager),
        ("image-segmentation", Imager),
    ],
)
def test_get_deserializer_octet_stream_resolves_by_task(task, expected):
    # Raw bytes say nothing about their own type, so the task decides
    assert ContentType.get_deserializer("application/octet-stream", task) is expected
    assert ContentType.get_deserializer("Application/Octet-Stream", task) is expected


@pytest.mark.parametrize("task", ["text-classification", None, ""])
def test_get_deserializer_octet_stream_rejected_for_other_tasks(task):
    with pytest.raises(Exception, match="not supported for task"):
        ContentType.get_deserializer("application/octet-stream", task)


def test_get_deserializer_requires_a_content_type():
    with pytest.raises(Exception, match="No content type provided"):
        ContentType.get_deserializer("", "text-classification")
    with pytest.raises(Exception, match="No content type provided"):
        ContentType.get_deserializer(None, "text-classification")


@pytest.mark.parametrize("content_type", ["text/plain", "text/csv", "application/xml"])
def test_get_deserializer_rejects_unsupported_types(content_type):
    with pytest.raises(Exception, match="not supported"):
        ContentType.get_deserializer(content_type, "text-classification")


@pytest.mark.parametrize(
    "accept,expected",
    [
        ("application/json", Jsoner),
        ("image/png", Imager),
        ("application/json; charset=UTF-8", Jsoner),
        # first supported type in the list wins
        ("text/html, application/json", Jsoner),
        ("text/html;q=0.9, image/png;q=0.8", Imager),
    ],
)
def test_get_serializer_accept_lists(accept, expected):
    assert ContentType.get_serializer(accept) is expected


def test_get_serializer_rejects_when_nothing_is_producible():
    with pytest.raises(Exception, match="not supported"):
        ContentType.get_serializer("text/html, application/xml")


@pytest.mark.parametrize(
    "accept,expected",
    [
        ("application/json", "application/json"),
        ("image/PNG", "image/png"),
        ("image/png;q=0.8", "image/png"),
        ("text/html;q=0.9, image/png;q=0.8", "image/png"),
    ],
)
def test_resolve_accept_returns_a_bare_media_type(accept, expected):
    # Imager.serialize derives the image format from this value and it also labels the response,
    # so it must never carry the parameters or the other entries of the header
    assert ContentType.resolve_accept(accept) == expected


def test_resolve_accept_rejects_when_nothing_is_producible():
    with pytest.raises(Exception, match="not supported"):
        ContentType.resolve_accept("text/html")
