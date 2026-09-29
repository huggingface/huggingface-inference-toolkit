import base64
from io import BytesIO

import pytest
from PIL import Image

from huggingface_inference_toolkit.serialization.audio_utils import Audioer
from huggingface_inference_toolkit.serialization.base import (
    AUDIO_INPUT_TASKS,
    IMAGE_INPUT_TASKS,
    ContentType,
    content_type_mapping,
    decode_media_string_input,
)
from huggingface_inference_toolkit.serialization.image_utils import MEDIA_TYPE_TO_PIL_FORMAT, Imager
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


@pytest.mark.parametrize(
    "accept,expected_format",
    [
        ("image/png", "PNG"),
        # PIL registers "JPEG", so deriving the format from the subtype raised a KeyError here
        ("image/jpg", "JPEG"),
        ("image/jpeg", "JPEG"),
        ("image/tiff", "TIFF"),
        ("image/bmp", "BMP"),
        ("image/gif", "GIF"),
        ("image/webp", "WEBP"),
        # Names no format at all: answer with the lossless one
        ("image/x-image", "PNG"),
    ],
)
def test_imager_serializes_every_supported_accept(accept, expected_format):
    body = Imager.serialize(Image.new("RGB", (8, 8), "red"), accept)
    assert Image.open(BytesIO(body)).format == expected_format


@pytest.mark.parametrize("accept", ["image/JPG", "image/jpeg;q=0.8", None])
def test_imager_tolerates_unresolved_accept_values(accept):
    # `resolve_accept` normally normalizes this, but a direct caller should not get a wrong format
    assert Image.open(BytesIO(Imager.serialize(Image.new("RGB", (8, 8)), accept))).format in {
        "JPEG",
        "PNG",
    }


@pytest.mark.parametrize("mode", ["RGBA", "P", "L"])
def test_imager_flattens_modes_jpeg_cannot_hold(mode):
    # Masks arrive as "P" and generated images can carry alpha; JPEG accepts neither
    body = Imager.serialize(Image.new(mode, (8, 8)), "image/jpeg")
    assert Image.open(BytesIO(body)).format == "JPEG"


def test_imager_rejects_types_it_cannot_produce():
    with pytest.raises(ValueError, match="Cannot serialize an image"):
        Imager.serialize(Image.new("RGB", (8, 8)), "application/json")


def test_imager_rejects_non_images():
    with pytest.raises(ValueError, match="Can only serialize"):
        Imager.serialize({"not": "an image"}, "image/png")


def test_every_advertised_image_type_can_be_serialized():
    # Guard against drift: a new image/* row in `content_type_mapping` that has no PIL format
    # would only fail at response time, on the accept path
    advertised = {media_type for media_type in content_type_mapping if media_type.startswith("image/")}
    assert advertised == set(MEDIA_TYPE_TO_PIL_FORMAT)


def test_decode_media_string_input_decodes_base64_audio():
    audio_bytes = b"\x00\x01RIFFfake-audio"
    decoded = decode_media_string_input(
        "automatic-speech-recognition", base64.b64encode(audio_bytes).decode()
    )
    # Decoded to the raw bytes the pipeline reads, never opened as a path
    assert decoded == audio_bytes


def test_decode_media_string_input_decodes_base64_image():
    buffer = BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="PNG")
    decoded = decode_media_string_input(
        "image-classification", base64.b64encode(buffer.getvalue()).decode()
    )
    assert isinstance(decoded, Image.Image)
    assert decoded.size == (8, 8)


@pytest.mark.parametrize("task", ["text-classification", "text-generation", "feature-extraction"])
def test_decode_media_string_input_leaves_text_tasks_untouched(task):
    # A string is literal text for these tasks, not encoded media
    assert decode_media_string_input(task, "/etc/passwd") == "/etc/passwd"


def test_decode_media_string_input_leaves_non_strings_untouched():
    # A binary body is already decoded to bytes / a PIL image before it reaches here
    image = Image.new("RGB", (8, 8))
    assert decode_media_string_input("image-classification", image) is image
    assert decode_media_string_input("automatic-speech-recognition", b"raw-bytes") == b"raw-bytes"


@pytest.mark.parametrize(
    "task",
    ["automatic-speech-recognition", "audio-classification", "image-classification", "image-to-text"],
)
@pytest.mark.parametrize(
    "malicious",
    [
        "/tmp/plop",  # local file path
        "http://127.0.0.1:8080/internal",  # a request the server would make on the caller's behalf
        "https://example.com/sample.wav",
    ],
)
def test_decode_media_string_input_rejects_paths_and_urls(task, malicious):
    # A media task must never receive a raw string: transformers would open it as a file or fetch
    # it as a URL. None of these is valid base64, so each is rejected instead of resolved.
    with pytest.raises(ValueError, match="must be the media content itself"):
        decode_media_string_input(task, malicious)


@pytest.mark.parametrize("task", ["image-classification", "image-to-text"])
def test_decode_media_string_input_rejects_a_path_that_is_itself_valid_base64(task):
    # A path is not always rejected by the base64 decode: `/var/lib/nonexistent` is 20 characters
    # of the base64 alphabet and decodes cleanly. What matters is that it is decoded rather than
    # opened -- the bytes are the caller's own string, never the file -- and for an image the
    # result is then caught as not-media, so the caller still gets the contract back.
    #
    # Not asserted for audio: `Audioer` hands the bytes to the pipeline as-is, since any byte
    # string is a candidate audio payload and there is nothing to validate it against. Such a
    # path decodes to noise and fails in the audio decoder instead, still never opened.
    with pytest.raises(ValueError, match="must be the media content itself"):
        decode_media_string_input(task, "/var/lib/nonexistent")


def _b64_png():
    buffer = BytesIO()
    Image.new("RGB", (8, 8), "blue").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.mark.parametrize(
    "task,key",
    [
        ("visual-question-answering", "image"),
        ("document-question-answering", "image"),
        ("image-text-to-text", "images"),
        ("image-text-to-text", "image"),
    ],
)
def test_decode_media_string_input_decodes_the_media_key_of_a_dict(task, key):
    # These arrive as a dict, which the handler splats into the pipeline as keyword arguments, so
    # the media is nested under a key rather than being `inputs` itself.
    decoded = decode_media_string_input(task, {key: _b64_png(), "question": "what is this?"})
    assert isinstance(decoded[key], Image.Image)
    assert decoded[key].size == (8, 8)
    # Everything that is not media is passed through as it was
    assert decoded["question"] == "what is this?"


@pytest.mark.parametrize(
    "task,key",
    [
        ("visual-question-answering", "image"),
        ("document-question-answering", "image"),
        ("image-text-to-text", "images"),
    ],
)
@pytest.mark.parametrize("malicious", ["/var/lib/nonexistent", "http://127.0.0.1:8080/internal"])
def test_decode_media_string_input_rejects_paths_and_urls_nested_in_a_dict(task, key, malicious):
    # `load_image` resolves a nested string exactly as it resolves a top-level one
    with pytest.raises(ValueError, match="must be the media content itself"):
        decode_media_string_input(task, {key: malicious, "question": "what is this?"})


def test_decode_media_string_input_leaves_non_media_dicts_untouched():
    # question-answering takes a dict too, but none of its values is media
    inputs = {"question": "who?", "context": "/etc/passwd is a path, read as literal text here"}
    assert decode_media_string_input("question-answering", inputs) == inputs


def test_decode_media_string_input_leaves_absent_and_non_string_media_keys_untouched():
    image = Image.new("RGB", (8, 8))
    assert decode_media_string_input("visual-question-answering", {"question": "who?"}) == {
        "question": "who?"
    }
    # A binary body is already a PIL image by the time it gets here
    assert decode_media_string_input("visual-question-answering", {"image": image})["image"] is image


@pytest.mark.parametrize("value", ["/var/lib/nonexistent", "http://127.0.0.1:8080/internal", "Zm9v"])
def test_decode_media_string_input_refuses_a_string_for_video_classification(value):
    # No media type in `content_type_mapping` can carry a video, so there is no encoding a caller
    # could legitimately send: a string can only be something for the server to open or fetch.
    with pytest.raises(ValueError, match="cannot be a string"):
        decode_media_string_input("video-classification", value)


# --- Every shape transformers resolves a string in, not only the bare-string and known-dict ones


_MALICIOUS = ["/var/lib/nonexistent", "http://127.0.0.1:8080/internal"]


@pytest.mark.parametrize("task", sorted(IMAGE_INPUT_TASKS))
@pytest.mark.parametrize("key", ["inputs", "images", "image"])
def test_every_image_task_decodes_the_media_key_of_a_dict(task, key):
    # The handler splats a dict into the pipeline as keyword arguments for every task, not just
    # the three that usually arrive that way: `inputs` is the base pipeline's keyword and
    # `images` / `image` the per-task aliases, so all of them are media for all image tasks.
    decoded = decode_media_string_input(task, {key: _b64_png()})
    assert isinstance(decoded[key], Image.Image)


@pytest.mark.parametrize(
    "task", ["image-classification", "object-detection", "zero-shot-image-classification", "image-to-text"]
)
@pytest.mark.parametrize("key", ["inputs", "images", "image"])
@pytest.mark.parametrize("malicious", _MALICIOUS)
def test_a_dict_for_an_unlisted_image_task_is_not_a_way_around_the_guard(task, key, malicious):
    with pytest.raises(ValueError, match="must be the media content itself"):
        decode_media_string_input(task, {key: malicious})


@pytest.mark.parametrize("task", sorted(AUDIO_INPUT_TASKS))
@pytest.mark.parametrize("key", ["inputs", "audios"])
def test_a_dict_for_an_audio_task_is_not_a_way_around_the_guard(task, key):
    with pytest.raises(ValueError, match="must be the media content itself"):
        decode_media_string_input(task, {key: "http://127.0.0.1:8080/internal"})
    # A path that happens to be valid base64 is decoded to its own bytes, never opened (see
    # `test_decode_media_string_input_rejects_a_path_that_is_itself_valid_base64` for why audio
    # does not reject it outright)
    decoded = decode_media_string_input(task, {key: "/var/lib/nonexistent"})
    assert decoded[key] == base64.b64decode("/var/lib/nonexistent")


@pytest.mark.parametrize("task", ["image-classification", "visual-question-answering"])
@pytest.mark.parametrize("malicious", _MALICIOUS)
def test_a_list_of_strings_is_not_a_way_around_the_guard(task, malicious):
    # The base pipeline iterates a list and preprocesses each element as it would a single input
    with pytest.raises(ValueError, match=r"'inputs\[1\]' for task"):
        decode_media_string_input(task, [_b64_png(), malicious])
    with pytest.raises(ValueError, match=r"'inputs\[1\]' for task"):
        decode_media_string_input("audio-classification", ["Zm9v", "http://127.0.0.1:8080/internal"])


def test_a_list_of_strings_is_decoded_element_by_element():
    decoded = decode_media_string_input("image-classification", [_b64_png(), _b64_png()])
    assert len(decoded) == 2
    assert all(isinstance(image, Image.Image) for image in decoded)


@pytest.mark.parametrize("task", ["visual-question-answering", "zero-shot-object-detection", "image-text-to-text"])
@pytest.mark.parametrize("malicious", _MALICIOUS)
def test_a_list_of_dicts_is_not_a_way_around_the_guard(task, malicious):
    with pytest.raises(ValueError, match=r"'inputs\[0\]\[\"image\"\]' for task"):
        decode_media_string_input(task, [{"image": malicious, "question": "what?"}])


@pytest.mark.parametrize("malicious", _MALICIOUS)
def test_a_list_nested_under_a_media_key_is_not_a_way_around_the_guard(malicious):
    # image-text-to-text takes a list of lists of images
    with pytest.raises(ValueError, match=r"'inputs\[\"images\"\]\[0\]\[1\]' for task"):
        decode_media_string_input("image-text-to-text", {"images": [[_b64_png(), malicious]], "text": ["hi"]})


def _chat(content_item):
    return {"text": [{"role": "user", "content": [content_item, {"type": "text", "text": "describe"}]}]}


@pytest.mark.parametrize(
    "content_item,where",
    [
        ({"type": "image", "url": "{m}"}, r'\["url"\]'),
        ({"type": "image", "path": "{m}"}, r'\["path"\]'),
        ({"type": "image", "image": "{m}"}, r'\["image"\]'),
        # `base64` is documented as carrying encoded content, but transformers resolves it through
        # the same loader as the others, so a URL there is fetched all the same
        ({"type": "image", "base64": "{m}"}, r'\["base64"\]'),
        ({"type": "image_url", "image_url": {"url": "{m}"}}, r'\["image_url"\]\["url"\]'),
    ],
)
@pytest.mark.parametrize("malicious", _MALICIOUS)
def test_an_image_inside_a_chat_message_is_not_a_way_around_the_guard(content_item, where, malicious):
    # The conversational shape nests the image several levels down, under `text`, which is not
    # itself media: the walk has to go through text positions to reach it.
    item = {k: (v.format(m=malicious) if isinstance(v, str) else {"url": malicious}) for k, v in content_item.items()}
    with pytest.raises(ValueError, match=r"'inputs\[\"text\"\]\[0\]\[\"content\"\]\[0\]" + where + "' for task"):
        decode_media_string_input("image-text-to-text", _chat(item))


def test_an_image_inside_a_chat_message_is_decoded_in_place():
    decoded = decode_media_string_input("image-text-to-text", _chat({"type": "image", "url": _b64_png()}))
    content = decoded["text"][0]["content"]
    assert isinstance(content[0]["url"], Image.Image)
    # The text item next to it is untouched
    assert content[1] == {"type": "text", "text": "describe"}


def test_text_values_of_a_media_task_dict_are_left_alone():
    # A question, a prompt or candidate labels are literal text, even when they look like a path.
    # Only media positions are decoded; the walk passes through the rest.
    inputs = {
        "image": _b64_png(),
        "question": "/etc/passwd",
        "prompt": "http://127.0.0.1:8080/internal",
        "candidate_labels": ["cat", "/etc/passwd"],
        "text": [{"role": "user", "content": "http://127.0.0.1:8080/internal"}],
    }
    decoded = decode_media_string_input("image-text-to-text", inputs)
    assert isinstance(decoded["image"], Image.Image)
    for key in ("question", "prompt", "candidate_labels", "text"):
        assert decoded[key] == inputs[key]


@pytest.mark.parametrize("value", [{"inputs": "/var/lib/nonexistent"}, {"videos": "Zm9v"}, ["http://127.0.0.1/x"]])
def test_video_classification_refuses_a_media_string_in_any_shape(value):
    with pytest.raises(ValueError, match="cannot be a string"):
        decode_media_string_input("video-classification", value)


def test_the_path_names_where_the_offending_string_was_found():
    # `instances` is the Vertex AI body: the caller is told which instance and key was rejected
    with pytest.raises(ValueError, match=r"'instances\[1\]\[\"images\"\]' for task 'image-classification'"):
        decode_media_string_input(
            "image-classification", [{"images": _b64_png()}, {"images": "/var/lib/nonexistent"}], path="instances"
        )


@pytest.mark.parametrize(
    "task,key",
    [
        ("image-classification", "images"),
        ("zero-shot-image-classification", "images"),
        ("image-text-to-text", "images"),
        ("video-classification", "videos"),
    ],
)
def test_parameters_are_walked_for_media_keys_but_are_not_media_themselves(task, key):
    # The handler splats `parameters` into the pipeline call as keyword arguments, next to the
    # input, and for these tasks the media keyword overrides the input. A string there is
    # resolved like one in `inputs`. The other parameters are what they are: `top_k` a number,
    # `prompt` a text.
    with pytest.raises(ValueError, match=r"'parameters\[\"" + key + r"\"\]' for task"):
        decode_media_string_input(task, {key: "http://127.0.0.1:8080/internal"}, path="parameters", is_media=False)
    parameters = {"top_k": 3, "prompt": "http://127.0.0.1:8080/internal", "candidate_labels": ["/etc/passwd"]}
    assert decode_media_string_input(task, parameters, path="parameters", is_media=False) == parameters


def test_a_chat_in_the_text_parameter_is_walked():
    # image-text-to-text takes the conversation as `text`, a parameter as much as an input
    with pytest.raises(ValueError, match=r"'parameters\[\"text\"\]\[0\]\[\"content\"\]\[0\]\[\"url\"\]'"):
        decode_media_string_input(
            "image-text-to-text",
            _chat({"type": "image", "url": "http://127.0.0.1:8080/internal"}),
            path="parameters",
            is_media=False,
        )
