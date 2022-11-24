import base64
import json
import numpy as np
import pytest
import os
from huggingface_inference_toolkit.serialization import Jsoner, Audioer, Imager
from PIL import Image


def test_json_serialization():
    t = {"res": np.array([2.0]), "text": "I like you.", "float": 1.2}
    assert b'{"res":[2.0],"text":"I like you.","float":1.2}' == Jsoner.serialize(t)


def test_json_image_serialization():
    t = [
        {"label": "refrigerator", "mask": Image.new("RGB", (60, 30), color="red"), "score": 0.9803440570831299},
        {"label": "LABEL_200", "mask": Image.new("RGB", (60, 30), color="red"), "score": 0.9631735682487488},
        {"label": "cat", "mask": Image.new("RGB", (60, 30), color="red"), "score": 0.9995332956314087},
    ]
    Jsoner.serialize(t)


def test_image_serialization():
    image = Image.new("RGB", (60, 30), color="red")
    Imager.serialize(image, accept="image/png")


def test_json_deserialization():
    raw_content = b'{\n\t"inputs": "i like you"\n}'
    assert {"inputs": "i like you"} == Jsoner.deserialize(raw_content)


def test_image_deserialization():
    image_files_path = os.path.join(os.getcwd(), "tests/resources/image")

    for image_file in os.listdir(image_files_path):
        image_bytes = open(os.path.join(image_files_path, image_file), "rb").read()
        decoded_data = Imager.deserialize(bytearray(image_bytes))

        assert isinstance(decoded_data, dict)
        assert isinstance(decoded_data["inputs"], Image.Image)


def test_audio_deserialization():
    audio_files_path = os.path.join(os.getcwd(), "tests/resources/audio")

    for audio_file in os.listdir(audio_files_path):
        audio_bytes = open(os.path.join(audio_files_path, audio_file), "rb").read()
        decoded_data = Audioer.deserialize(bytearray(audio_bytes))

        assert {"inputs": audio_bytes} == decoded_data
