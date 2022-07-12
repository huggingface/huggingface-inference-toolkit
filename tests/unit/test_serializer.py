import pytest
import numpy as np
from huggingface_inference_toolkit.serialization.json_utils import Jsoner


def test_json_serializeation():
    t = {"res": np.array([2.0]), "text": "I like you.", "float": 1.2}
    assert b'{"res":[2.0],"text":"I like you.","float":1.2}' == Jsoner.serialize(t)


def test_json_deserialization():
    raw_content = b'{\n\t"inputs": "i like you"\n}'
    assert {"inputs": "i like you"} == Jsoner.deserialize(raw_content)
