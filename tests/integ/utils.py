import logging
import re
import signal
from contextlib import contextmanager
from time import time


LOGGER = logging.getLogger("timeout")


def validate_classification(result=None, snapshot=None):
    for idx, _ in enumerate(result):
        assert result[idx].keys() == snapshot[idx].keys()
        # assert result[idx]["score"] >= snapshot[idx]["score"]
    return True


def validate_zero_shot_classification(result=None, snapshot=None):
    assert result.keys() == snapshot.keys()
    # assert result["labels"] == snapshot["labels"]
    # assert result["sequence"] == snapshot["sequence"]
    # for idx in range(len(result["scores"])):
    # assert result["scores"][idx] >= snapshot["scores"][idx]
    return True


def validate_ner(result=None, snapshot=None):
    assert result[0].keys() == snapshot[0].keys()
    # for idx, _ in enumerate(result):
    # assert result[idx]["score"] >= snapshot[idx]["score"]
    # assert result[idx]["entity"] == snapshot[idx]["entity"]
    # assert result[idx]["entity"] == snapshot[idx]["entity"]
    return True


def validate_question_answering(result=None, snapshot=None):
    assert result.keys() == snapshot.keys()
    # assert result["answer"] == snapshot["answer"]
    # assert result["score"] >= snapshot["score"]
    return True


def validate_summarization(result=None, snapshot=None):
    assert result is not None
    return True


def validate_text2text_generation(result=None, snapshot=None):
    assert result is not None
    return True


def validate_translation(result=None, snapshot=None):
    assert result is not None
    return True


def validate_text_generation(result=None, snapshot=None):
    assert result is not None
    return True


def validate_feature_extraction(result=None, snapshot=None):
    assert result is not None
    return True


def validate_fill_mask(result=None, snapshot=None):
    assert result is not None
    return True


def validate_automatic_speech_recognition(result=None, snapshot=None):
    assert result is not None
    assert "text" in result
    return True


def validate_object_detection(result=None, snapshot=None):
    assert result[0].keys() == snapshot[0].keys()
    return True


def validate_text_to_image(result=None, snapshot=None):
    assert isinstance(result, snapshot)
    return True
