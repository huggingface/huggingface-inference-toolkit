import logging


def validate_classification(result=None, snapshot=None):
    for idx, _ in enumerate(result):
        assert result[idx].keys() == snapshot[idx].keys()
    return True


def validate_conversational(result=None, snapshot=None):
    assert len(result[0]["generated_text"]) >= len(snapshot)


def validate_zero_shot_classification(result=None, snapshot=None):
    logging.info(f"Result: {result}")
    logging.info(f"Snapshot: {snapshot}")
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


def validate_image_text_to_text(result=None, snapshot=None):
    assert isinstance(result, list)
    assert all(isinstance(d, dict) and d.keys() == {"input_text", "generated_text"} for d in result)
    return True


def validate_custom(result=None, snapshot=None):
    logging.info(f"Validate custom task - result: {result}, snapshot: {snapshot}")
    assert result == snapshot
    return True
