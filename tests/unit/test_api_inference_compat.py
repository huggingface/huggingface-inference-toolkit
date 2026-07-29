import pytest

from huggingface_inference_toolkit.env_utils import api_inference_compat
from huggingface_inference_toolkit.handler import HuggingFaceHandler


class FakePipeline:
    """A transformers-pipeline stand-in: a task name and a recorded call."""

    def __init__(self, task, returns=None):
        self.task = task
        self.returns = returns
        self.calls = []

    def __call__(self, *args, **kwargs):
        # dict inputs are spread as kwargs by the handler, everything else is positional
        self.calls.append((args[0] if args else None, kwargs))
        return self.returns


def handler_for(task, returns=None):
    return HuggingFaceHandler(FakePipeline(task, returns))


@pytest.fixture
def compat_on(monkeypatch):
    monkeypatch.setenv("API_INFERENCE_COMPAT", "true")


@pytest.mark.parametrize(
    "value,expected",
    [(None, False), ("false", False), ("0", False), ("true", True), ("1", True), ("YES", True)],
)
def test_api_inference_compat_env(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("API_INFERENCE_COMPAT", raising=False)
    else:
        monkeypatch.setenv("API_INFERENCE_COMPAT", value)
    assert api_inference_compat() is expected


def test_off_by_default_leaves_the_response_untouched(monkeypatch):
    monkeypatch.delenv("API_INFERENCE_COMPAT", raising=False)
    flat = [{"label": "POSITIVE", "score": 0.9}]
    handler = handler_for("text-classification", returns=flat)

    assert handler({"inputs": "good"}) is flat
    # no top_k default injected either
    assert handler.pipeline.calls == [("good", {})]


def test_text_classification_batches_a_single_string_and_asks_for_the_ranking(compat_on):
    nested = [[{"label": "POSITIVE", "score": 0.9}, {"label": "NEGATIVE", "score": 0.1}]]
    handler = handler_for("text-classification", returns=nested)

    assert handler({"inputs": "good"}) == nested
    assert handler.pipeline.calls == [(["good"], {"top_k": 5})]


def test_default_top_k_is_passed_as_an_int(compat_on, monkeypatch):
    # it reaches the pipeline as a string otherwise, which transformers rejects
    monkeypatch.setenv("DEFAULT_TOP_K", "3")
    handler = handler_for("text-classification", returns=[[]])

    handler({"inputs": "good"})

    assert handler.pipeline.calls == [(["good"], {"top_k": 3})]


def test_caller_parameters_win_over_the_defaults(compat_on):
    handler = handler_for("text-classification", returns=[[]])

    handler({"inputs": "good", "parameters": {"top_k": 1}})

    assert handler.pipeline.calls == [(["good"], {"top_k": 1})]


def test_token_classification_gets_an_aggregation_strategy(compat_on):
    handler = handler_for("token-classification", returns=[])

    handler({"inputs": "Wolfgang lives in Berlin"})

    assert handler.pipeline.calls == [("Wolfgang lives in Berlin", {"aggregation_strategy": "simple"})]


A = {"label": "A", "score": 0.9}
B = {"label": "B", "score": 0.1}


@pytest.mark.parametrize(
    "inputs,resp,expected",
    [
        # one input, flat ranking -> the whole list is that input's ranking
        (["a"], [A, B], [[A, B]]),
        (["a"], [A], [[A]]),
        # already grouped per input: untouched
        (["a"], [[A, B]], [[A, B]]),
        (["a", "b"], [[A], [B]], [[A], [B]]),
        # several inputs, flat: one entry per input (what top_k=1 returns), so group each on its
        # own rather than wrapping the lot — wrapping would lose which result belongs to which input
        (["a", "b", "c"], [A, B, A], [[A], [B], [A]]),
        ([], [], []),
    ],
)
def test_text_classification_groups_scores_per_input(compat_on, inputs, resp, expected):
    handler = handler_for("text-classification", returns=resp)
    assert handler({"inputs": inputs}) == expected


def test_text_classification_leaves_a_length_mismatch_alone(compat_on):
    # 2 inputs but 3 results: we cannot tell how they map, so do not invent a grouping
    resp = [A, B, A]
    handler = handler_for("text-classification", returns=resp)
    assert handler({"inputs": ["a", "b"]}) == resp


def test_text_classification_does_not_default_top_k_for_list_inputs(compat_on):
    # only a single string is batched and given the full ranking; defaulting top_k here would
    # change how many labels each input comes back with
    handler = handler_for("text-classification", returns=[A, B])
    handler({"inputs": ["a", "b"]})
    assert handler.pipeline.calls == [(["a", "b"], {})]


def test_feature_extraction_drops_the_batch_dimension(compat_on):
    handler = handler_for("feature-extraction", returns=[[[1.0, 2.0]], [[3.0, 4.0]]])
    assert handler({"inputs": ["a", "b"]}) == [[1.0, 2.0], [3.0, 4.0]]

    handler = handler_for("feature-extraction", returns=[[[1.0, 2.0]]])
    assert handler({"inputs": "a"}) == [[1.0, 2.0]]


@pytest.mark.parametrize(
    "inputs,resp",
    [
        # resp/input length mismatch
        (["a", "b"], [[[1.0]]]),
        # a batch dimension that is not 1
        (["a"], [[[1.0], [2.0]]]),
        # more than one embedding for a single string input
        ("a", [[[1.0]], [[2.0]]]),
        # not a list at all
        (["a"], {"unexpected": True}),
    ],
)
def test_feature_extraction_leaves_unexpected_shapes_alone(compat_on, inputs, resp):
    handler = handler_for("feature-extraction", returns=resp)
    assert handler({"inputs": inputs}) == resp


def test_image_segmentation_gets_a_score(compat_on):
    handler = handler_for(
        "image-segmentation",
        returns=[{"label": "cat", "mask": "..."}, {"label": "dog", "mask": "...", "score": 0.5}],
    )

    assert handler({"inputs": "img"}) == [
        {"label": "cat", "mask": "...", "score": 1},
        {"label": "dog", "mask": "...", "score": 0.5},
    ]


def test_zero_shot_classification_becomes_label_score_pairs(compat_on):
    handler = handler_for(
        "zero-shot-classification",
        returns={"sequence": "s", "labels": ["a", "b"], "scores": [0.7, 0.3]},
    )

    assert handler({"inputs": "s", "parameters": {"candidate_labels": ["a", "b"]}}) == [
        {"label": "a", "score": 0.7},
        {"label": "b", "score": 0.3},
    ]


@pytest.mark.parametrize(
    "resp",
    [
        {"labels": ["a", "b"], "scores": [0.7]},  # parallel arrays of different length
        {"labels": ["a"]},  # scores missing
        [{"label": "a", "score": 1.0}],  # not a dict
    ],
)
def test_zero_shot_classification_leaves_unexpected_shapes_alone(compat_on, resp):
    handler = handler_for("zero-shot-classification", returns=resp)
    assert handler({"inputs": "s", "parameters": {"candidate_labels": ["a"]}}) == resp


def test_an_unmapped_task_is_passed_through(compat_on):
    resp = {"answer": "42", "score": 0.9}
    handler = handler_for("question-answering", returns=resp)
    assert handler({"inputs": {"question": "q", "context": "c"}}) is resp
