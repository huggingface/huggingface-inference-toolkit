import tempfile

import numpy as np
import pytest
from transformers.testing_utils import require_torch

from huggingface_inference_toolkit.heavy_utils import (
    get_pipeline,
    load_repository_from_hf,
)
from huggingface_inference_toolkit.sentence_transformers_utils import (
    SentenceEmbeddingPipeline,
    get_sentence_transformers_pipeline,
    is_multi_vector,
    st_model_type,
)


@require_torch
def test_get_sentence_transformers_pipeline():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = load_repository_from_hf("sentence-transformers/all-MiniLM-L6-v2", tmpdirname)
        pipe = get_pipeline("sentence-embeddings", storage_dir.as_posix())
        assert isinstance(pipe, SentenceEmbeddingPipeline)


@require_torch
def test_sentence_embedding_task():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = load_repository_from_hf("sentence-transformers/all-MiniLM-L6-v2", tmpdirname)
        pipe = get_sentence_transformers_pipeline("sentence-embeddings", storage_dir.as_posix())
        res = pipe(sentences="Lets create an embedding")
        assert isinstance(res["embeddings"], np.ndarray)
        res = pipe(sentences=["Lets create an embedding", "Lets create another embedding"])
        assert isinstance(res["embeddings"], np.ndarray)
        assert len(res["embeddings"]) == 2


@require_torch
def test_sentence_similarity():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = load_repository_from_hf("sentence-transformers/all-MiniLM-L6-v2", tmpdirname)
        pipe = get_sentence_transformers_pipeline("sentence-similarity", storage_dir.as_posix())
        res = pipe(source_sentence="Lets create an embedding", sentences=["Lets create an embedding"])
        assert isinstance(res["similarities"], list)


@require_torch
def test_sentence_ranking():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = load_repository_from_hf("cross-encoder/ms-marco-MiniLM-L-6-v2", tmpdirname)
        pipe = get_sentence_transformers_pipeline("sentence-ranking", storage_dir.as_posix())
        res = pipe(
            sentences=[
                ["Lets create an embedding", "Lets create another embedding"],
                ["Lets create an embedding", "Lets create another embedding"],
            ]
        )
        assert isinstance(res["scores"], list)
        res = pipe(sentences=["Lets create an embedding", "Lets create an embedding"])
        assert isinstance(res["scores"], float)


@require_torch
def test_sentence_ranking_tei():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = load_repository_from_hf("cross-encoder/ms-marco-MiniLM-L-6-v2", tmpdirname, framework="pytorch")
        pipe = get_sentence_transformers_pipeline("sentence-ranking", storage_dir.as_posix())
        res = pipe(
            query="Lets create an embedding",
            texts=["Lets create an embedding", "I like noodles"],
        )
        assert isinstance(res, list)
        assert all(r.keys() == {"index", "score"} for r in res)

        res = pipe(
            query="Lets create an embedding",
            texts=["Lets create an embedding", "I like noodles"],
            return_documents=True,
        )
        assert isinstance(res, list)
        assert all(r.keys() == {"index", "score", "text"} for r in res)


@require_torch
def test_sentence_ranking_validation_errors():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = load_repository_from_hf("cross-encoder/ms-marco-MiniLM-L-6-v2", tmpdirname, framework="pytorch")
        pipe = get_sentence_transformers_pipeline("sentence-ranking", storage_dir.as_posix())

        with pytest.raises(
            ValueError,
            match=(
                "you should provide either only 'sentences' i.e. 'inputs' "
                "of both 'query' and 'texts' to run the ranking task."
            ),
        ):
            pipe(
                sentences="Lets create an embedding",
                query="Lets create an embedding",
                texts=["Lets create an embedding", "I like noodles"],
            )

        with pytest.raises(
            ValueError,
            match=(
                "No inputs have been provided within the input payload, make sure that the input payload "
                "contains either 'sentences' i.e. 'inputs', or both 'query' and 'texts'"
            ),
        ):
            pipe(sentences=None, query=None, texts=None)

        with pytest.raises(
            ValueError,
            match=("Provided texts=None, but a list of non-empty strings should be provided instead."),
        ):
            pipe(sentences=None, query="Lets create an embedding", texts=None)


def test_st_model_type_reads_the_declared_family(tmp_path):
    (tmp_path / "config_sentence_transformers.json").write_text('{"model_type": "SparseEncoder"}')
    assert st_model_type(tmp_path.as_posix()) == "SparseEncoder"


def test_st_model_type_is_none_when_undeclared(tmp_path):
    # No file at all: the repository is not a sentence-transformers model
    assert st_model_type(tmp_path.as_posix()) is None
    # Present but without the key: a checkpoint saved before sentence-transformers 5.0
    (tmp_path / "config_sentence_transformers.json").write_text('{"__version__": {}}')
    assert st_model_type(tmp_path.as_posix()) is None
    # Unreadable: treated as undeclared rather than failing the load
    (tmp_path / "config_sentence_transformers.json").write_text("not json")
    assert st_model_type(tmp_path.as_posix()) is None


@pytest.mark.parametrize(
    "declared,expected",
    [
        ("SparseEncoder", "SparseEncoder"),
        # A late-interaction model has no single embedding per text to return, so this task is
        # not where it is handled -- see the similarity test below.
        ("ColBERT", "SentenceTransformer"),
        ("SentenceTransformer", "SentenceTransformer"),
        (None, "SentenceTransformer"),
    ],
)
def test_sentence_embedding_pipeline_dispatches_on_the_declared_family(
    tmp_path, monkeypatch, declared, expected
):
    """The class comes from the checkpoint, not from the task: both families serve the same task."""
    if declared is not None:
        (tmp_path / "config_sentence_transformers.json").write_text(
            '{"model_type": "%s"}' % declared
        )

    import huggingface_inference_toolkit.sentence_transformers_utils as stu

    loaded = {}

    def fake(name):
        def _init(model_dir, device=None, **kwargs):
            loaded["class"] = name
            return object()

        return _init

    monkeypatch.setattr(stu, "SparseEncoder", fake("SparseEncoder"))
    monkeypatch.setattr(stu, "SentenceTransformer", fake("SentenceTransformer"))
    stu.SentenceEmbeddingPipeline(tmp_path.as_posix(), device="cpu")
    assert loaded["class"] == expected


@pytest.mark.parametrize(
    "config,modules,expected",
    [
        # declared outright, as sentence-transformers >= 5.0 writes it
        ('{"model_type": "ColBERT"}', None, "MultiVectorEncoder"),
        # saved before 5.0: no model_type, but it asks for the scoring it needs
        ('{"similarity_fn_name": "MaxSim"}', None, "MultiVectorEncoder"),
        # older still: neither, but built by PyLate
        ('{"similarity_fn_name": null}', '[{"type": "pylate.models.Dense.Dense"}]', "MultiVectorEncoder"),
        ("{}", '[{"type": "sentence_transformers.models.Pooling"}]', "SentenceTransformer"),
        ('{"model_type": "SentenceTransformer"}', None, "SentenceTransformer"),
        ('{"model_type": "SparseEncoder"}', None, "SentenceTransformer"),
        (None, None, "SentenceTransformer"),
    ],
)
def test_sentence_similarity_pipeline_dispatches_on_the_checkpoint(
    tmp_path, monkeypatch, config, modules, expected
):
    """Scoring a late-interaction checkpoint needs its own class: MaxSim, not a cosine."""
    if config is not None:
        (tmp_path / "config_sentence_transformers.json").write_text(config)
    if modules is not None:
        (tmp_path / "modules.json").write_text(modules)

    import huggingface_inference_toolkit.sentence_transformers_utils as stu

    loaded = {}

    def fake(name):
        def _init(model_dir, device=None, **kwargs):
            loaded["class"] = name
            loaded["kwargs"] = kwargs
            return object()

        return _init

    monkeypatch.setattr(stu, "MultiVectorEncoder", fake("MultiVectorEncoder"))
    monkeypatch.setattr(stu, "SentenceTransformer", fake("SentenceTransformer"))
    pipe = stu.SentenceSimilarityPipeline(tmp_path.as_posix(), device="cpu")
    assert loaded["class"] == expected
    assert pipe.is_multi_vector == (expected == "MultiVectorEncoder")
    if expected == "MultiVectorEncoder":
        # Bounded scores, so the task answers in the same range as every other model on it
        assert loaded["kwargs"]["similarity_fn_name"] == "meanmaxsim"


@pytest.mark.parametrize(
    "config,modules,expected",
    [
        ('{"model_type": "ColBERT"}', None, True),
        ('{"similarity_fn_name": "MaxSim"}', None, True),
        ('{"similarity_fn_name": "maxsim"}', None, True),
        (None, '[{"type": "pylate.models.Dense.Dense"}]', True),
        ('{"model_type": "SparseEncoder"}', None, False),
        ('{"model_type": "SentenceTransformer"}', '[{"type": "sentence_transformers.models.Pooling"}]', False),
        ("not json", "not json", False),
        (None, None, False),
    ],
)
def test_is_multi_vector_accepts_every_marker_these_checkpoints_carry(
    tmp_path, config, modules, expected
):
    # Of the ten late-interaction checkpoints HF Inference serves, only four declare `model_type`;
    # the rest are recognised by the scoring they ask for or by their PyLate module.
    if config is not None:
        (tmp_path / "config_sentence_transformers.json").write_text(config)
    if modules is not None:
        (tmp_path / "modules.json").write_text(modules)
    assert is_multi_vector(tmp_path.as_posix()) is expected
