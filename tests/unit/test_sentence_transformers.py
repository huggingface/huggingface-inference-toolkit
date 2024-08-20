import tempfile

from transformers.testing_utils import require_torch

from huggingface_inference_toolkit.sentence_transformers_utils import (
    SentenceEmbeddingPipeline,
    get_sentence_transformers_pipeline,
)
from huggingface_inference_toolkit.utils import (
    _load_repository_from_hf,
    get_pipeline,
)


@require_torch
def test_get_sentence_transformers_pipeline():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_repository_from_hf(
            "sentence-transformers/all-MiniLM-L6-v2", tmpdirname
        )
        pipe = get_pipeline("sentence-embeddings", storage_dir.as_posix())
        assert isinstance(pipe, SentenceEmbeddingPipeline)


@require_torch
def test_sentence_embedding_task():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_repository_from_hf(
            "sentence-transformers/all-MiniLM-L6-v2", tmpdirname
        )
        pipe = get_sentence_transformers_pipeline("sentence-embeddings", storage_dir.as_posix())
        res = pipe("Lets create an embedding")
        assert isinstance(res["embeddings"], list)


@require_torch
def test_sentence_similarity():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_repository_from_hf(
            "sentence-transformers/all-MiniLM-L6-v2", tmpdirname
        )
        pipe = get_sentence_transformers_pipeline("sentence-similarity", storage_dir.as_posix())
        res = pipe({"source_sentence": "Lets create an embedding", "sentences": ["Lets create an embedding"]})
        assert isinstance(res["similarities"], list)


@require_torch
def test_sentence_ranking():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_repository_from_hf("cross-encoder/ms-marco-MiniLM-L-6-v2", tmpdirname)
        pipe = get_sentence_transformers_pipeline("sentence-ranking", storage_dir.as_posix())
        res = pipe(
            [
                ["Lets create an embedding", "Lets create an embedding"],
                ["Lets create an embedding", "Lets create an embedding"],
            ]
        )
        assert isinstance(res["scores"], list)
        res = pipe(
            ["Lets create an embedding", "Lets create an embedding"],
        )
        assert isinstance(res["scores"], float)
