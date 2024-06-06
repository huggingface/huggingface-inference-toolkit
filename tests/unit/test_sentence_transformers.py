import os
import tempfile

from transformers import pipeline
from transformers.file_utils import is_torch_available
from transformers.testing_utils import require_tf, require_torch, slow

from huggingface_inference_toolkit.handler import (
    get_inference_handler_either_custom_or_default_handler,
)
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
            "sentence-transformers/all-MiniLM-L6-v2", tmpdirname, framework="pytorch"
        )
        pipe = get_pipeline("sentence-embeddings", storage_dir.as_posix())
        assert isinstance(pipe, SentenceEmbeddingPipeline)


@require_torch
def test_sentence_embedding_task():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_repository_from_hf(
            "sentence-transformers/all-MiniLM-L6-v2", tmpdirname, framework="pytorch"
        )
        pipe = get_sentence_transformers_pipeline(
            "sentence-embeddings", storage_dir.as_posix()
        )
        res = pipe("Lets create an embedding")
        assert isinstance(res["embeddings"], list)


@require_torch
def test_sentence_similarity():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_repository_from_hf(
            "sentence-transformers/all-MiniLM-L6-v2", tmpdirname, framework="pytorch"
        )
        pipe = get_sentence_transformers_pipeline(
            "sentence-similarity", storage_dir.as_posix()
        )
        res = pipe(
            {
                "source_sentence": "Lets create an embedding",
                "sentences": ["Lets create an embedding"],
            }
        )
        assert isinstance(res["similarities"], list)


@require_torch
def test_sentence_ranking():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_repository_from_hf(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", tmpdirname, framework="pytorch"
        )
        pipe = get_sentence_transformers_pipeline(
            "sentence-ranking", storage_dir.as_posix()
        )
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


@require_torch
def test_sentence_ranking_tei():
    with tempfile.TemporaryDirectory() as tmpdirname:
        storage_dir = _load_repository_from_hf(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", tmpdirname, framework="pytorch"
        )
        pipe = get_sentence_transformers_pipeline(
            "sentence-ranking", storage_dir.as_posix()
        )
        res = pipe(
            {
                "query": "Lets create an embedding",
                "texts": ["Lets create an embedding", "I like noodles"],
            }
        )
        assert isinstance(res, list)
        for r in res:
            assert "index" in r
            assert isinstance(r["score"], float) == True
