import tempfile

import pytest
from transformers.testing_utils import require_torch

from huggingface_inference_toolkit.heavy_utils import (
    get_pipeline,
    load_repository_from_hf,
)
from huggingface_inference_toolkit.sentence_transformers_utils import (
    SentenceEmbeddingPipeline,
    get_sentence_transformers_pipeline,
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
        assert isinstance(res["embeddings"], list)
        res = pipe(sentences=["Lets create an embedding", "Lets create another embedding"])
        assert isinstance(res["embeddings"], list)
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
