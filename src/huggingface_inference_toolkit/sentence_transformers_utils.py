import importlib.util
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from huggingface_inference_toolkit.env_utils import api_inference_compat

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

_sentence_transformers = importlib.util.find_spec("sentence_transformers") is not None


def is_sentence_transformers_available():
    return _sentence_transformers


if is_sentence_transformers_available():
    from sentence_transformers import CrossEncoder, SentenceTransformer, SparseEncoder, util


def st_model_type(model_dir: str) -> Optional[str]:
    """
    The model family the checkpoint declares in `config_sentence_transformers.json`.

    sentence-transformers writes it from 5.0 onwards ("SentenceTransformer", "SparseEncoder",
    "CrossEncoder", "ColBERT"); older checkpoints have no key, and repositories that are not
    sentence-transformers models have no file. Both mean "load it as a dense model", which is
    what those checkpoints are.
    """
    try:
        with open(os.path.join(model_dir, "config_sentence_transformers.json")) as f:
            return json.load(f).get("model_type")
    except (OSError, ValueError):
        return None


class SentenceSimilarityPipeline:
    def __init__(self, model_dir: str, device: Union[str, None] = None, **kwargs: Any) -> None:
        # `device` needs to be set to "cuda" for GPU
        self.model = SentenceTransformer(model_dir, device=device, **kwargs)

    def __call__(self, source_sentence: str, sentences: List[str]) -> Dict[str, float]:
        embeddings1 = self.model.encode(source_sentence, convert_to_tensor=True)
        embeddings2 = self.model.encode(sentences, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(embeddings1, embeddings2).tolist()[0]
        # The widgets expect the bare list, not a wrapper object
        return similarities if api_inference_compat() else {"similarities": similarities}


class SentenceEmbeddingPipeline:
    def __init__(self, model_dir: str, device: Union[str, None] = None, **kwargs: Any) -> None:
        # The task cannot tell the two families apart -- a SPLADE checkpoint and a dense embedding
        # model are both served as `sentence-embeddings` -- so the checkpoint decides, not HF_TASK.
        #
        # This matters because the wrong class is not an error. Asked for a SparseEncoder
        # checkpoint, SentenceTransformer performs a cross-family conversion: it drops
        # `cls.predictions.*`, the MLM head that produces the vocabulary-space scores SPLADE *is*,
        # initializes an untrained pooler in its place, and returns a dense hidden-state vector.
        # On sentence-transformers 5.x that happens with no warning at all, so the endpoint answers
        # 200 with an embedding that is the wrong length, wrong density and wrong sign range.
        #
        # `ColBERT` is deliberately not routed here: it needs MultiVectorEncoder, which arrives in
        # sentence-transformers 6.0, so those checkpoints keep failing loudly for now.
        # `device` needs to be set to "cuda" for GPU
        if st_model_type(model_dir) == "SparseEncoder":
            self.model = SparseEncoder(model_dir, device=device, **kwargs)
        else:
            self.model = SentenceTransformer(model_dir, device=device, **kwargs)

    def __call__(self, sentences: Union[str, List[str]]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        # Deliberately not `.tolist()`: it widens float32 to float64 and serializes
        # 0.1 as 0.10000000149011612.
        embeddings = self.model.encode(sentences)
        # SparseEncoder returns a torch sparse tensor, where SentenceTransformer already returns an
        # array. Densified rather than returned as indices/values so the response stays the flat
        # vector every caller of this task already expects.
        if hasattr(embeddings, "to_dense"):
            embeddings = embeddings.to_dense()
        if hasattr(embeddings, "cpu"):
            embeddings = embeddings.cpu().numpy()
        # The widgets expect the bare array, not a wrapper object
        return embeddings if api_inference_compat() else {"embeddings": embeddings}


class SentenceRankingPipeline:
    def __init__(self, model_dir: str, device: Union[str, None] = None, **kwargs: Any) -> None:
        # `device` needs to be set to "cuda" for GPU
        self.model = CrossEncoder(model_dir, device=device, **kwargs)

    def __call__(
        self,
        sentences: Union[Tuple[str, str], List[str], List[List[str]], List[Tuple[str, str]], None] = None,
        query: Union[str, None] = None,
        texts: Union[List[str], None] = None,
        return_documents: bool = False,
    ) -> Union[Dict[str, List[float]], List[Dict[Literal["index", "score", "text"], Any]]]:
        if all(x is not None for x in [sentences, query, texts]):
            raise ValueError(
                f"The provided payload contains {sentences=} (i.e. 'inputs'), {query=}, and {texts=}"
                " but all of those cannot be provided, you should provide either only 'sentences' i.e. 'inputs'"
                " of both 'query' and 'texts' to run the ranking task."
            )

        if all(x is None for x in [sentences, query, texts]):
            raise ValueError(
                "No inputs have been provided within the input payload, make sure that the input payload"
                " contains either 'sentences' i.e. 'inputs', or both 'query' and 'texts' to run the ranking task."
            )

        if sentences is not None:
            scores = self.model.predict(sentences).tolist()
            return {"scores": scores}

        if query is None or not isinstance(query, str):
            raise ValueError(f"Provided {query=} but a non-empty string should be provided instead.")

        if texts is None or not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise ValueError(f"Provided {texts=}, but a list of non-empty strings should be provided instead.")

        scores = self.model.rank(query, texts, return_documents=return_documents)
        # rename "corpus_id" key to "index" for all scores to match TEI
        for score in scores:
            score["index"] = score.pop("corpus_id")  # type: ignore
        return scores  # type: ignore


SENTENCE_TRANSFORMERS_TASKS = {
    "sentence-similarity": SentenceSimilarityPipeline,
    "sentence-embeddings": SentenceEmbeddingPipeline,
    "sentence-ranking": SentenceRankingPipeline,
}


def get_sentence_transformers_pipeline(task=None, model_dir=None, device=-1, **kwargs):
    device = "cuda" if device == 0 else "cpu"

    kwargs.pop("tokenizer", None)
    kwargs.pop("framework", None)

    if task not in SENTENCE_TRANSFORMERS_TASKS:
        raise ValueError(f"Unknown task {task}. Available tasks are: {', '.join(SENTENCE_TRANSFORMERS_TASKS.keys())}")
    return SENTENCE_TRANSFORMERS_TASKS[task](model_dir=model_dir, device=device, **kwargs)
