import importlib.util
from typing import Any, Dict, List, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

_sentence_transformers = importlib.util.find_spec("sentence_transformers") is not None


def is_sentence_transformers_available():
    return _sentence_transformers


if is_sentence_transformers_available():
    from sentence_transformers import CrossEncoder, SentenceTransformer, util


class SentenceSimilarityPipeline:
    def __init__(self, model_dir: str, device: Union[str, None] = None, **kwargs: Any) -> None:
        # `device` needs to be set to "cuda" for GPU
        self.model = SentenceTransformer(model_dir, device=device, **kwargs)

    def __call__(self, source_sentence: str, sentences: List[str]) -> Dict[str, float]:
        embeddings1 = self.model.encode(source_sentence, convert_to_tensor=True)
        embeddings2 = self.model.encode(sentences, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(embeddings1, embeddings2).tolist()[0]
        return {"similarities": similarities}


class SentenceEmbeddingPipeline:
    def __init__(self, model_dir: str, device: Union[str, None] = None, **kwargs: Any) -> None:
        # `device` needs to be set to "cuda" for GPU
        self.model = SentenceTransformer(model_dir, device=device, **kwargs)

    def __call__(self, sentences: Union[str, List[str]]) -> Dict[str, List[float]]:
        embeddings = self.model.encode(sentences).tolist()
        return {"embeddings": embeddings}


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
