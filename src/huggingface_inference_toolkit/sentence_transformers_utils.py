import importlib.util

_sentence_transformers = importlib.util.find_spec("hf_api_sentence_transformers") is not None


def is_sentence_transformers_available():
    return _sentence_transformers


if is_sentence_transformers_available():
    from hf_api_sentence_transformers import FeatureExtractionPipeline
    from hf_api_sentence_transformers import SentenceSimilarityPipeline as SentenceSimilarityPipelineImpl


class SentenceSimilarityPipeline:
    def __init__(self, model_dir: str, device: str = None, **kwargs):  # needs "cuda" for GPU
        self.model = SentenceSimilarityPipelineImpl(model_dir)

    def __call__(self, inputs=None):
        return {"similarities": self.model(inputs)}


class SentenceEmbeddingPipeline:
    def __init__(self, model_dir: str, device: str = None, **kwargs):  # needs "cuda" for GPU
        self.model = FeatureExtractionPipeline(model_dir)

    def __call__(self, inputs):
        return {"embeddings": self.model(inputs)}



SENTENCE_TRANSFORMERS_TASKS = {
    "sentence-similarity": SentenceSimilarityPipeline,
    "sentence-embeddings": SentenceEmbeddingPipeline,
    #"sentence-ranking": RankingPipeline, # To be implemented
}


def get_sentence_transformers_pipeline(task=None, model_dir=None, device=-1, **kwargs):
    device = "cuda" if device == 0 else "cpu"

    kwargs.pop("tokenizer", None)
    kwargs.pop("framework", None)

    if task not in SENTENCE_TRANSFORMERS_TASKS:
        raise ValueError(
            f"Unknown task {task}. Available tasks are: {', '.join(SENTENCE_TRANSFORMERS_TASKS.keys())}"
        )
    return SENTENCE_TRANSFORMERS_TASKS[task](model_dir=model_dir)
