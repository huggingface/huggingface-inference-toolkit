"""Tasks whose transformers pipeline was deleted in v5.

transformers v5 removed the `question-answering`, `summarization`, `translation` and
`text2text-generation` pipelines (MIGRATION_GUIDE_V5.md), suggesting a chat model with
`TextGenerationPipeline` instead. That is advice for picking a model, not for a serving layer:
we still have to run the seq2seq and QA models that exist on the Hub, and routing them to
`text-generation` is not equivalent — encoder-decoder models are either rejected outright
(T5, Marian) or silently loaded as a decoder-only model with the encoder discarded (BART).

The model classes those pipelines need are all still in v5, so the pipelines themselves are
vendored under `vendor/` and dispatched from here, the same way the sentence-transformers and
diffusers tasks are.
"""

import re

from huggingface_inference_toolkit.vendor.question_answering import QuestionAnsweringPipeline
from huggingface_inference_toolkit.vendor.text2text_generation import (
    SummarizationPipeline,
    Text2TextGenerationPipeline,
    TranslationPipeline,
)

# task -> (vendored pipeline class, name of the auto model class to load it with)
LEGACY_TRANSFORMERS_TASKS = {
    "question-answering": (QuestionAnsweringPipeline, "AutoModelForQuestionAnswering"),
    "summarization": (SummarizationPipeline, "AutoModelForSeq2SeqLM"),
    "text2text-generation": (Text2TextGenerationPipeline, "AutoModelForSeq2SeqLM"),
    "translation": (TranslationPipeline, "AutoModelForSeq2SeqLM"),
}

# `translation_en_to_fr` & co: the language pair lives in the task name, and
# TranslationPipeline._sanitize_parameters parses it back out of `self.task`.
TRANSLATION_TASK_PATTERN = re.compile(r"^translation(_\w+_to_\w+)?$")


def normalize_legacy_task(task):
    """Return the LEGACY_TRANSFORMERS_TASKS key for `task`, or None if it is not one of them."""
    if task in LEGACY_TRANSFORMERS_TASKS:
        return task
    if task and TRANSLATION_TASK_PATTERN.match(task):
        return "translation"
    return None


def is_legacy_transformers_task(task) -> bool:
    return normalize_legacy_task(task) is not None


def get_legacy_transformers_pipeline(task=None, model_dir=None, device=-1, **kwargs):
    """Build one of the vendored pipelines for `task` from a local model directory."""
    import transformers

    pipeline_class, auto_model_name = LEGACY_TRANSFORMERS_TASKS[normalize_legacy_task(task)]

    # We load the tokenizer ourselves from model_dir, so drop what get_pipeline pre-filled.
    kwargs.pop("tokenizer", None)
    kwargs.pop("feature_extractor", None)
    trust_remote_code = kwargs.pop("trust_remote_code", False)

    model = getattr(transformers, auto_model_name).from_pretrained(
        model_dir, trust_remote_code=trust_remote_code
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code)

    # `task` is forwarded verbatim: TranslationPipeline needs the language pair it encodes.
    return pipeline_class(model=model, tokenizer=tokenizer, task=task, device=device, **kwargs)
