import os

from tests.integ.utils import (
    validate_automatic_speech_recognition,
    validate_classification,
    validate_feature_extraction,
    validate_fill_mask,
    validate_ner,
    validate_object_detection,
    validate_question_answering,
    validate_summarization,
    validate_text2text_generation,
    validate_text_generation,
    validate_text_to_image,
    validate_translation,
    validate_zero_shot_classification,
)


task2model = {
    "text-classification": {
        "pytorch": "hf-internal-testing/tiny-random-distilbert",
        "tensorflow": "hf-internal-testing/tiny-random-distilbert",
    },
    "zero-shot-classification": {
        "pytorch": "hf-internal-testing/tiny-random-bart",
        "tensorflow": "typeform/distilbert-base-uncased-mnli",
    },
    "feature-extraction": {
        "pytorch": "hf-internal-testing/tiny-random-bert",
        "tensorflow": "hf-internal-testing/tiny-random-bert",
    },
    "ner": {
        "pytorch": "hf-internal-testing/tiny-random-roberta",
        "tensorflow": "hf-internal-testing/tiny-random-roberta",
    },
    "question-answering": {
        "pytorch": "hf-internal-testing/tiny-random-electra",
        "tensorflow": "hf-internal-testing/tiny-random-electra",
    },
    "fill-mask": {
        "pytorch": "hf-internal-testing/tiny-random-bert",
        "tensorflow": "hf-internal-testing/tiny-random-bert",
    },
    "summarization": {
        "pytorch": "hf-internal-testing/tiny-random-bart",
        "tensorflow": "hf-internal-testing/tiny-random-bart",
    },
    "translation_xx_to_yy": {
        "pytorch": "hf-internal-testing/tiny-random-t5",
        "tensorflow": "hf-internal-testing/tiny-random-marian",
    },
    "text2text-generation": {
        "pytorch": "hf-internal-testing/tiny-random-t5",
        "tensorflow": "hf-internal-testing/tiny-random-bart",
    },
    "text-generation": {
        "pytorch": "hf-internal-testing/tiny-random-gpt2",
        "tensorflow": "hf-internal-testing/tiny-random-gpt2",
    },
    "image-classification": {
        "pytorch": "hf-internal-testing/tiny-random-vit",
        "tensorflow": "hf-internal-testing/tiny-random-vit",
    },
    "automatic-speech-recognition": {
        "pytorch": "hf-internal-testing/tiny-random-wav2vec2",
        "tensorflow": None,
    },
    "audio-classification": {
        "pytorch": "hf-internal-testing/tiny-random-wavlm",
        "tensorflow": None,
    },
    "object-detection": {
        "pytorch": "hustvl/yolos-tiny",
        "tensorflow": None,
    },
    "image-segmentation": {
        "pytorch": "hf-internal-testing/tiny-random-beit-pipeline",
        "tensorflow": None,
    },
    "table-question-answering": {
        "pytorch": "philschmid/tapex-tiny",
        "tensorflow": None,
    },
    "zero-shot-image-classification": {
        "pytorch": "hf-internal-testing/tiny-random-clip-zero-shot-image-classification",
        "tensorflow": "hf-internal-testing/tiny-random-clip-zero-shot-image-classification",
    },
    "conversational": {
        "pytorch": "hf-internal-testing/tiny-random-blenderbot",
        "tensorflow": "hf-internal-testing/tiny-random-blenderbot",
    },
    "sentence-similarity": {
        "pytorch": "sentence-transformers/all-MiniLM-L6-v2",
        "tensorflow": None,
    },
    "sentence-embeddings": {
        "pytorch": "sentence-transformers/all-MiniLM-L6-v2",
        "tensorflow": None,
    },
    "sentence-ranking": {
        "pytorch": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "tensorflow": None,
    },
    "text-to-image": {
        "pytorch": "hf-internal-testing/tiny-stable-diffusion-torch",
        "tensorflow": None,
    },
}


task2input = {
    "text-classification": {"inputs": "I love you. I like you"},
    "zero-shot-classification": {
        "inputs": "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!",
        "parameters": {"candidate_labels": ["refund", "legal", "faq"]},
    },
    "feature-extraction": {"inputs": "What is the best book."},
    "ner": {"inputs": "My name is Wolfgang and I live in Berlin"},
    "question-answering": {
        "inputs": {
            "question": "What is used for inference?",
            "context": "My Name is Philipp and I live in Nuremberg. This model is used with sagemaker for inference.",
        }
    },
    "fill-mask": {"inputs": "Paris is the [MASK] of France."},
    "summarization": {
        "inputs": "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."
    },
    "translation_xx_to_yy": {"inputs": "My name is Sarah and I live in London"},
    "text2text-generation": {
        "inputs": "question: What is 42 context: 42 is the answer to life, the universe and everything."
    },
    "text-generation": {"inputs": "My name is philipp and I am"},
    "image-classification": open(os.path.join(os.getcwd(), "tests/resources/image/tiger.jpeg"), "rb").read(),
    "zero-shot-image-classification": open(os.path.join(os.getcwd(), "tests/resources/image/tiger.jpeg"), "rb").read(),
    "object-detection": open(os.path.join(os.getcwd(), "tests/resources/image/tiger.jpeg"), "rb").read(),
    "image-segmentation": open(os.path.join(os.getcwd(), "tests/resources/image/tiger.jpeg"), "rb").read(),
    "automatic-speech-recognition": open(os.path.join(os.getcwd(), "tests/resources/audio/sample1.flac"), "rb").read(),
    "audio-classification": open(os.path.join(os.getcwd(), "tests/resources/audio/sample1.flac"), "rb").read(),
    "table-question-answering": {
        "inputs": {
            "query": "How many stars does the transformers repository have?",
            "table": {
                "Repository": ["Transformers", "Datasets", "Tokenizers"],
                "Stars": ["36542", "4512", "3934"],
                "Contributors": ["651", "77", "34"],
                "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
            },
        }
    },
    "conversational": {
        "inputs": {
            "past_user_inputs": ["Which movie is the best ?"],
            "generated_responses": ["It's Die Hard for sure."],
            "text": "Can you explain why?",
        }
    },
    "sentence-similarity": {
        "inputs": {"source_sentence": "Lets create an embedding", "sentences": ["Lets create an embedding"]}
    },
    "sentence-embeddings": {"inputs": "Lets create an embedding"},
    "sentence-ranking": {"inputs": ["Lets create an embedding", "Lets create an embedding"]},
    "text-to-image": {"inputs": "a man on a horse jumps over a broken down airplane."},
}

task2output = {
    "text-classification": [{"label": "POSITIVE", "score": 0.01}],
    "zero-shot-classification": {
        "sequence": "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!",
        "labels": ["refund", "faq", "legal"],
        "scores": [0.96, 0.027, 0.008],
    },
    "ner": [
        {"word": "Wolfgang", "score": 0.99, "entity": "I-PER", "index": 4, "start": 11, "end": 19},
        {"word": "Berlin", "score": 0.99, "entity": "I-LOC", "index": 9, "start": 34, "end": 40},
    ],
    "question-answering": {"score": 0.99, "start": 68, "end": 77, "answer": "sagemaker"},
    "summarization": [{"summary_text": " The A The The ANew York City has been installed in the US."}],
    "translation_xx_to_yy": [{"translation_text": "Mein Name ist Sarah und ich lebe in London"}],
    "text2text-generation": [{"generated_text": "42 is the answer to life, the universe and everything"}],
    "feature-extraction": None,
    "fill-mask": None,
    "text-generation": None,
    "image-classification": [
        {"score": 0.8858247399330139, "label": "tiger, Panthera tigris"},
        {"score": 0.10940514504909515, "label": "tiger cat"},
        {"score": 0.0006216464680619538, "label": "jaguar, panther, Panthera onca, Felis onca"},
        {"score": 0.0004262699221726507, "label": "dhole, Cuon alpinus"},
        {"score": 0.00030842673731967807, "label": "lion, king of beasts, Panthera leo"},
    ],
    "zero-shot-image-classification": [
        {"score": 0.8858247399330139, "label": "tiger, Panthera tigris"},
        {"score": 0.10940514504909515, "label": "tiger cat"},
        {"score": 0.0006216464680619538, "label": "jaguar, panther, Panthera onca, Felis onca"},
        {"score": 0.0004262699221726507, "label": "dhole, Cuon alpinus"},
        {"score": 0.00030842673731967807, "label": "lion, king of beasts, Panthera leo"},
    ],
    "automatic-speech-recognition": {
        "text": "GOING ALONG SLUSHY COUNTRY ROADS AND SPEAKING TO DAMP OAUDIENCES IN DROFTY SCHOOL ROOMS DAY AFTER DAY FOR A FORT NIGHT HE'LL HAVE TO PUT IN AN APPEARANCE AT SOME PLACE OF WORSHIP ON SUNDAY MORNING AND HE CAN COME TO US IMMEDIATELY AFTERWARDS"
    },
    "audio-classification": [
        {"label": "no", "score": 0.5052680969238281},
        {"label": "yes", "score": 0.49473199248313904},
    ],
    "object-detection": [{"score": 0.9143241047859192, "label": "cat", "box": {}}],
    "image-segmentation": [{"score": 0.9143241047859192, "label": "cat", "mask": {}}],
    "table-question-answering": {"answer": "36542"},
    "conversational": {"generated_text": "", "conversation": {}},
    "sentence-similarity": {"similarities": ""},
    "sentence-embeddings": {"embeddings": ""},
    "sentence-ranking": {"scores": ""},
    "text-to-image": bytes,
}


task2validation = {
    "text-classification": validate_classification,
    "zero-shot-classification": validate_zero_shot_classification,
    "zero-shot-image-classification": validate_zero_shot_classification,
    "feature-extraction": validate_feature_extraction,
    "ner": validate_ner,
    "question-answering": validate_question_answering,
    "fill-mask": validate_fill_mask,
    "summarization": validate_summarization,
    "translation_xx_to_yy": validate_translation,
    "text2text-generation": validate_text2text_generation,
    "text-generation": validate_text_generation,
    "image-classification": validate_classification,
    "automatic-speech-recognition": validate_automatic_speech_recognition,
    "audio-classification": validate_classification,
    "object-detection": validate_object_detection,
    "image-segmentation": validate_object_detection,
    "table-question-answering": validate_zero_shot_classification,
    "conversational": validate_zero_shot_classification,
    "sentence-similarity": validate_zero_shot_classification,
    "sentence-embeddings": validate_zero_shot_classification,
    "sentence-ranking": validate_zero_shot_classification,
    "text-to-image": validate_text_to_image,
}
