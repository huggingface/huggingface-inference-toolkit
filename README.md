# Hugging Face Inference Toolkit 

## Get Started

### Starlette

```bash
pip install -r starlette_requirements.txt
```

```bash
uvicorn src.huggingface_inference_toolkit.webservice_starlette:app
```

## Docker

1. build docker container 

_cpu images_
```bash
docker build -t starlette-transformers:cpu -f dockerfiles/starlette/Dockerfile.cpu .
```

_gpu images_
```bash
docker build -t starlette-transformers:gpu -f dockerfiles/starlette/Dockerfile.gpu .
```

2. run container

```bash
docker run -ti -p 5000:5000 starlette-transformers:cpu
```

## Hey Benchmark

```bash
hey -n 1000 -m POST -H 'Content-Type: application/json' -d '{	"inputs": "I love you. I like you. I am your friend."}' http://127.0.0.1:5000/predict
```

--workers=1


## Supported & Tested Tasks

```bash
"text-classification",
"zero-shot-classification",
"ner",
"question-answering",
"fill-mask",
"summarization",
"translation_xx_to_yy",
"text2text-generation",
"text-generation",
"feature-extraction",
"image-classification",
"automatic-speech-recognition",
"audio-classification",
"object-detection",
"image-segmentation",
"table-question-answering",
"conversational"
"sentence-similarity",
"sentence-embeddings",
"sentence-ranking",
# TODO currently not supported due to multimodality input
# "visual-question-answering",
# "zero-shot-image-classification",
```