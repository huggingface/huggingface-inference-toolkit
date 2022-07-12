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
docker build -t robyn-transformers:cpu -f dockerfiles/Dockerfile.cpu .
```

_gpu images_
```bash
docker build -t robyn-transformers:gpu -f dockerfiles/Dockerfile.gpu .
```

2. run container

```bash
docker run -it -p 5000:5000 robyn-transformers:cpu
```

## Hey Benchmark

```bash
hey -n 1000 -m POST -H 'Content-Type: application/json' -d '{	"inputs": "I love you. I like you. I am your friend."}' http://127.0.0.1:5000/predict
```

--workers=1


