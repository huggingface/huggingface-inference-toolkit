<img src="https://huggingface.co/front/assets/huggingface_logo.svg" width="100"/>

# Hugging Face Inference Toolkit 

Hugging Face Inference Toolkit is for serving 🤗 Transformers models in containers. This library provides default pre-processing, prediction, and postprocessing for Transformers, diffusers, and Sentence Transformers. It is also possible to define a custom `handler.py` for customization. The Toolkit is built to work with the [Hugging Face Hub](https://huggingface.co/models) and is used as the "default" option in [Inference Endpoints](https://ui.endpoints.huggingface.co/)

## 💻 Getting Started with Hugging Face Inference Toolkit

- Clone the repository `git clone https://github.com/huggingface/huggingface-inference-toolkit`
- Install the dependencies in dev mode `pip install -e ".[torch,st,diffusers,test,quality]"`
  - If you develop on AWS Inferentia2 install with `pip install -e ".[inf2,test,quality]" --upgrade`
  - If you develop on Google Cloud install with `pip install -e ".[torch,st,diffusers,google,test,quality]"`
- Unit Testing: `make unit-test`
- Integration testing: `make integ-test`

### Local run

```bash
mkdir tmp2/
HF_MODEL_ID=hf-internal-testing/tiny-random-distilbert HF_MODEL_DIR=tmp2 HF_TASK=text-classification uvicorn src.huggingface_inference_toolkit.webservice_starlette:app  --port 5000
```

### Container

1. build the preferred container for either CPU or GPU for PyTorch.

_CPU Images_

```bash
make inference-pytorch-cpu
```

_GPU Images_

```bash
make inference-pytorch-gpu
```

2. Run the container and provide either environment variables to the HUB model you want to use or mount a volume to the container, where your model is stored.

```bash
docker run -ti -p 5000:5000 -e HF_MODEL_ID=distilbert-base-uncased-distilled-squad -e HF_TASK=question-answering integration-test-pytorch:cpu
docker run -ti -p 5000:5000 --gpus all -e HF_MODEL_ID=nlpconnect/vit-gpt2-image-captioning -e HF_TASK=image-to-text integration-test-pytorch:gpu
docker run -ti -p 5000:5000 --gpus all -e HF_MODEL_ID=echarlaix/tiny-random-stable-diffusion-xl -e HF_TASK=text-to-image integration-test-pytorch:gpu
docker run -ti -p 5000:5000 --gpus all -e HF_MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0 -e HF_TASK=text-to-image integration-test-pytorch:gpu
docker run -ti -p 5000:5000 -e HF_MODEL_DIR=/repository -v $(pwd)/distilbert-base-uncased-emotion:/repository integration-test-pytorch:cpu
```

3. Send request. The API schema is the same as from the [inference API](https://huggingface.co/docs/api-inference/detailed_parameters)

```bash
curl --request POST \
  --url http://localhost:5000 \
  --header 'Content-Type: application/json' \
  --data '{
 "inputs": {
  "question": "What is used for inference?",
  "context": "My Name is Philipp and I live in Nuremberg. This model is used with sagemaker for inference."
 }
}'
```

### Custom Handler and dependency support

The Hugging Face Inference Toolkit allows user to provide a custom inference through a `handler.py` file which is located in the repository.

For an example check [philschmid/custom-pipeline-text-classification](https://huggingface.co/philschmid/custom-pipeline-text-classification):

```bash
model.tar.gz/
|- pytorch_model.bin
|- ....
|- handler.py
|- requirements.txt
```

In this example, `pytroch_model.bin` is the model file saved from training, `handler.py` is the custom inference handler, and `requirements.txt` is a requirements file to add additional dependencies.
The custom module can override the following methods:

### Vertex AI Support

The Hugging Face Inference Toolkit is also supported on Vertex AI, based on [Custom container requirements for prediction](https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements). [Environment variables set by Vertex AI](https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#aip-variables) are automatically detected and used by the toolkit.

#### Local run with HF_MODEL_ID and HF_TASK

Start Hugging Face Inference Toolkit with the following environment variables.

```bash
mkdir tmp2/
AIP_MODE=PREDICTION AIP_PORT=8080 AIP_PREDICT_ROUTE=/pred AIP_HEALTH_ROUTE=/h HF_MODEL_DIR=tmp2 HF_MODEL_ID=distilbert/distilbert-base-uncased-finetuned-sst-2-english HF_TASK=text-classification uvicorn src.huggingface_inference_toolkit.webservice_starlette:app  --port 8080
```

Send request

```bash
curl --request POST \
  --url http://localhost:8080/pred \
  --header 'Content-Type: application/json' \
  --data '{
 "instances": ["I love this product", "I hate this product"],
 "parameters": { "top_k": 2 }
}'
```

#### Container run with HF_MODEL_ID and HF_TASK

1. build the preferred container for either CPU or GPU for PyTorch o.

```bash
docker build -t vertex -f dockerfiles/pytorch/Dockerfile -t vertex-test-pytorch:gpu .
```

2. Run the container and provide either environment variables to the HUB model you want to use or mount a volume to the container, where your model is stored.

```bash
docker run -ti -p 8080:8080 -e AIP_MODE=PREDICTION -e AIP_HTTP_PORT=8080 -e AIP_PREDICT_ROUTE=/pred -e AIP_HEALTH_ROUTE=/h -e HF_MODEL_ID=distilbert/distilbert-base-uncased-finetuned-sst-2-english -e HF_TASK=text-classification vertex-test-pytorch:gpu
```

3. Send request

```bash
curl --request POST \
 --url http://localhost:8080/pred \
 --header 'Content-Type: application/json' \
 --data '{
 "instances": ["I love this product", "I hate this product"],
 "parameters": { "top_k": 2 }
}'
```

### AWS Inferentia2 Support

The Hugging Face Inference Toolkit provides support for deploying Hugging Face on AWS Inferentia2. To deploy a model on Inferentia2 you have 3 options:

- Provide `HF_MODEL_ID`, the model repo id on huggingface.co which contains the compiled model under `.neuron` format e.g. `optimum/bge-base-en-v1.5-neuronx`
- Provide the `HF_OPTIMUM_BATCH_SIZE` and `HF_OPTIMUM_SEQUENCE_LENGTH` environment variables to compile the model on the fly, e.g. `HF_OPTIMUM_BATCH_SIZE=1 HF_OPTIMUM_SEQUENCE_LENGTH=128`
- Include `neuron` dictionary in the [config.json](https://huggingface.co/optimum/tiny_random_bert_neuron/blob/main/config.json) file in the model archive, e.g. `neuron: {"static_batch_size": 1, "static_sequence_length": 128}`

The currently supported tasks can be found [here](https://huggingface.co/docs/optimum-neuron/en/package_reference/supported_models). If you plan to deploy an LLM, we recommend taking a look at [Neuronx TGI](https://huggingface.co/blog/text-generation-inference-on-inferentia2), which is purposly build for LLMs.

#### Local run with HF_MODEL_ID and HF_TASK

Start Hugging Face Inference Toolkit with the following environment variables.

_Note: You need to run this on an Inferentia2 instance._

- transformers `text-classification` with `HF_OPTIMUM_BATCH_SIZE` and `HF_OPTIMUM_SEQUENCE_LENGTH`

```bash
mkdir tmp2/
HF_MODEL_ID="distilbert/distilbert-base-uncased-finetuned-sst-2-english" HF_TASK="text-classification" HF_OPTIMUM_BATCH_SIZE=1 HF_OPTIMUM_SEQUENCE_LENGTH=128  HF_MODEL_DIR=tmp2 uvicorn src.huggingface_inference_toolkit.webservice_starlette:app  --port 5000
```

- sentence transformers `feature-extraction` with `HF_OPTIMUM_BATCH_SIZE` and `HF_OPTIMUM_SEQUENCE_LENGTH`

```bash
HF_MODEL_ID="sentence-transformers/all-MiniLM-L6-v2" HF_TASK="feature-extraction" HF_OPTIMUM_BATCH_SIZE=1 HF_OPTIMUM_SEQUENCE_LENGTH=128 HF_MODEL_DIR=tmp2 uvicorn src.huggingface_inference_toolkit.webservice_starlette:app  --port 5000
```

Send request

```bash
curl --request POST \
 --url http://localhost:5000 \
 --header 'Content-Type: application/json' \
 --data '{
 "inputs": "Wow, this is such a great product. I love it!"
}'
```

#### Container run with HF_MODEL_ID and HF_TASK

1. build the preferred container for either CPU or GPU for PyTorch o.

```bash
make inference-pytorch-inf2
```

2. Run the container and provide either environment variables to the HUB model you want to use or mount a volume to the container, where your model is stored.

```bash
docker run -ti -p 5000:5000 -e HF_MODEL_ID="distilbert/distilbert-base-uncased-finetuned-sst-2-english" -e HF_TASK="text-classification" -e HF_OPTIMUM_BATCH_SIZE=1 -e HF_OPTIMUM_SEQUENCE_LENGTH=128 --device=/dev/neuron0 integration-test-pytorch:inf2
```

3. Send request

```bash
curl --request POST \
 --url http://localhost:5000 \
 --header 'Content-Type: application/json' \
 --data '{
 "inputs": "Wow, this is such a great product. I love it!",
 "parameters": { "top_k": 2 }
}'
```

---

## 🛠️ Environment variables

The Hugging Face Inference Toolkit implements various additional environment variables to simplify your deployment experience. A full list of environment variables is given below. The model and task variables are defined in [const.py](src/huggingface_inference_toolkit/const.py); the serving, overload and routing flags below live next to the code they control, in [env_utils.py](src/huggingface_inference_toolkit/env_utils.py), [idle.py](src/huggingface_inference_toolkit/idle.py), [latency_guard.py](src/huggingface_inference_toolkit/latency_guard.py) and [entrypoint.sh](scripts/entrypoint.sh).

### `HF_MODEL_DIR`

The `HF_MODEL_DIR` environment variable defines the directory where your model is stored or will be stored.
If `HF_MODEL_ID` is not set the toolkit expects a model artifact at this directory. This value should be set to the value where you mount your model artifacts.
If `HF_MODEL_ID` is set the toolkit and the directory where `HF_MODEL_DIR` is pointing to is empty. The toolkit will download the model from the Hub to this directory.

The default value is `/opt/huggingface/model`

```bash
HF_MODEL_ID="/opt/mymodel"
```

### `HF_TASK`

The `HF_TASK` environment variable defines the task for the used Transformers pipeline or Sentence Transformers. A full list of tasks can be found in [supported & tested task section](#supported--tested-tasks)

```bash
HF_TASK="question-answering"
```

### `HF_MODEL_ID`

The `HF_MODEL_ID` environment variable defines the model id, which will be automatically loaded from [huggingface.co/models](https://huggingface.co/models) when starting the container.

```bash
HF_MODEL_ID="distilbert-base-uncased-finetuned-sst-2-english"
```

### `HF_REVISION`

The `HF_REVISION` is an extension to `HF_MODEL_ID` and allows you to define/pin a revision of the model to make sure you always load the same model on your SageMaker Endpoint.

```bash
HF_REVISION="03b4d196c19d0a73c7e0322684e97db1ec397613"
```

### `HF_HUB_TOKEN`

The `HF_HUB_TOKEN` environment variable defines your Hugging Face authorization token. The `HF_HUB_TOKEN` is used as a HTTP bearer authorization for remote files, like private models. You can find your token at your [settings page](https://huggingface.co/settings/token).

```bash
HF_HUB_TOKEN="api_XXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
```

### `HF_TRUST_REMOTE_CODE`

The `HF_TRUST_REMOTE_CODE` environment variable defines whether to trust remote code. This flag is already used for community defined inference code, and is therefore quite representative of the level of confidence you are giving the model providers when loading models from the Hugging Face Hub. The default value is `"0"`; set it to `"1"` to trust remote code.

```bash
HF_TRUST_REMOTE_CODE="0"
```

### `HF_FRAMEWORK`

The `HF_FRAMEWORK` environment variable defines the base deep learning framework used in the container. This is important when loading large models from the Hugging Face Hub to avoid extra file downloads.

```bash
HF_FRAMEWORK="pytorch"
```

#### `HF_OPTIMUM_BATCH_SIZE`

The `HF_OPTIMUM_BATCH_SIZE` environment variable defines the batch size, which is used when compiling the model to Neuron. The default value is `1`. Not required when model is already converted.

```bash
HF_OPTIMUM_BATCH_SIZE="1"
```

#### `HF_OPTIMUM_SEQUENCE_LENGTH`

The `HF_OPTIMUM_SEQUENCE_LENGTH` environment variable defines the sequence length, which is used when compiling the model to Neuron. There is no default value. Not required when model is already converted.

```bash
HF_OPTIMUM_SEQUENCE_LENGTH="128"
```

### Serving and scaling

#### `WORKERS`

The `WORKERS` environment variable defines how many worker processes gunicorn runs. Each worker loads its own copy of the model, so raising this multiplies memory use unless combined with `UNLOAD_IDLE`. The default value is `1`.

```bash
WORKERS="1"
```

#### `UNLOAD_IDLE`

The `UNLOAD_IDLE` environment variable makes a worker load its model on the first request rather than at startup, and terminate itself once it has been idle for `IDLE_TIMEOUT`, releasing the memory and the GPU. gunicorn immediately replaces the worker, and the next request pays for a fresh load. This lets a node host more workers than it has models resident. The default value is `"0"`; set it to `"1"` to enable it.

An idle worker still shows the model files in the container's page cache, which counts towards the reported memory usage but is reclaimable — the resident footprint of a worker with no model loaded is a few tens of MB.

```bash
UNLOAD_IDLE="1"
```

#### `IDLE_TIMEOUT`

The `IDLE_TIMEOUT` environment variable defines how many seconds a worker must be inactive before `UNLOAD_IDLE` terminates it. Requests in flight are counted, so a worker is never terminated in the middle of one, however long it takes. The default value is `15`.

```bash
IDLE_TIMEOUT="15"
```

#### `GRACEFUL_TIMEOUT`

The `GRACEFUL_TIMEOUT` environment variable defines how long, in seconds, gunicorn waits for a worker to finish the request it is serving after being asked to stop, before killing it. Inference longer than this is cut off mid-request when a pod is asked to shut down, so it should exceed your slowest expected inference. The default value is `300`, against gunicorn's own default of 30.

```bash
GRACEFUL_TIMEOUT="300"
```

#### `TIMEOUT`

The `TIMEOUT` environment variable defines gunicorn's worker timeout in seconds. The default value is `30`.

```bash
TIMEOUT="30"
```

#### `DISCARD_LEFT`

The `DISCARD_LEFT` environment variable makes the toolkit skip a queued request whose caller has already disconnected, answering `204` instead of spending a model call on a response nobody will read. Useful when clients time out and retry. The default value is `"0"`; set it to `"1"` to enable it.

```bash
DISCARD_LEFT="1"
```

#### `LOG_LEVEL`

The `LOG_LEVEL` environment variable defines the logging level, as a Python logging level name. The default value is `INFO`.

```bash
LOG_LEVEL="DEBUG"
```

### Overload protection

#### `LATENCY_GUARD_ENABLED`

The `LATENCY_GUARD_ENABLED` environment variable enables load shedding based on inference latency. The toolkit tracks a fast and a slow exponential moving average of pure inference time and answers `503` while the fast one sits far above the slow one, on the grounds that refusing quickly beats queueing work whose callers will time out. There is no absolute threshold to configure: the baseline is learned from the worker's own history. Two gauges are exposed on `/metrics`, `inf_accepting` and `inf_auto_frozen`. The default value is `"0"`; set it to `"1"` to enable it.

```bash
LATENCY_GUARD_ENABLED="1"
```

The tuning variables below only apply when the guard is enabled, and the defaults are a reasonable starting point:

| Variable | Default | Meaning |
| --- | --- | --- |
| `LATENCY_WARMUP_REQUESTS` | `10` | Inferences observed before the guard may fire |
| `LATENCY_FAST_ALPHA` | `0.3` | EMA weight for the recent window (~3-5 requests) |
| `LATENCY_SLOW_ALPHA` | `0.05` | EMA weight for the baseline window (~20 requests) |
| `LATENCY_OVERLOAD_FACTOR` | `3.0` | Freeze when recent latency exceeds this multiple of the baseline |
| `LATENCY_RECOVERY_FACTOR` | `1.5` | Unfreeze below this multiple (hysteresis) |
| `LATENCY_FREEZE_SECONDS` | `10` | How long to refuse before letting requests through to re-measure |

`LATENCY_FREEZE_SECONDS` matters more than it looks: a frozen worker runs no inference, so it records no latency and would never recover on its own. After this delay the guard accepts requests again to get fresh samples, then either unfreezes or freezes for another round.

### Multi-task serving

#### `ENABLE_TASK_ROUTE`

The `ENABLE_TASK_ROUTE` environment variable exposes `POST /pipeline/{task}`, letting a caller name the pipeline to serve a request with instead of only the task the endpoint was deployed for. The route is off unless you set this, and a deployment that does not set it answers for `HF_TASK` alone.

> [!WARNING]
> Enable this only for callers you trust. The task comes from the request path, and every task asked for builds a **separate pipeline with its own copy of the model weights**, kept until the worker exits. Nothing bounds that cache, so a caller can walk through the tasks a model happens to support and exhaust the worker's memory or the GPU, taking the service down for everyone using it. On a small model with authenticated callers this is a useful capability; on a public endpoint serving a large model it is a way to break the deployment with a handful of requests.

```bash
ENABLE_TASK_ROUTE="1"
```

### Task and handler defaults

#### `IGNORE_CUSTOM_HANDLER`

The `IGNORE_CUSTOM_HANDLER` environment variable makes the toolkit serve a model with the default pipeline even when its repository ships a `handler.py`. The file is not merely discarded but never imported, since importing it runs whatever is at its top level. The default value is `"false"`; set it to `"true"` to enable it.

```bash
IGNORE_CUSTOM_HANDLER="true"
```

#### `DEFAULT_NUM_INFERENCE_STEPS` and `DEFAULT_GUIDANCE_SCALE`

These two environment variables define deployment-wide defaults for the `text-to-image` parameters that decide how long a generation takes, for setups that need a predictable cost per request. They are applied only when the request does not carry the parameter itself, so an explicit request parameter always wins. Neither has a default: left unset, the pipeline's own values apply. They are parsed at startup, so a value that is not a number fails the worker immediately rather than every generation.

```bash
DEFAULT_NUM_INFERENCE_STEPS="25"
DEFAULT_GUIDANCE_SCALE="7.5"
```

---

## ⚙ Supported Front-Ends

- [x] Starlette (HF Endpoints)
- [x] Starlette (Vertex AI)
- [ ] Starlette (Azure ML)
- [ ] Starlette (SageMaker)

## 📜 License

This project is licensed under the Apache-2.0 License.

