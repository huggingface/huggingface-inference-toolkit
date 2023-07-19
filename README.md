
<div style="display:flex; text-align:center; justify-content:center;">
<img src="https://huggingface.co/front/assets/huggingface_logo.svg" width="100"/> 
<h1 style="margin-top:auto;"> Hugging Face Inference Toolkit <h1>
</div>


Hugging Face Inference Toolkit is for serving 🤗 Transformers models in containers. This library provides default pre-processing, predict and postprocessing for Transformers, Sentence Tranfsformers. It is also possible to define custom `handler.py` for customization. The Toolkit is build to work with the [Hugging Face Hub](https://huggingface.co/models).

---
## 💻  Getting Started with Hugging Face Inference Toolkit

### Local run

```bash
mkdir tmp2/
HF_MODEL_ID=hf-internal-testing/tiny-random-distilbert HF_MODEL_DIR=tmp2 HF_TASK=text-classification uvicorn src.huggingface_inference_toolkit.webservice_starlette:app  --port 5000
```

### Container


1. build the preferred container for either CPU or GPU for PyTorch or TensorFlow.

_cpu images_
```bash
docker build -t starlette-transformers:cpu -f dockerfiles/pytorch/cpu/Dockerfile .
docker build -t starlette-transformers:cpu -f dockerfiles/tensorflow/cpu/Dockerfile .
```

_gpu images_
```bash
docker build -t starlette-transformers:gpu -f dockerfiles/pytorch/gpu/Dockerfile .
docker build -t starlette-transformers:gpu -f dockerfiles/tensorflow/gpu/Dockerfile .
```

2. Run the container and provide either environment variables to the HUB model you want to use or mount a volume to the container, where your model is stored.


```bash
docker run -ti -p 5000:5000 -e HF_MODEL_ID=distilbert-base-uncased-distilled-squad -e HF_TASK=question-answering starlette-transformers:cpu
docker run -ti -p 5000:5000 --gpus all -e HF_MODEL_ID=nlpconnect/vit-gpt2-image-captioning -e HF_TASK=image-to-text starlette-transformers:gpu
docker run -ti -p 5000:5000 -e HF_MODEL_DIR=/repository -v $(pwd)/distilbert-base-uncased-emotion:/repository starlette-transformers:cpu
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


---

## 🛠️ Environment variables

The Hugging Face Inference Toolkit implements various additional environment variables to simplify your deployment experience. A full list of environment variables is given below. All potential environment varialbes can be found in [const.py](src/huggingface_inference_toolkit/const.py)

### `HF_MODEL_DIR`

The `HF_MODEL_DIR` environment variable defines the directory where your model is stored or will be stored. 
If `HF_MODEL_ID` is not set the toolkit expects a the model artifact at this directory. This value should be set to the value where you mount your model artifacts. 
If `HF_MODEL_ID` is set the toolkit and the directory where `HF_MODEL_DIR` is pointing to is empty. The toolkit will download the model from the Hub to this directory. 

The default value is `/opt/huggingface/model`

```bash
HF_MODEL_ID="/opt/mymodel"
```

### `HF_TASK`

The `HF_TASK` environment variable defines the task for the used Transformers pipeline or Sentence Transformers. A full list of tasks can be find in [supported & tested task section](#supported--tested-tasks)

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

The `HF_HUB_TOKEN` environment variable defines the your Hugging Face authorization token. The `HF_HUB_TOKEN` is used as a HTTP bearer authorization for remote files, like private models. You can find your token at your [settings page](https://huggingface.co/settings/token).

```bash
HF_HUB_TOKEN="api_XXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
```

### `HF_FRAMEWORK`

The `HF_FRAMEWORK` environment variable defines the base deep learning framework used in the container. This is important when loading large models from the Hugguing Face Hub to avoid extra file downloads.

```bash
HF_FRAMEWORK="pytorch"
```

### `HF_ENDPOINT`

The `HF_ENDPOINT` environment variable indicates whether the service is run inside the HF Inference endpoint service to adjust the `logging` config.

```bash
HF_ENDPOINT="True"
```


---

## 🧑🏻‍💻 Custom Handler and dependency support

The Hugging Face Inference Toolkit allows user to provide a custom inference through a `handler.py` file which is located in the repository. 
For an example check [https://huggingface.co/philschmid/custom-pipeline-text-classification](https://huggingface.co/philschmid/custom-pipeline-text-classification):  
```bash
model.tar.gz/
|- pytorch_model.bin
|- ....
|- handler.py
|- requirements.txt 
```
In this example, `pytroch_model.bin` is the model file saved from training, `handler.py` is the custom inference handler, and `requirements.txt` is a requirements file to add additional dependencies.
The custom module can override the following methods:  


## ☑️ Supported & Tested Tasks

Below you ll find a list of supported and tested transformers and sentence transformers tasks. Each of those are always tested through integration tests. In addition to those tasks you can always provide `custom`, which expect a `handler.py` file to be provided.

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

##  ⚙ Supported Frontend

- [x] Starlette (HF Endpoints)
- [ ] Starlette (Azure ML)
- [ ] Starlette (SageMaker)

---
## 🤝 Contributing

TBD. 

---
## 📜  License

TBD. 

---
