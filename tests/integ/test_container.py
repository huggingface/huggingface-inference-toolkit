import json
import os
import re

import numpy as np
import pytest
from integ.config import task2input, task2model, task2output, task2validation


@pytest.mark.parametrize(
    "task",
    [
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
    ],
)
@pytest.mark.parametrize(
    "framework",
    ["pytorch"],
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
    ],
)
def test_deployment(task, device, framework):
    # image_uri = get_container(framework=framework, device=device)
    model = task2model[task][framework]

    # load model from hub and mount it into docker
    # whith tempdir
    # docker run
    
    # check health route of docker
    
    # send request and validate response 
    input = task2input[task]
    # pred = r.post
    pred=None
    
    assert task2validation[task](result=pred, snapshot=task2output[task]) == True


    env = {"HF_MODEL_ID": model, "HF_TASK": task}

