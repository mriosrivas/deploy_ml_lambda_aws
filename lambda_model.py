#!/usr/bin/env python
# coding: utf-8


import numpy as np
import requests
from PIL import Image

import tflite_runtime.interpreter as tflite


def get_from_url(url):
    r = requests.get(url)
    path = '/tmp/image'

    with open(path, 'wb') as f:
        f.write(r.content) 


def preprocess(img, scale=1./255):
    return np.float32(img*scale)


#url = 'https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg'
#url = 'https://www.nj.com/resizer/mg42jsVYwvbHKUUFQzpw6gyKmBg=/1280x0/smart/advancelocal-adapter-image-uploads.s3.amazonaws.com/image.nj.com/home/njo-media/width2048/img/somerset_impact/photo/sm0212petjpg-7a377c1c93f64d37.jpg'

def lambda_handler(event, context):
    classes = ['cat', 'dog']
    url = event['url']

    get_from_url(url)

    with Image.open('/tmp/image') as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((150, 150), Image.NEAREST)

    X = preprocess(np.array([np.array(img)]))

    # Load model to interpreter
    interpreter = tflite.Interpreter(model_path='cat_dog_classifier.tflite')
    # Load weights
    interpreter.allocate_tensors()

    # Find input pointer
    input_index = interpreter.get_input_details()[0]['index']

    # Find output pointer
    output_index = interpreter.get_output_details()[0]['index']

    # Load data into input
    interpreter.set_tensor(input_index, X)

    # Run inference
    interpreter.invoke()

    # Load output into memory
    lite_pred = interpreter.get_tensor(output_index)

    return dict(zip(classes, (np.array([1-lite_pred[0][0], lite_pred[0][0]])).tolist()))





