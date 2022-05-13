# Deploying Deep Learning Models in Lambda AWS

In this short tutorial we will deploy a Deep Learning model developed in Keras into Lambda AWS. Our assumtion is that you already have a model trained and now you want to use it.

## Transform saved Keras model into Tensorflow Lite

We first load a `h5` model using Keras

```python
from tensorflow.keras.models import load_model

model = load_model('deep_learning_model.h5')
```

Then we need to transform this model into a Tensorflow Lite model

```python
import tensorflow.lite as tflite

converter = tflite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('deep_learning_model.tflite', 'wb') as file:
    file.write(tflite_model)
```

A Tensorflow Lite model named `deep_learning_model.tflite` is created.

## Change library dependencies

Usually when we deploy a model using Keras we relly on some libraries as helpers for processing our data, since we are using Tensorflow Lite we might need to replace these  libraries with our own custom implementations.

For example a function that was used to preprocess our image for inference would look like

```python
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Generator object to rescale image.
datagen = ImageDataGenerator(rescale=1./255)

# Load image
img = load_img('image.jpeg', target_size=(150,150))

# Image to array format expected by CNN
X = np.array([np.array(img)])
# ImageDataGenerator method to rescale image
X = datagen.flow(X).next()
```

And our custom implementation would look like

```python
import numpy as np
from PIL import Image

def preprocess(img, scale=1./255):
    return np.float32(img*scale)

with Image.open('image.jpeg') as img:
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((150, 150), Image.NEAREST)

X = preprocess(np.array([np.array(img)]))
```

Which relies in a more lightweight library such as `pillow`.

## Use model for prediction

Before we could make a prediction in the following way

```python
pred = model.predict(X)
```

But now it is a little more difficult than that, because we need to specify our data flow using inputs and output pointers, when to load our weghts and when to perform our inference. Below is the implementation for the prediction of our Tensorflow Lite model.

```python
import tflite_runtime.interpreter as tflite

# Load model to interpreter
interpreter = tflite.Interpreter(model_path='deep_learning_model.tflite')
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
```

## AWS lambda_handler

Lambda AWS uses a wraper function called `lambda_handler` to perform calculations,  the complete implementation is stored in `lambda_model.py` but the important part here is how we pass data.

```python
 def lambda_handler(event, context):
    url = event['url']
    # Implementation of our model
    return prediction
```

Wich in this case the `url` is the location of the image itself. Note that our prediction must be returned in a JSON format.

## Docker container

We will create a Docker container with all dependencies needed for us to use AWS Lambda.

```dockerfile
FROM public.ecr.aws/lambda/python:3.8

COPY cat_dog_classifier.tflite .
COPY lambda_model.py .

RUN pip3 install numpy
RUN pip3 install requests
RUN pip3 install pillow
RUN pip3 install https://github.com/alexeygrigorev/tflite-aws-lambda/blob/main/tflite/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl?raw=true

CMD ["lambda_model.lambda_handler"]
```

Type `docker build -t model_name .` to build the container.

## Run Docker locally

```bash
docker run -it --rm -p 8080:8080 model_name:latest
```

## Test the container locally

To test the cointaner locally we will send a `data_url` from an image to the localhost `url` that AWS uses

```python
import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

data_url = 'https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg'

result = requests.post(url, json={'url': data_url}).json()
print(result)
```

If the results are correct we expect to have a prediction of our model. The test file is stored in `test.py`.

## Create AWS ECR respository

```bash
aws ecr create-repository --repository-name model-repo
```

## Login to Docker account

```bash
$(aws ecr get-login --no-include-email)
```

## Tag Docker container

For this you will use the `respositoryUri` generated when you created your ECR repository.

```bash
ACCOUNT=*************
REGION=us-east-1
REGISTRY=model-repo
TAG=latest

PREFIX=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REGISTRY

REMOTE_URI=$PREFIX:$TAG

docker tag model_name:latest $REMOTE_URI
```

## Push container

```bash
docker push $REMOTE_URI
```

## Create AWS Lambda function

![Alt text](images/lambda.png "a title")

Be sure to edit it's preferences and add more memory and time for your Lambda function to execute.

## Create API Gateway

Go to AWS Console and create a REST API with a resource named `predict`. Be sure to select CORS.

![Alt text](images/api-gateway-resources.png "a title")



Make a POST method with your previously created lambda function

![Alt text](images/api-gateway-post.png "a title")



Deploy your API and give it a stage name, i.e. `production`.



You will get an IP  like this

```bash
https://*********.execute-api.us-east-1.amazonaws.com/production/predict
```

where you can make your API calls .



## Add an API Key

To protect you deployment from undisired requests, you can add an API Key to your service. To do so you first need to add an **Api Key Required** parameter to your POST method.

![Alt text](images/api-key.png "a title")

Then go to API Keys on the left panel and on Actions select Create API key and give a name to your API key.

![Alt text](images/api-key-create.png "a title")



Now go to Usage Plans on the left panel and create a usage plan. Set a Throttling and Quota that you prefer.

![Alt text](images/usage_plan.png "a title")

And then associate your Usage Plan with your API Key.

Deploy your API again. (Maybe you will need to wait a minute in order for your changes to take effect.)



## Test your deployment

To test your deployment you need to modify the `test.py` script and a header with the API Key as follows

```python
headers = {
  'X-API-KEY': api_key,
  'Content-Type': 'application/json'
}

result = requests.post(url, json={'url': data}, headers=headers).json()
```

The modified script is saved as `test_with_key.py`

**It is a good idea to store your API Key separately in another file to avoid malicious usage.**
