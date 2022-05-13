FROM public.ecr.aws/lambda/python:3.8

COPY cat_dog_classifier.tflite .

COPY lambda_model.py .

RUN pip3 install numpy

RUN pip3 install requests

RUN pip3 install pillow

RUN pip3 install https://github.com/alexeygrigorev/tflite-aws-lambda/blob/main/tflite/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl?raw=true

CMD ["lambda_model.lambda_handler"]