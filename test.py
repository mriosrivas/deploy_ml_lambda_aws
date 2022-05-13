import requests

# Local deployment
#url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

# AWS deployment
url = 'https://**********.execute-api.us-east-1.amazonaws.com/prod/predict'

# Dog image
#data = 'https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg'

# Cat image
data = 'https://www.nj.com/resizer/mg42jsVYwvbHKUUFQzpw6gyKmBg=/1280x0/smart/advancelocal-adapter-image-uploads.s3.amazonaws.com/image.nj.com/home/njo-media/width2048/img/somerset_impact/photo/sm0212petjpg-7a377c1c93f64d37.jpg'

result = requests.post(url, json={'url': data}).json()
print(result)
