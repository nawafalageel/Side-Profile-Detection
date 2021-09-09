FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

COPY . /app

RUN pip install -r requirements.txt

WORKDIR /app

ENV readingimage default_reading
ENV image default_image

CMD [ "python",  " DetectFacePose.py ${readingimage} ${image}"]

