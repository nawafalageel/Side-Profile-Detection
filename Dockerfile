FROM python:3.7-slim

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

RUN pip install jupyter

EXPOSE 8888

CMD jupyter notebook --no-browser --ip=0.0.0.0 --port 8888 --allow-root --NotebookApp.token='' --NotebookApp.allow_origin='*'

