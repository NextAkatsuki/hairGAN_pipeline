FROM tensorflow/tensorflow:2.4.0
WORKDIR /app
COPY . /app

RUN apt-get update && \
    apt-get -y install libgl1-mesa-glx && \
    apt-get install unzip && \
    python -m pip install --upgrade pip

RUN pip install -r requirements.txt

WORKDIR /app/volume
RUN wget https://github.com/NextAkatsuki/hairGAN_pipeline/releases/download/v0.1/output.zip && \
    unzip output.zip && \
    rm -rf output.zip

WORKDIR /app


EXPOSE 8000
CMD uvicorn api.api:app --host=0.0.0.0 --port 8000

