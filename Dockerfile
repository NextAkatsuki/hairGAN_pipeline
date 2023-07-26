FROM tensorflow/tensorflow:2.4.0
WORKDIR /app
COPY . /app

ENV TF_GPU_DEVICE_NAME=

RUN apt-get update && \
    apt-get -y install libgl1-mesa-glx && \
    python -m pip install --upgrade pip

RUN pip install -r requirements.txt

# RUN mkdir image && \
#     cd image && \
#     mkdir rawhair && \
#     mkdir seghair

EXPOSE 8000
CMD uvicorn api.api:app --host=0.0.0.0 --port 8000

