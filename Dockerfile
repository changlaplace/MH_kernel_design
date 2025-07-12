FROM tensorflow/tensorflow:2.13.0-gpu

RUN pip install --upgrade pip \
 && pip install tqdm scipy matplotlib notebook jupyterlab ipykernel

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
