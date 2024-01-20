FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /usr/src/app

# dont write pyc files
# dont buffer to stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY ./requirements.txt /usr/src/app/requirements.txt

# dependencies
RUN apt-get update && apt-get install -y python3-pip 
RUN pip install --upgrade pip setuptools wheel

# Instalar otras dependencias
RUN pip install -q -U transformers peft accelerate optimum \
    && pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu117/ \
    && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    && pip install -r requirements.txt \
    && pip install --upgrade bitsandbytes \
    && rm -rf /root/.cache/pip

COPY ./ /usr/src/app
