# GPU or CPU
ARG env_type=gpu

FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS build-gpu

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ai

COPY requirements.txt .

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt --index-url https://download.pytorch.org/whl/cu126

COPY . ./ai

EXPOSE 80000
CMD ["python3", "main.py"]

FROM python:3.11-slim AS build-cpu

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ai

COPY requirements.txt .

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt --index-url https://download.pytorch.org/whl/cu126

COPY . ./ai

EXPOSE 80000
CMD ["python3", "main.py"]

FROM stage-${BUILD_TYPE} AS final
WORKDIR /ai/