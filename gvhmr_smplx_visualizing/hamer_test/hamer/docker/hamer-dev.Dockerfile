ARG BASE=nvidia/cuda:12.6.2-devel-ubuntu22.04
FROM ${BASE} AS hamer

# Install OS dependencies:
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends --fix-missing \
    gcc g++ \
    make \
    python3 python3-dev python3-pip python3-venv python3-wheel \
    espeak-ng libsndfile1-dev \
    git \
    wget \
    ffmpeg \
    libsm6 libxext6 \
    libglfw3-dev libgles2-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# Install hamer:
WORKDIR /app

# Create virtual environment:
RUN python3 -m venv /opt/venv

# Add virtual environment to PATH
ENV PATH="/opt/venv/bin:$PATH"

# Activate virtual environment and install dependencies:
# REVIEW: We need to install/upgrade wheel and setuptools first because otherwise installation fails:
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade wheel setuptools

# Install torch and torchvision:
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118

# REVIEW: Numpy is installed separately because otherwise installation fails:
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install numpy

# Install gdown (used for fetching scripts):
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install gdown

# Install third-party dependencies ViTPose:
COPY third-party/ third-party/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -v -e third-party/ViTPose

# Install project dependencies:
COPY . .
# Install hamer:
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e .[all]
