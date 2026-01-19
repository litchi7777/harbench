# HARBench Artifact Docker Image
# For paper reproducibility
# Use CUDA 12.8 for Blackwell (sm_120) GPU support
FROM nvidia/cuda:12.8.0-base-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y \
        curl \
        python3 \
        python3-distutils \
        python3-dev \
        git \
        build-essential \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py \
    && ln -sf $(which python3) /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt /workspace/

# Install Python dependencies
# Use nightly build for Blackwell (sm_120) GPU support
RUN pip install --upgrade pip && \
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 && \
    pip install -r requirements.txt

# Copy the artifact source code
COPY . /workspace/

# Set environment variables
ENV PYTHONPATH=/workspace:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Default command: show help
CMD ["python", "finetune.py", "--help"]
