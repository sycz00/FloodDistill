# Training, evaluation, QAT and ONNX export image.
# Build from the repository root: docker build -f docker/Dockerfile -t flooddistill .
# TensorRT engines are built on the Jetson; see jetson_repro/README.md.

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl unzip ca-certificates libgl1 libglib2.0-0 libgdal-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# gsutil, for the Sen1Floods11 public bucket (datasets/prepare_sen1floods11.sh).
RUN pip3 install --no-cache-dir gsutil

RUN wget -qO /tmp/mc.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/mc.sh -b -p /opt/conda && rm /tmp/mc.sh
ENV PATH=/opt/conda/bin:$PATH

WORKDIR /workspace/FloodDistill

# Conda environment first for layer caching. TerraTorch is needed by data/sen1floods11.py.
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

# EfficientViT is not on PyPI. Its install downgrades numpy, so numpy is restored afterwards.
RUN git clone --depth 1 https://github.com/mit-han-lab/efficientvit /opt/efficientvit \
    && conda run -n esi pip install --no-cache-dir -e /opt/efficientvit \
    && conda run -n esi pip install --no-cache-dir "numpy==2.2.6"

COPY . .

# EfficientViT-B0 segmentation weights, loaded relative to WORKDIR; downloaded at build time.
RUN bash datasets/download_efficientvit_ckpts.sh

RUN cp docker/docker-entrypoint.sh /usr/local/bin/ && chmod +x /usr/local/bin/docker-entrypoint.sh

ENV PYTHONPATH=/workspace/FloodDistill:/opt/efficientvit
# All datasets are mounted under /data; every script derives its paths from DATA_ROOT
# (utils/paths.py). The default ~/data does not apply: $HOME is /tmp inside the container.
ENV DATA_ROOT=/data
ENV PSEUDO_DIR=/pseudo
ENV WANDB_MODE=disabled
# Mount a host Hugging Face cache here; world-writable for --user <uid>.
ENV HF_HOME=/hf
RUN mkdir -p /hf && chmod 777 /hf

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["--help"]
