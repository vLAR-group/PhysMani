# 3d_diffuser_actor follows feature/mom_baseline:docker/.
# lerobot follows master:docker/setup.sh.
FROM nvidia/cuda:11.6.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libavcodec-dev \
    libavformat-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libswscale-dev \
    ninja-build \
    qt5-default \
    wget \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet \
      https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
      -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH
RUN conda init bash
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

WORKDIR /workspace

COPY docker/environment.yaml /workspace/docker/environment.yaml
RUN conda update -y conda \
    && conda env create -f /workspace/docker/environment.yaml

SHELL ["conda", "run", "--no-capture-output", "-n", "3d_diffuser_actor", "/bin/bash", "-c"]

# Docker build cannot infer GPU arch because no GPU is visible during extension builds.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"

RUN pip install diffusers==0.34.0 msgpack h5py
RUN conda install -y -c conda-forge glm \
    && conda install -y -c nvidia \
      libcusparse-dev \
      libcublas-dev \
      libcusolver-dev \
      libcurand-dev \
    && conda clean -afy
RUN pip install dgl==1.1.3+cu116 \
      -f https://data.dgl.ai/wheels/cu116/dgl-1.1.3%2Bcu116-cp38-cp38-manylinux1_x86_64.whl
RUN pip install packaging ninja
RUN pip install mkl==2024.0

RUN wget \
      https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz \
    && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz \
    && rm CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

ENV COPPELIASIM_ROOT=/workspace/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
ENV QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

RUN pip install open3d
RUN pip install termcolor imageio plyfile wandb 'wandb[media]'

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY docker/lerobot_environment.yaml /workspace/docker/lerobot_environment.yaml
RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda env create -n lerobot -f /workspace/docker/lerobot_environment.yaml
RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda activate lerobot \
    && pip install --no-cache-dir \
      torch==2.7.1+cu126 \
      torchvision \
      torchaudio \
      --index-url https://download.pytorch.org/whl/cu126
RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda activate lerobot \
    && conda install -y -c nvidia \
      cuda-nvcc=12.6 \
      cuda-cudart-dev=12.6 \
      libcusparse-dev \
      libcublas-dev \
      libcusolver-dev \
      libcurand-dev \
      --no-channel-priority
RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda activate lerobot \
    && conda install -y -c conda-forge glm
RUN source /opt/conda/etc/profile.d/conda.sh \
    && conda activate lerobot \
    && pip install fire open3d websockets msgpack lion_pytorch diffusers==0.34.0
RUN conda clean -afy

WORKDIR /workspace
CMD ["/bin/bash"]
