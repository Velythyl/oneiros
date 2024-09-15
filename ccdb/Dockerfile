FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install python3.9 -y && apt install python3-pip -y
RUN python3.9 -m pip install --upgrade "jax[cuda11_pip]==0.4.16" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN python3.9 -m pip install flax==0.7.4
RUN python3.9 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN apt update && apt install git -y
RUN echo reset here 2
RUN git clone https://github.com/Velythyl/brax.git -b vsys && cd brax && python3.9 -m pip install -e .

COPY requirements.txt requirements.txt
RUN python3.9 -m pip install -r requirements.txt

RUN apt install python3.9-venv -y
RUN python3.9 -m venv  venv --system-site-package
ENV PATH=/venv/bin:$PATH

WORKDIR /workdir
