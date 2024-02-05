FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
RUN apt update && apt install python3-pip -y
RUN pip3 install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN apt install git -y
RUN git clone https://github.com/Velythyl/brax.git -b vsys && cd brax && pip3 install -e .

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN pip3 install "mujoco<3.0" # wtf

WORKDIR /mbrma

COPY . /mbrma/
