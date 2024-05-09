FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

RUN apt-get update -y
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y && apt update -y
RUN apt install python3.10 python3.10-venv python3.10-dev python3.10-distutils -y
RUN apt install curl -y

RUN curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.10 get-pip.py && rm get-pip.py

RUN python3.10 -m pip install tqdm numpy flask sentence-transformers
RUN python3.10 -m pip install https://github.com/kyamagu/faiss-wheels/releases/download/v1.7.3/faiss_gpu-1.7.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
