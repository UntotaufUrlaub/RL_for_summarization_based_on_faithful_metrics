FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

# setup python3.8
RUN apt-get update && \
    apt-get install -y software-properties-common curl wget unzip && \
    apt-get clean
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && \
    apt-get install -y python3.8 python3.8-distutils python3.8-venv && \
    apt-get clean
RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python3.8 /tmp/get-pip.py && \
    rm /tmp/get-pip.py
RUN apt-get update && \
    apt-get install -y build-essential libffi-dev python3.8-dev && \
    apt-get clean
RUN python3.8 -m pip install --upgrade pip setuptools wheel
# RUN python3.8 -m pip install jsonnet

# setup the project
RUN apt-get update && apt-get install git -y --no-install-recommends && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/salesforce/QAFactEval
WORKDIR QAFactEval
RUN python3.8 -m pip install -e .
RUN python3.8 -m pip install gdown
RUN ./download_models.sh

COPY qafacteval_minimal.py qafacteval_minimal.py

# RUN python3.8 -m pip install torch==1.6.0 torchvision==0.7.0 torchaudio==0.6.0 -f https://download.pytorch.org/whl/cu101/torch_stable.html

CMD python3.8 qafacteval_minimal.py

