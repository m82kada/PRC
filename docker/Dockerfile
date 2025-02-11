FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04
ARG user_id=0
ARG group_id=0
RUN echo "user_id: $user_id, group_id: $group_id"
RUN useradd --shell /bin/bash -u $user_id -o -c "" -m user && groupmod -g $group_id -o user

ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i.bak -e "s%http://[^ ]\+%http://ftp.jaist.ac.jp/pub/Linux/ubuntu/%g" /etc/apt/sources.list \
  && apt-get update && apt-get -y upgrade \
  && apt-get install -y python3 python3-pip software-properties-common vim zsh git golang-go graphviz \
  && apt-get clean && rm -rf /var/lib/apt/lists/*
#RUN unlink /usr/bin/python3 && ln -s /usr/bin/python3.10 /usr/bin/python3

USER user
WORKDIR /home/user

RUN python3 -m pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip uninstall clu -y
RUN python3 -m pip install git+https://github.com/google/CommonLoopUtils.git

# protobuf
RUN python3 -m pip install protobuf
RUN cp /home/user/.local/lib/python3.10/site-packages/google/protobuf/internal/builder.py ~/
RUN python3 -m pip install protobuf~=3.19.0 numpy==1.23.4
RUN mv ~/builder.py /home/user/.local/lib/python3.10/site-packages/google/protobuf/internal/builder.py
#RUN python3 -m pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.1+cuda11.cudnn82-cp38-cp38-manylinux2014_x86_64.whl
RUN python3 -m pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN python3 -m pip install git+https://github.com/deepmind/optax.git

RUN go install github.com/google/pprof@latest

EXPOSE 7777

RUN rm requirements.txt 
COPY --chown=user .zshrc .zshrc 
COPY --chown=user .vimrc .vimrc 
COPY --chown=user .jupyter .jupyter 

ENV PYTHON_PATH="/home/user/app/cloud_tpu"

CMD ["zsh"]
