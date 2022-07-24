FROM ubuntu:20.04 as base_build
SHELL [ "/bin/bash" , "-c" ]

ENV DEBIAN_FRONTEND noninteractive
ENV PYTHON_VERSION="3.8"

ARG ML_ARCHITECTURE_VERSION_GIT_BRANCH=master
ARG ML_ARCHITECTURE_VERSION_GIT_COMMIT=HEAD

LABEL maintainer=ronaldsonbellande@gmail.com
LABEL ml_architecture_github_branchtag=${ML_ARCHITECTURE_VERSION_GIT_BRANCH}
LABEL ml_architecture_github_commit=${ML_ARCHITECTURE_VERSION_GIT_COMMIT}

# Ubuntu setup
RUN apt-get update -y
RUN apt-get upgrade -y

# RUN workspace and sourcing
WORKDIR ./
COPY requirements.txt .
COPY system_requirements.txt .

# Install dependencies for system
RUN apt-get update && apt-get install -y --no-install-recommends <system_requirements.txt && \
  apt-get upgrade -y && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Install python 3.8 and make primary 
RUN apt-get update && apt-get install -y \
  python3.8 python3.8-dev python3-pip python3.8-venv && \
  update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# Pip install update 
RUN pip3 install --upgrade pip

# Install python libraries
RUN pip --no-cache-dir install -r requirements.txt
