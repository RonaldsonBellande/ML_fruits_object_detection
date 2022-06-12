FROM ubuntu:20.04 as base_build
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

 # Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        automake \
        build-essential \
        ca-certificates \
        curl \
        git \
        python3-pip \
        libcurl3-dev \
        libfreetype6-dev \
        libpng-dev \
        libtool \
        libzmq3-dev \
        mlocate \
        openjdk-8-jdk\
        openjdk-8-jre-headless \
        pkg-config \
        python-dev \
        software-properties-common \
        swig \
        unzip \
        wget \
        zip \
        zlib1g-dev \
        python3-distutils \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Install python 3.8 and make primary 
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.8 python3.8-dev python3-pip python3.8-venv && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# pip install 
RUN pip3 install --upgrade pip

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py


# Install python libraries
RUN pip --no-cache-dir install -r requirements.txt

