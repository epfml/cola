FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    make \
    libc-dev \
    musl-dev \
    openssh-server \
    g++ \
    git \
    curl \
    sudo

# -----–––---------------------- Cuda Dependency --------------------
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get install -y --no-install-recommends --allow-downgrades \
        --allow-change-held-packages \
         libnccl2=2.0.5-3+cuda9.0 \
         libnccl-dev=2.0.5-3+cuda9.0 &&\
     rm -rf /var/lib/apt/lists/*

# -------------------- Conda environment --------------------
RUN curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     sh ~/miniconda.sh -b -p /conda && rm ~/miniconda.sh
ENV PATH /conda/bin:$PATH
ENV LD_LIBRARY_PATH /conda/lib:$LD_LIBRARY_PATH

# TODO: Source code in Channel Anaconda can be outdated, switch to conda-forge if posible.
RUN conda install -y -c anaconda numpy pyyaml scipy mkl setuptools cmake cffi mkl-include typing \
    && conda install -y -c mingfeima mkldnn \
    && conda install -y -c soumith magma-cuda90 \
    && conda install -y -c conda-forge python-lmdb opencv numpy \
    && conda clean --all -y

# -------------------- Open MPI --------------------
RUN mkdir /.openmpi/
RUN apt-get update && apt-get install -y --no-install-recommends wget \
    && wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz\
    && gunzip -c openmpi-3.0.0.tar.gz | tar xf - \
    && cd openmpi-3.0.0 \
    && ./configure --prefix=/.openmpi/ --with-cuda\
    && make all install \
    && rm /openmpi-3.0.0.tar.gz \
    && rm -rf /openmpi-3.0.0 \
    && apt-get remove -y wget

ENV PATH /.openmpi/bin:$PATH
ENV LD_LIBRARY_PATH /.openmpi/lib:$LD_LIBRARY_PATH

RUN mv /.openmpi/bin/mpirun /.openmpi/bin/mpirun.real && \
    echo '#!/bin/bash' > /.openmpi/bin/mpirun && \
    echo "/.openmpi/bin/mpirun.real" '--allow-run-as-root "$@"' >> /.openmpi/bin/mpirun && \
    chmod a+x /.openmpi/bin/mpirun

# Configure OpenMPI to run good defaults:
#   --bind-to none --map-by slot --mca btl_tcp_if_exclude lo,docker0
RUN echo "hwloc_base_binding_policy = none" >> /.openmpi/etc/openmpi-mca-params.conf && \
    echo "rmaps_base_mapping_policy = slot" >> /.openmpi/etc/openmpi-mca-params.conf && \
    echo "btl_tcp_if_exclude = lo,docker0" >> /.openmpi/etc/openmpi-mca-params.conf

# configure the path.
RUN echo export 'PATH=$HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin:$HOME/.openmpi/bin:$PATH' >> ~/.bashrc
RUN echo export 'LD_LIBRARY_PATH=$HOME/.openmpi/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

RUN apt-get install -y libgl1-mesa-glx


# -------------------- Others --------------------
RUN echo "orte_keep_fqdn_hostnames=t" >> /.openmpi/etc/openmpi-mca-params.conf

RUN sudo apt-get install -y vim
RUN pip install pandas click
RUN pip install joblib

RUN pip install Cython
RUN pip install scikit-learn

# # Copy your application code to the container (make sure you create a .dockerignore file if any large files or directories should be excluded)
RUN mkdir /src/
WORKDIR /src/
ADD . /src/

RUN make build && make install && make clean
RUN env MPICC=/.openmpi/bin/mpicc pip install mpi4py
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN mkdir /app/
WORKDIR /app/
ADD ./run_cola.py /app/
ADD ./split_dataset.py /app/
