FROM nvcr.io/nvidia/tensorrt:20.03-py3

# optional python dependecies for tensorrt
RUN /opt/tensorrt/python/python_setup.sh

# system wide dependecies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    locales \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# locale fixes
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# patch graphsurgeon
RUN sed -i 's/.*for key.*/    node.attr["dtype"].type = 1\n&/' \
    /usr/lib/python3.6/dist-packages/graphsurgeon/node_manipulation.py && \
    sed -i 's/.*for key.*/    node.attr["dtype"].type = 1\n&/' \
    /usr/lib/python2.7/dist-packages/graphsurgeon/node_manipulation.py

# build libflattenconcat.so
ADD flattenconcat /tmp/flattenconcat
RUN mkdir -p /tmp/flattenconcat/build && \
    cd /tmp/flattenconcat/build && \
    cmake .. && \
    make && \
    mkdir -p /workspace/lib && \
    mv libflattenconcat.so /workspace/lib && \
    rm -rf /tmp/flattenconcat

WORKDIR /workspace
