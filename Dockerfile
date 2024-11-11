# Use Python 3.5.4 base image
FROM python:3.5.4-slim

# Set environment variables
ENV TEMP_MRCNN_DIR=/tmp/mrcnn \
    TEMP_COCOAPI_DIR=/tmp/coco \
    TEMP_PYTHON_VERSION=3.5
	ENV MRCNN_DIR /mrcnn
# Update apt sources to use archived Debian repositories and install dependencies
RUN sed -i 's|http://deb.debian.org|http://archive.debian.org|g' /etc/apt/sources.list && \
    sed -i '/security.debian.org/d' /etc/apt/sources.list && \
    sed -i '/jessie-updates/d' /etc/apt/sources.list && \
    apt-get update -o Acquire::Check-Valid-Until=false && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
        build-essential \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libglib2.0-0 \
        wget \
        git \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Install specific version of CMake
RUN curl -L https://github.com/Kitware/CMake/releases/download/v3.7.2/cmake-3.7.2-Linux-x86_64.tar.gz -o cmake.tar.gz && \
    tar -zxvf cmake.tar.gz -C /usr --strip-components=1 && \
    rm cmake.tar.gz

# Upgrade pip to compatible version for Python 3.5
RUN pip install --no-cache-dir --upgrade "pip<21.0" --trusted-host pypi.python.org --trusted-host pypi.org --trusted-host files.pythonhosted.org

# Install essential packages in a targeted order to avoid conflicts
RUN pip install --no-cache-dir --trusted-host pypi.python.org --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    "numpy==1.15.4" \
    "scipy==1.1.0" \
    "matplotlib==2.2.5" \
    "h5py==2.9.0" \
    "keras==2.1.5" \
    "tensorflow==1.5.0"

# Install scikit-image with an older version and necessary testing utilities
RUN pip install --no-cache-dir --trusted-host pypi.python.org --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    "scikit-image==0.13.1" \
    "nose"  # Adding nose for testing decorators

# Install Cython, which is required for COCO API
RUN pip install --no-cache-dir --trusted-host pypi.python.org --trusted-host pypi.org --trusted-host files.pythonhosted.org Cython

# Clone Mask R-CNN repository and install it
RUN git clone https://github.com/matterport/Mask_RCNN.git $TEMP_MRCNN_DIR && \
    cd $TEMP_MRCNN_DIR && python3 setup.py install

RUN mkdir -p $MRCNN_DIR/coco
RUN mkdir -p $MRCNN_DIR/logs/tshirt_dataset20241109T1557/
RUN wget -O $MRCNN_DIR/mask_rcnn_coco.h5 https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5


# Clone COCO API repository and install it
RUN git clone https://github.com/waleedka/coco.git $TEMP_COCOAPI_DIR && \
    cd $TEMP_COCOAPI_DIR/PythonAPI && \
    make && \
    make install && \
    python3 setup.py install

# Copy custom training files to Mask R-CNN directory
COPY visualize.py $TEMP_MRCNN_DIR/mrcnn
COPY image_for_training/images /mrcnn/dataset/images
COPY image_for_training/annotations /mrcnn/dataset
COPY dataset.py /mrcnn
COPY train.py /mrcnn
COPY inference.py /mrcnn
COPY mask_rcnn_tshirt_dataset_0005.h5 /mrcnn/logs/tshirt_dataset20241109T1557
# Set the working directory
WORKDIR /mrcnn
CMD ["/bin/bash"]