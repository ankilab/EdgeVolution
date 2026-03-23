# ===========================================================================
# EdgeVolution Multi-Target Dockerfile
#
# Targets:
#   ml       - TensorFlow 2.9.1 GPU + Python deps + project code (DEFAULT)
#   embedded - Extends ml with nRF tools, J-Link, Zephyr SDK, west workspace
#
# Usage:
#   docker build -t edgevolution .                                # builds ml
#   docker build --target embedded -t edgevolution-embedded .     # builds embedded
# ===========================================================================

# ===========================================================================
# Stage 1: ML / NAS runtime (default target)
# Base: tensorflow/tensorflow:2.9.1-gpu (Ubuntu 20.04, Python 3.8, CUDA 11.2)
# ===========================================================================
FROM tensorflow/tensorflow:2.9.1-gpu AS ml

ENV DEBIAN_FRONTEND=noninteractive

# The TF base image already provides python3, pip, wget, curl, etc.
# Only install packages that are actually missing.
#   git         - required by GitPython (used in utils/saver.py)
#   ffmpeg      - required by librosa/pydub for audio processing
#   libsndfile1 - required by SoundFile for audio I/O
#   vim         - convenience for interactive debugging
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /EdgeVolution

# Copy requirements.txt FIRST for Docker layer caching.
# pip install is only re-run when requirements.txt changes,
# not on every source code change.
COPY requirements.txt /EdgeVolution/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --timeout 600 -r requirements.txt

# Copy the rest of the project source code.
# .dockerignore prevents copying .git, docs, notebooks, build artifacts, etc.
COPY . /EdgeVolution
RUN pip install --no-cache-dir -e .

CMD ["/bin/bash"]


# ===========================================================================
# Stage 2: Embedded toolchain (extends ml)
# Adds: nRF CLI tools, Segger J-Link, CMake >= 3.20, Zephyr SDK, west workspace
# ===========================================================================
FROM ml AS embedded

ENV DEBIAN_FRONTEND=noninteractive

# System packages required by Zephyr, west, nRF tools, and the C/C++ build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg \
    lsb-release \
    udev \
    libusb-1.0-0 \
    build-essential \
    ninja-build \
    gperf \
    ccache \
    dfu-util \
    device-tree-compiler \
    xz-utils \
    file \
    make \
    gcc \
    gcc-multilib \
    g++-multilib \
    libsdl2-dev \
    libmagic1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# CMake 3.28 (compatible with Python 3.8; CMake 4.x requires Python 3.9+)
# ---------------------------------------------------------------------------
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.28.6/cmake-3.28.6-linux-x86_64.tar.gz \
    && tar xf cmake-3.28.6-linux-x86_64.tar.gz --strip-components=1 -C /usr/local \
    && rm cmake-3.28.6-linux-x86_64.tar.gz

# ---------------------------------------------------------------------------
# nRF Command Line Tools (including nrfjprog)
# ---------------------------------------------------------------------------
WORKDIR /tmp/nrf
RUN wget -q "https://nsscprodmedia.blob.core.windows.net/prod/software-and-other-downloads/desktop-software/nrf-command-line-tools/sw/versions-10-x-x/10-21-0/nrf-command-line-tools-10.21.0_linux-amd64.tar.gz" \
    && tar xf nrf-command-line-tools-10.21.0_linux-amd64.tar.gz \
    && cp -a nrf-command-line-tools /opt/ \
    && rm -rf /tmp/nrf

ENV PATH="${PATH}:/opt/nrf-command-line-tools/bin"

# ---------------------------------------------------------------------------
# Segger J-Link
# The J-Link .deb postinst script tries to start udev/systemd services
# which don't exist in Docker. Workaround:
#   1. dpkg --unpack  (extract files without running postinst)
#   2. Remove the postinst script
#   3. dpkg --configure  (mark package as configured)
#   4. apt-get install -yf  (resolve any missing dependencies)
# ---------------------------------------------------------------------------
WORKDIR /tmp/jlink
RUN wget -q --post-data="accept_license_agreement=accepted&submit=Download+software" \
        https://www.segger.com/downloads/jlink/JLink_Linux_V788n_x86_64.deb \
    && dpkg --unpack JLink_Linux_V788n_x86_64.deb \
    && rm -f /var/lib/dpkg/info/jlink.postinst \
    && apt-get update && apt-get install -yf \
    && rm -rf /var/lib/apt/lists/* /tmp/jlink

# ---------------------------------------------------------------------------
# Zephyr SDK 0.16.5-1
# ---------------------------------------------------------------------------
WORKDIR /opt
RUN wget -q https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.16.5-1/zephyr-sdk-0.16.5-1_linux-x86_64.tar.xz \
    && wget -q -O - https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.16.5-1/sha256.sum | shasum --check --ignore-missing \
    && tar xf zephyr-sdk-0.16.5-1_linux-x86_64.tar.xz \
    && rm zephyr-sdk-0.16.5-1_linux-x86_64.tar.xz

ENV ZEPHYR_SDK_INSTALL_DIR="/opt/zephyr-sdk-0.16.5-1"

# ---------------------------------------------------------------------------
# Upgrade west to 1.0.0 (Zephyr 4.0.0 build.py requires west.commands.Verbosity,
# which was removed in west 0.14.0 and re-added in west 1.0.0)
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir west==1.0.0

# ---------------------------------------------------------------------------
# Zephyr workspace is provided via volume mount from the host.
# The host must have .west/, zephyr/, modules/, etc. in tflite/.
# ---------------------------------------------------------------------------
WORKDIR /EdgeVolution

CMD ["/bin/bash"]
