# # Use an official lightweight Python image.
FROM tensorflow/tensorflow:2.9.1-gpu

# # Prevent interactive dialogs during apt-get installs
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# System dependencies & basic setup
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    dpkg \
    git \
    vim \
    vim-common \
    ca-certificates \
    udev \
    libusb-1.0-0 \
    libxcb-render-util0 \
    libxcb-randr0 \
    libxcb-icccm4 \
    libxcb-keysyms1 \
    libxcb-image0 \
    libxkbcommon-x11-0 \
    sudo \
    gnupg \
    lsb-release \
    build-essential \
    ffmpeg \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
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
    && rm -rf /var/lib/apt/lists/*


# ---------------------------------------------------------------------------
# Install nRF Command Line Tools (including nrfjprog)
# ---------------------------------------------------------------------------
WORKDIR /tmp/nrf
RUN wget "https://nsscprodmedia.blob.core.windows.net/prod/software-and-other-downloads/desktop-software/nrf-command-line-tools/sw/versions-10-x-x/10-21-0/nrf-command-line-tools-10.21.0_linux-amd64.tar.gz" \
    && tar xvf nrf-command-line-tools-10.21.0_linux-amd64.tar.gz \
    && sudo cp -a nrf-command-line-tools /opt/ \
    && rm -rf nrf-command-line-tools-10.21.0_linux-amd64.tar.gz nrf-command-line-tools

ENV PATH="${PATH}:/opt/nrf-command-line-tools/bin"

# ---------------------------------------------------------------------------
# Install Segger J-Link software
# (You might want to update the version to the latest.)
# ---------------------------------------------------------------------------
RUN /lib/systemd/systemd-udevd --daemon
RUN udevadm monitor &

RUN wget --post-data="accept_license_agreement=accepted&submit=Download+software" https://www.segger.com/downloads/jlink/JLink_Linux_V788n_x86_64.deb 

# Took this small workaround from: https://forums.docker.com/t/udevadm-monitor-in-docker-file/125723
RUN dpkg --unpack JLink_Linux_V788n_x86_64.deb
RUN rm /var/lib/dpkg/info/jlink.postinst -f
RUN dpkg --configure jlink
RUN apt install -yf 

# ---------------------------------------------------------------------------
# Install CMake (some distributions already have an older version).
# ---------------------------------------------------------------------------
RUN wget https://apt.kitware.com/kitware-archive.sh \
    && sudo bash kitware-archive.sh \
    && rm kitware-archive.sh \
    && apt-get update && apt-get install -y cmake

# ---------------------------------------------------------------------------
# Install west (Zephyr's meta-tool) and dependencies
# ---------------------------------------------------------------------------
RUN pip install west

# ---------------------------------------------------------------------------
# Install Zephyr SDK to /opt/zephyr-sdk-0.16.5-1
# ---------------------------------------------------------------------------
WORKDIR /opt
RUN wget https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.16.5-1/zephyr-sdk-0.16.5-1_linux-x86_64.tar.xz \
    && wget -O - https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.16.5-1/sha256.sum | shasum --check --ignore-missing \
    && tar xvf zephyr-sdk-0.16.5-1_linux-x86_64.tar.xz \
    && rm zephyr-sdk-0.16.5-1_linux-x86_64.tar.xz

ENV ZEPHYR_SDK_INSTALL_DIR="/opt/zephyr-sdk-0.16.5-1"

# ---------------------------------------------------------------------------
# Create a non-root user
# ---------------------------------------------------------------------------
ARG USERNAME=edgedev
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Add user to dialout group for serial port access
RUN usermod -a -G dialout,plugdev $USERNAME

# ---------------------------------------------------------------------------
# Create a project directory and copy your files
# (Assuming you have requirements.txt and a tflite/ folder, etc.)
# ---------------------------------------------------------------------------
WORKDIR /EdgeVolution
COPY . /EdgeVolution

# ---------------------------------------------------------------------------
# Install Python dependencies (as root, but accessible to user)
# ---------------------------------------------------------------------------
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

RUN pip install --upgrade numba librosa

# ---------------------------------------------------------------------------
# Zephyr project initialization (as root first)
# ---------------------------------------------------------------------------
WORKDIR /EdgeVolution/tflite
RUN if [ ! -f .west/config ]; then west init .; else echo "West workspace already initialized"; fi && \
    west config manifest.group-filter -- +optional && \
    west zephyr-export

# ---------------------------------------------------------------------------
# Change ownership of the project directory to the non-root user
# ---------------------------------------------------------------------------
RUN chown -R $USERNAME:$USERNAME /EdgeVolution

# Switch to non-root user
USER $USERNAME

WORKDIR /EdgeVolution
CMD ["/bin/bash"]