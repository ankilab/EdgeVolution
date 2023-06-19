#! /bin/sh

# Check if nrfjprog is already installed
command -v nrfjprog >/dev/null
nrfjprog_installed=$?

if ! [ $nrfjprog_installed -eq 1]; then
    echo "installing nrf jprog"
    wget "https://nsscprodmedia.blob.core.windows.net/prod/software-and-other-downloads/desktop-software/nrf-command-line-tools/sw/versions-10-x-x/10-21-0/nrf-command-line-tools-10.21.0_linux-amd64.tar.gz"
    tar xvf nrf-command-line-tools-10.21.0_linux-amd64.tar.gz
    sudo cp -a nrf-command-line-tools /opt/

    rm nrf-command-line-tools-10.21.0_linux-amd64.tar.gz
    rm -r -f nrf-command-line-tools/

    # readme txt containted in 
    rm Readme.txt



    # install jlink segger
    wget  --post-data="accept_license_agreement=accepted&submit=Download+software" https://www.segger.com/downloads/jlink/JLink_Linux_V788f_x86_64.deb

    sudo dpkg JLink_Linux_V788f_x86_64.deb
    rm JLink_Linux_V788f_x86_64.deb
    
    # this file gets automatically installed from nrfjprog but can not be installed correctly
    rm JLink_Linux_V780c_x86_64.tgz

    # add variables to PATH temporarily

    # Define the path variable    
    nrf_path="/opt/nrf-command-line-tools/bin"

    # this makes path variable accessible in current script
    export PATH=$PATH:$nrf_path

fi
exit

# Check if Python is already installed
command -v python3 >/dev/null
python_installed=$?

# Check if apt-get installed
command -v apt-get >/dev/null
apt_get_installed=$?



# Check the exit code and display a message
if ! [ $python_installed -eq 0 ]; then
    echo "Python 3 is not installed. Installing Python 3..."

    # Update package manager
    if ! [ $apt_get_installed -eq 0 ]; then
        sudo apt-get update 
        sudo apt-get install -y python3
    else
        echo "Unable to install Python 3. Please install it manually."
        exit 1
    fi

    echo "Python 3 has been installed successfully!"
else
    echo "Python 3 is already installed."
fi



# Now creating virtual environment
venv_name=".venv"

# Check if the venv directory already exists
if [ -d "$venv_name" ]; then
    echo "Virtual environment $venv_name already exists."
else
    # Create the virtual environment
    python3 -m venv "$venv_name"

    # Display a success message
    echo "Virtual environment $venv_name has been created and activated."
fi
# Set the activation script path
activation_script="./$venv_name/bin/activate"

# Activate the virtual environment
chmod +x $activation_script
. $activation_script


# install requirements
pip install -r requirements.txt



# install some dependencies for dataset
command -v ffmpeg >/dev/null
ffmpeg_installed=$?
if ! [ $ffmpeg_installed -eq 0 ]; then
    echo "ffmpeg will be installed" 
    sudo apt install ffmpeg
fi


project_dir=$(pwd)

# now install zephyr

# Check if zephyr installed
command -v west >/dev/null
west_installed=$?

if  ! [ $west_installed -eq 0 ]; then
    echo "installing west"

    wget https://apt.kitware.com/kitware-archive.sh
    sudo bash kitware-archive.sh
    rm kitware-archive.sh

    sudo apt install --no-install-recommends git cmake ninja-build gperf \
    ccache dfu-util device-tree-compiler wget \
    python3-dev python3-pip python3-setuptools python3-tk python3-wheel xz-utils file \
    make gcc gcc-multilib g++-multilib libsdl2-dev libmagic1

    cmake --version
    dtc --version
    pip install west
else
    echo "West dependencies already set up."

fi

# now install zephyr sdk
sdk="/opt/zephyr-sdk-0.16.1"

# Check if the directory exists
if ! [ -d "$sdk" ]; then
    echo $(pwd)

    echo "creating zepyhr sdk"
    sudo mkdir $sdk
    echo $(pwd)
    wget https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.16.1/zephyr-sdk-0.16.1_linux-x86_64.tar.xz
    wget -O - https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.16.1/sha256.sum | shasum --check --ignore-missing

    sudo tar xvf zephyr-sdk-0.16.1_linux-x86_64.tar.xz -C /opt
    rm zephyr-sdk-0.16.1_linux-x86_64.tar.xz
    cd sdk
    ./setup.sh
    ls
else

    echo "Zephyr SDK already installed."
fi

# now creating west build 
cd $project_dir
cd ./tflite/

# init local zephyr project
zeph="zephyr/"

if ! [ -d "$zeph" ]; then
    west init .
    west update
    west zephyr-export

    # change protobuf version to 3.20.0 because of dependency conflicts
    sed -i 's/protobuf>=3\.20\.3/protobuf=3.19.0/' ~/zephyrproject/zephyr/scripts/requirements-extras.txt
    sed -i 's/img-tool>=1\.9/image-tool/' ~/zephyrproject/zephyr/scripts/requirements-extras.txt
    sed -i 's/pyocd>=0\.35\.0/pyocd/' ~/zephyrproject/zephyr/scripts/requirements-extras.txt
    sed -i 's/pyocd>=0\.35\.0/pyocd/' ~/zephyrproject/zephyr/scripts/requirements-run-test.txt

    pip install -r ./zephyr/scripts/requirements.txt
else
    echo "Local zephyr directory already set up." 

fi 
