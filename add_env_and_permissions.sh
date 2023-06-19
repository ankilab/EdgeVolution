#! /bin/sh

# this script adds the path variables to profile permanently and adds device permissions from zephyr sdk

# Define the path variable    
nrf_path="/opt/nrf-command-line-tools/bin"

# this makes path variable accessible for next login
echo export PATH=$PATH:$nrf_path >> ~/.profile

# this makes path variable accessible in current script
export PATH=$PATH:$nrf_path

echo "$nrf_path was automatically added to PATH variable in ~/.profile"  

sudo cp /opt/zephyr-sdk-0.16.1/sysroots/x86_64-pokysdk-linux/usr/share/openocd/contrib/60-openocd.rules /etc/udev/rules.d
sudo udevadm control --reload


