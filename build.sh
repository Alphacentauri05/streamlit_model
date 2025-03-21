#!/bin/bash
# Update package lists
apt-get update 

# Install PortAudio and required dependencies
apt-get install -y portaudio19-dev python3-pyaudio

# Install required Python packages
pip install --no-cache-dir -r requirements.txt
