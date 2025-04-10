#!/bin/bash
# Install system dependencies
apt-get update
apt-get install -y cmake libx11-dev libopenblas-dev liblapack-dev wget bzip2

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Run setup script
bash ./setup.sh
