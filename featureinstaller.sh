#!/usr/bin/bash

# Update package indices
sudo apt-get update

# Update Ubuntu software and all installed packages
sudo apt-get upgrade -y

# Install Python pip
sudo apt install python3-pip=21.1.1

echo "----------PIP Installed----------"

# Required for Pycario and further for gcp
sudo apt-get install libcairo2-dev

echo "----------Libcario Installed----------"

# Required for mysql connector and mysql and mysql client
sudo apt-get install libmysqlclient-dev

echo "----------MySQL Client Installed----------"

# Install python packages
pip3 install -r requirements.txt

echo "----------Requirements Installed----------"

# Update package indices
sudo apt-get update

# Update Ubuntu software and all installed packages
sudo apt-get upgrade -y

echo "----------Dependencies updated----------"