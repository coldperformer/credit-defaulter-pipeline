#!/usr/bin/bash

# Update package indices
sudo apt-get update

# Update Ubuntu software and all installed packages
sudo apt-get upgrade -y

# Install Python pip
sudo apt install python3-pip==21.1.1

# Required for Pycario and further for gcp
sudo apt-get install libcairo2-dev

# Required for mysql connector and mysql and mysql client
sudo apt-get install libmysqlclient-dev

# Install python packages
pip3 install -r requirements.txt

# Update package indices
sudo apt-get update

# Update Ubuntu software and all installed packages
sudo apt-get upgrade -y

## MySQL Setup
# Installing MySQL server
sudo apt install mysql-server -y

# Starting the MySQL server
sudo /etc/init.d/mysql start

# First time setup for secure installation
sudo yes | mysql_secure_installation