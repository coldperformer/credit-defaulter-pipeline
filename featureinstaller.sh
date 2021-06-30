#!/usr/bin/bash

# Update package indices
sudo apt-get update

# Update Ubuntu software and all installed packages
sudo apt-get upgrade -y

# Install Python pip
sudo apt install python3-pip==21.1.1

echo "----------PIP Installed----------"

sudo apt-get install expect -y

echo "----------expect Installed----------"

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

## MySQL Setup
# Installing MySQL server
sudo apt install mysql-server -y

echo "----------MySQL Server Installed----------"

# Starting the MySQL server
sudo /etc/init.d/mysql start -u root -p

echo "----------MySQL Server Started----------"

sudo mysql -u root -p -e "UPDATE mysql.user SET Password=PASSWORD('root') WHERE User='root';"

echo "----------MySQL Client User Updated----------"

# First time setup for secure installation
# The MySQL secure installation is automated using expect
# sudo yes | mysql_secure_installation -u root -p 

# echo "----------MySQL Installation Success!----------"