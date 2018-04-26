#!/bin/bash
#

update() {
  apt-get update
  apt-get -y install git python3-tk
}

downloadCode() {
  cd /tmp
  git clone https://github.com/pattric/NewIT.git
  cd NewIT/POC
  cp -rf EAT /usr/local/
}
installPackage()
{
    cd /usr/local/EAT
    pip3 install -r requirements.txt
}

launchApp(){
    cd /usr/local/EAT
    ls
    python3 app.py
}
echo "===> update sys package"
update

echo "===> Downloading code from github"
downloadCode
echo "===> Installing required python packages"
installPackage
echo "===> Starting Flask app to sever predict function"
launchApp
