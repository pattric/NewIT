#!/bin/bash
#

update() {
  apt-get update
  apt-get -y install git python3-tk
}

downloadCode() {
  git clone https://github.com/pattric/NewIT.git
  cp -rf NewIT/POC/EAT /usr/local/
}
installPackage()
{
    pip3 install -r requirements.txt
}

launchApp(){
    cd /usr/local/eat
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
