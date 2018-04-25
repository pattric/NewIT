#!/bin/bash
#
downloadCode() {
  git clone https://github.com/pattric/NewIT.git
  cp -rf NewIT/POC/EAT /usr/local/eat
}
installPackage()
{
    cd /usr/local/eat
    pip install -r requirements.txt
}

launchApp(){
    python app.py
}

downloadCode
installPackage
launchApp