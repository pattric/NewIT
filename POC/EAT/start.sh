#!/bin/bash
#
installPackageOnContainer()
{
    apt-get update
    apt-get -y install python3-tk
    cd /usr/local/eat
    pip3 install -r requirements.txt
}

launchApp(){
    cd /usr/local/eat
    # python3 app.py
    python3 -m flask run --host=0.0.0.0 --port=8080

}

echo "===> Installing required python packages"
installPackageOnContainer
echo "===> Starting Flask app to sever predict function"
launchApp
