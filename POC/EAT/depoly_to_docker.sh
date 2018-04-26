#!/bin/bash
#

deploy_dir=$(pwd)

update() {
  apt-get update
  apt-get -y install git python3-tk unzip wget
}

downloadCode() {

  rm -rf /tmp/eat && mkdir /tmp/eat
  
  if [ ! -f /usr/local/eat/data/glove.6B.50d.txt ]; then
    wget http://nlp.stanford.edu/data/glove.6B.zip -O /tmp/eat/glove.6B.zip
    unzip /tmp/eat/glove.6B.zip
    cp -rf /tmp/eat/glove.6B.50d.txt /usr/local/eat/data
  fi
  cd /tmp/eat
  git clone https://github.com/pattric/NewIT.git
  ls
  #rm -rf /usr/local/EAT
  cd NewIT/POC/EAT && cp -rf . /usr/local/eat
  cd /usr/local/eat
  ls
}




downloadCode


cd $deploy_dir
echo $deploy_dir 

cp $deploy_dir/start.sh /usr/local/eat
sudo chmod 777 -R /usr/local/eat

docker-compose -f docker-compose.yaml down
docker-compose -f docker-compose.yaml up -d