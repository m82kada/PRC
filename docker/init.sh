#!/bin/bash

if [ $# -ne 1 ]; then
  image="vmoe:latest"
else
  image=$1
fi

user_id=`id -u $USER`
group_id=`id -g $USER`


echo $image > image.txt

echo "docker build --build-arg user_id=$user_id --build-arg group_id=$group_id -t $image -f Dockerfile ."
docker build --build-arg user_id=$user_id --build-arg group_id=$group_id -t $image -f Dockerfile .
