#!/bin/bash

set -o nounset
set -o errexit

PORT=10300
#LLM=mistralai/Mistral-7B-Instruct-v0.2
LLM_NAME=$(basename $1)
LLM="/models/$LLM_NAME"
IMAGE="ghcr.io/huggingface/text-generation-inference:2.0"

if [ ! -e models/$LLM_NAME ] ; then
  echo "Could not find LLM in models folder"
  exit 10
fi
docker pull $IMAGE
NAME=tgi

#LLM_NAME="mistralai/Mistral-7B-Instruct-v0.2"
LLM_FOLDER_NAME="models--$(echo $LLM_NAME | sed 's/\//--/g')"

docker rm -f $NAME || true

echo "Using LLM $LLM with folder $LLM_FOLDER_NAME"
read
docker run \
      -d \
      --gpus all --shm-size 1g \
      --name $NAME \
      -p 0.0.0.0:$PORT:80/tcp \
      -v $(pwd)/$1:/data/$LLM_FOLDER_NAME \
      $IMAGE \
      --model-id "/data/$LLM_FOLDER_NAME"

      #-v $(pwd)/tgi-data:/data \
#docker run -d --shm-size 16G \
#      -p 0.0.0.0:$PORT:8000/tcp \
#      -v $(pwd)/.cache:/root/.cache/huggingface \
#      $IMAGE
#      --model-id $LLM

echo "Starting log output. Press CTRL+C to exit log"
docker logs -f $NAME

