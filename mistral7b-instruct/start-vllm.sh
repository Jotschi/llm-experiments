#!/bin/bash

set -o nounset
set -o errexit

PORT=10300
#LLM=mistralai/Mistral-7B-Instruct-v0.2
LLM_NAME=$(basename $1)
LLM="/models/$LLM_NAME"
IMAGE="vllm/vllm-openai"
VERSION="v0.3.3"
NAME=vllm

if [ ! -e models/$LLM_NAME ] ; then
  echo "Could not find LLM in models folder"
  exit 10
fi
docker pull $IMAGE:$VERSION


echo "Using LLM $LLM"
docker rm -f $NAME || true
docker run -d --shm-size 16G \
      --name $NAME \
      -p 0.0.0.0:$PORT:8000/tcp \
      -v $(pwd)/.cache:/root/.cache/huggingface \
      -v $(pwd)/models:/models \
      $IMAGE:$VERSION \
      --model $LLM \
      --max-model-len 4096 \
      --load-format safetensors

#--quantization awq \

echo "Starting log output. Press CTRL+C to exit log"
docker logs -f $NAME

