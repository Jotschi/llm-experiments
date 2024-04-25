#!/bin/bash

NAME=ollama
VERSION=0.1.32

docker rm -f $NAME
docker run -d  --gpus=all --shm-size 1g \
    -v ollama:/root/.ollama \
    -p 11434:11434 \
    --network=ollama-network \
    --name $NAME \
    ollama/ollama:$VERSION
docker logs -f $NAME
