#!/bin/bash

set -o nounset
set -o errexit


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

. $SCRIPT_DIR/venv/bin/activate


NAME="$1"
PEFT_CHECKPOINT="checkpoint-$2"

PEFT_PATH="$NAME/$PEFT_CHECKPOINT"
if [ ! -e $PEFT_PATH ] ; then
  echo "Failed to find model folder $PEFT_PATH"
  exit 10
fi


echo "Using name $NAME"

python merge_peft_adapter.py \
    --peft_model_path=$PEFT_PATH \
    --output_dir=models/$NAME-$2 \
    --base_model_name_or_path=mistralai/Mistral-7B-v0.1

