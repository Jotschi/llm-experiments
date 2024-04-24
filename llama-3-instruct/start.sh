#!/bin/bash

. venv/bin/activate

# Login via huggingface-cli login

ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=1 ./run_fsdp_qlora.py --config llama_3_70b_fsdp_qlora.yaml
#python train.py