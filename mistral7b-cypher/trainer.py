#!pip install wandb -qqq
import logging
from dataclasses import dataclass, field
import os
import random
import wandb
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, TrainingArguments
from trl.commands.cli_utils import  TrlParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
        set_seed,

)
from trl import setup_chat_format
from peft import LoraConfig
from trl import (SFTTrainer)


wandb.init(project="mistral7b-instruct-test")

# Parameters
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
max_seq_length=256
dataset_name="Jotschi/kleiner-astronaut"



################
# Dataset
################

dataset = load_dataset(dataset_name)

#ds_splits = DatasetDict({
#    'train': dataset['train'],
#    'test': dataset['test']
#})

#print(ds_splits['train'])
#print()

################
# Settings
################

training_arguments = TrainingArguments(
    output_dir= "./results",
    num_train_epochs= 1,
    per_device_train_batch_size= 2,
    gradient_accumulation_steps= 2,
    optim = "paged_adamw_8bit",
    save_steps=50,
    logging_steps= 25,
    learning_rate= 2e-4,
    weight_decay= 0.001,
    fp16= False,
    bf16= False,
    max_grad_norm= 0.3,
    max_steps= -1,
    warmup_ratio= 0.3,
    group_by_length= True,
    lr_scheduler_type= "constant",
    report_to="wandb"
)


################
# Tokenizer
################

# Tokenizer        
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# template dataset
def template_dataset(examples):
    #print(examples["text"])
    chat = [
       #{"role": "user", "content": examples["text"]},
       {"role": "user", "content": "Write hello world"},
       {"role": "assistant", "content": "Hallo Welt"},
    ]
    
    return {"text":  tokenizer.apply_chat_template(chat, tokenize=False)}

train_dataset = dataset['train'].map(template_dataset, remove_columns=["text"])
test_dataset = dataset['test'].map(template_dataset, remove_columns=["text"])

# print random sample
with training_arguments.main_process_first(
    desc="Log a few random samples from the processed training set"
):
    for index in random.sample(range(len(train_dataset)), 2):
        print(train_dataset[index]["text"])
        
        
################
# Model
################

torch_dtype = torch.bfloat16
quant_storage_dtype = torch.bfloat16

quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=quant_storage_dtype,
    )

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    attn_implementation="sdpa", # use sdpa, alternatively use "flash_attention_2"
    torch_dtype=quant_storage_dtype,
    use_cache=False if training_arguments.gradient_checkpointing else True,  # this is needed for gradient checkpointing
)

if training_arguments.gradient_checkpointing:
    model.gradient_checkpointing_enable()
    
    
    
    # ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=4 ./scripts/run_fsdp_qlora.py --config llama_3_70b_fsdp_qlora.yaml

################
# PEFT
################

# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)


################
# Training
################

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    dataset_text_field="text",
    eval_dataset=test_dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
)

#dataset_kwargs={
#    "add_special_tokens": False,  # We template with special tokens
#    "append_concat_token": False,  # No need to add additional separator token
#},
if trainer.accelerator.is_main_process:
    trainer.model.print_trainable_parameters()

##########################
# Train model
##########################
#checkpoint = None
#if training_args.resume_from_checkpoint is not None:
#    checkpoint = training_args.resume_from_checkpoint
#trainer.train(resume_from_checkpoint=checkpoint)

# train
trainer.train()

if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
trainer.save_model()
wandb.finish()