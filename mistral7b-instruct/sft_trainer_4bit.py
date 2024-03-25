# This is a modified version of the modified version from https://gist.githubusercontent.com/lewtun/b9d46e00292d9ecdd6fd9628d53c2814/raw/113d9cc98b1556c8b49f608645e2cf269030995d/sft_trainer.py
# This is a modified version of TRL's `SFTTrainer` example (https://github.com/huggingface/trl/blob/main/examples/scripts/sft_trainer.py), 
# adapted to run with DeepSpeed ZeRO-3 and Mistral-7B-V1.0. The settings below were run on 1 node of 8 x A100 (80GB) GPUs.
#
# Usage:
#   - Install the latest transformers & accelerate versions: `pip install -U transformers accelerate`
#   - Install deepspeed: `pip install deepspeed==0.9.5`
#   - Install TRL from main: pip install git+https://github.com/huggingface/trl.git
#   - Clone the repo: git clone github.com/huggingface/trl.git
#   - Copy this Gist into trl/examples/scripts
#   - Run from root of trl repo with: accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero3.yaml --gradient_accumulation_steps 8 examples/scripts/sft_trainer.py  
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset, Features, Sequence, Value, DatasetDict
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, AutoTokenizer
import wandb
from trl import SFTTrainer


# https://github.com/neuralwork/instruct-finetune-mistral/blob/main/finetune_model.py
wandb.init(project="mistral7b-instruct-news-title")
tqdm.pandas()

output_name = "r256-b4-a16-seq512-l2.0e-5_reformat"

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(default="mistralai/Mistral-7B-Instruct-v0.2", metadata={"help": "The model name"})
    dataset_name: Optional[str] = field(
        default="Jotschi/german-news-titles", metadata={"help": "The dataset name"}
    )
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "The text field of the dataset"})
    log_with: Optional[str] = field(default="wandb", metadata={"help": "Use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=2.0e-5, metadata={"help": "The learning rate"})
    batch_size: Optional[int] = field(default=2, metadata={"help": "The batch size"})
    seq_length: Optional[int] = field(default=2048, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=2, metadata={"help": "The number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "Load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "Load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Whether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default=output_name, metadata={"help": "The output directory"})
    peft_lora_r: Optional[int] = field(default=256, metadata={"help": "The r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "The alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=25, metadata={"help": "The number of logging steps"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "The number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "The number of training steps"})
    save_steps: Optional[int] = field(
        default=500, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default="output/mistral-7b-finetuned-chat", metadata={"help": "The name of the model on HF Hub"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Step 1: Load the dataset
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
coco_dataset = load_dataset(script_args.dataset_name)

def prepare_dialogue(text, title, eos_token):
    instruction="Erstelle einen Titelvorschlag für den Text."
    blurb = 'Below is an instruction that describes a task, paired with an input that provides' \
               ' further context. Write a response that appropriately completes the request.'
    blurb2= 'Below is an instruction that describes a task. Write a response that appropriately completes the request.'
    sample = blurb + '\n\n'
    sample += f'### Instruction:\n{instruction}\n\n'
    sample += f'### Input:\n{text}\n\n'
    sample += f'### Response:\n{title}'
    
    prompt_template = """
{blurb}
### Instruction:
{instruction}
### Input:
{text}
### Answer:
{title}
"""
    return prompt_template.format(blurb=blurb, instruction=instruction, text=text, title=title)
    


def chunk_examples(batch):
    all_samples = []
    batched_text = batch["text"]
    batched_titles = batch["titles"]
    for i in range(len(batched_text)):
        text = batched_text[i]
        titles = batched_titles[i]
        for title in titles:
            all_samples += [ prepare_dialogue(text, title, tokenizer.eos_token) ]
    return {"text": all_samples}

chunked_dataset = coco_dataset.map(chunk_examples, batched=True, num_proc=4,
                      remove_columns=["titles", "text"])

#chunked_dataset = chunked_dataset.shuffle(seed=1234)  # Shuffle dataset here
#chunked_dataset = chunked_dataset.map(lambda samples: tokenizer(samples["text"]), batched=True)


ds_train = chunked_dataset['train'].train_test_split(test_size=0.2, seed=42)


ds_splits = DatasetDict({
    'train': ds_train['train'],
    'test': ds_train['test']
})

#accelerator = Accelerator()
#with accelerator.local_main_process_first():
#    # This will be printed first by local process 0 then in a seemingly
#    # random order by the other processes.
#    print(f"This will be printed by process {accelerator.local_process_index}")
#    val = input("Enter your value: ") 

# Step 2: Load the model
device_map = "auto"
if script_args.load_in_8bit and script_args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif script_args.load_in_8bit or script_args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
    )
    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}
    #device_map = None
    torch_dtype = torch.bfloat16
else:
    device_map = {"": Accelerator().local_process_index}
    #device_map = None
    quantization_config = None
    torch_dtype = None

print("Using device: " + str(device_map))
model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    #device_map=device_map,
    #low_cpu_mem_usage=False,
    quantization_config=quantization_config,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
)

print(device_map)
print("Model: " + str(next(model.parameters()).device))

#if torch.cuda.current_device() == 0:
#    script_args.batch_size=4

# Step 3: Define the training arguments
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=True,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    save_total_limit=script_args.save_total_limit,
    push_to_hub=script_args.push_to_hub,
    hub_model_id=script_args.hub_model_id,
    bf16=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    evaluation_strategy="epoch",
    logging_first_step=True,
)

# Step 4: Define the LoraConfig
if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    peft_config = None


# Step 5: Define the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    max_seq_length=script_args.seq_length,
    train_dataset=ds_splits["train"],
    eval_dataset=ds_splits["test"],
    dataset_text_field=script_args.dataset_text_field,
    peft_config=peft_config,
    packing=True,
)


trainer.train()

# Step 6: Save the model
trainer.save_model(script_args.output_dir)
