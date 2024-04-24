import gc
import os
from huggingface_hub import login
import torch
import wandb
import multiprocessing
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from trl import ORPOConfig, ORPOTrainer, setup_chat_format

wandb.init(project="llama3-instruct-news-title")

LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}"
        "{% elif message['role'] == 'user' %}"
            "{{ '\n\nHuman: ' + message['content'] +  eos_token }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\n\nAssistant: ' }}"
    "{% endif %}"
)

if torch.cuda.get_device_capability()[0] >= 8:
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
else:
    attn_implementation = "eager"
    torch_dtype = torch.float16
     
# Model
base_model = "meta-llama/Meta-Llama-3-8B"
new_model = "OrpoLlama3-8B-FT"
dataset_name = "Jotschi/german-news-titles"
system_message = """Du bist Llama, ein AI Assistent erstellt von Johannes. Du wurdest optimiert Titelvorschäge von Nachrichten Artikel zu erstellen."""

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation
)
#model, tokenizer = setup_chat_format(model, tokenizer)
model = prepare_model_for_kbit_training(model)

def create_conversation(text, title):
    msgs = []
    msgs += [{"role": "system", "content": system_message}] 
    msgs += [{"role": "user", "content": text}] 
    msgs += [{"role": "assistant", "content": title}] 
    return msgs;

def chunk_examples(batch):
    all_samples = []
    batched_text = batch["text"]
    batched_titles = batch["titles"]
    for i in range(len(batched_text)):
        text = batched_text[i]
        titles = batched_titles[i]
        for title in titles:
            all_samples +=  [create_conversation(text, title)]
    return {"text": all_samples}

def process(row):
    row["text"] = tokenizer.apply_chat_template(row["text"], tokenize=False)
    return row

# Load and process the dataset
dataset = load_dataset(dataset_name)
chunked_dataset = dataset.map(chunk_examples, batched=True, num_proc=4,
                      remove_columns=["titles", "text"])

# Apply the chat template
chunked_dataset = chunked_dataset.map(process, num_proc= 4, load_from_cache_file=False)

# Now split the DS
ds_train = dataset['train'].train_test_split(test_size=0.2, seed=42)

ds_splits = DatasetDict({
    'train': ds_train['train'],
    'test': ds_train['test']
})

orpo_args = ORPOConfig(
    learning_rate=8e-6,
    beta=0.1,
    lr_scheduler_type="linear",
    max_length=1024,
    max_prompt_length=512,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    max_steps=1000,
    evaluation_strategy="steps",
    eval_steps=0.2,
    logging_steps=1,
    warmup_steps=10,
    report_to="wandb",
    output_dir="./results/",
)

trainer = ORPOTrainer(
    model=model,
    args=orpo_args,
    train_dataset=ds_train['train'],
    eval_dataset=ds_train['test'],
    peft_config=peft_config,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(new_model)
wandb.finish()