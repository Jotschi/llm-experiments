from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging, TextStreamer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os, torch, wandb, platform, warnings
from datasets import load_dataset, DatasetDict
from trl import SFTTrainer


base_model = "mistralai/Mistral-7B-Instruct-v0.3"
dataset_name = "Jotschi/german-news-titles"
new_model = "Jotschi/Mistral-7B-v0.2-german-news-titles-v5"

dataset = load_dataset(dataset_name)

wandb.init(project="mistral7b-instruct-news-title")

max_seq_length=256



################
# Tokenizer
################


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
#tokenizer.add_eos_token = True
#tokenizer.add_bos_token, tokenizer.add_eos_token
#tokenizer.padding_side = 'right'

################
# Dataset
################


def prepare_dialogue(text, title):
  
  #text  = "Einstein gilt als einer der bedeutendsten Physiker der Wissenschaftsgeschichte und weltweit als einer der bekanntesten Wissenschaftler der Neuzeit."
  #title = "Albert Einstein war ein Genie!"

  chat = [
       #{"role": "user", "content": examples["text"]},
       {"role": "user", "content": "Erstelle einen Titelvorschlag für folgenden Artikel:\n" + text},
       {"role": "assistant", "content": "Titelvorschlag: " + title},
    ]
  
  #return {"text":  tokenizer.apply_chat_template(chat, tokenize=False)}
  return tokenizer.apply_chat_template(chat, tokenize=False)


def chunk_examples(batch):
    all_samples = []
    batched_text = batch["text"]
    batched_titles = batch["titles"]
    for i in range(len(batched_text)):
        text = batched_text[i]
        titles = batched_titles[i]
        for title in titles:
            #print("Title: " + title)
            all_samples += [ prepare_dialogue(text, title) ]
    return {"text": all_samples}

#chunked_dataset = coco_dataset.map(chunk_examples, batched=True, num_proc=4,
#                      remove_columns=["titles", "text"])
print(dataset['train'])

train_dataset = dataset['train'].map(chunk_examples, batched=True, num_proc=4, remove_columns=["titles", "text"])
test_dataset = dataset['test'].map(chunk_examples, batched=True, num_proc=4, remove_columns=["titles", "text"])

print(train_dataset);

#ds_train = chunked_dataset['train'].train_test_split(test_size=0.2, seed=42)


#ds_splits = DatasetDict({
#    'train': chunked_dataset['train'],
#    'test': chunked_dataset['test']
#})

################
# Model
################

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map={"": 0}
)
model.config.use_cache = False # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

################
# PEFT
################

peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
    )
model = get_peft_model(model, peft_config)

################
# Training
################

training_arguments = TrainingArguments(
    output_dir= "./results",
    num_train_epochs= 1,
    per_device_train_batch_size= 4,
    gradient_accumulation_steps= 2,
    optim = "paged_adamw_8bit",
    save_steps=50,
    logging_steps= 25,
    learning_rate= 2e-4,
    weight_decay= 0.001,
    fp16= False,
    bf16= True,
    max_grad_norm= 0.3,
    max_steps= -1,
    warmup_ratio= 0.3,
    group_by_length= True,
    lr_scheduler_type= "constant",
    report_to="wandb"
)
# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)
trainer.train()
# Save the fine-tuned model
trainer.model.save_pretrained(new_model)
wandb.finish()
model.config.use_cache = True
model.eval()
