from transformers import (AutoTokenizer)
from datasets import load_dataset, DatasetDict
dataset_name = "Jotschi/german-news-titles"
system_message = """Du bist Llama, ein AI Assistent erstellt von Johannes. Du wurdest optimiert Titelvorschäge von Nachrichten Artikel zu erstellen."""


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

base_model = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE

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


dataset = load_dataset(dataset_name)
chunked_dataset = dataset.map(chunk_examples, batched=True, num_proc=4,
                      remove_columns=["titles", "text"])



#def process(row):
#    row["text"] = tokenizer.apply_chat_template(row["text"], tokenize=False)
#    return row
#
#chunked_dataset = chunked_dataset.map(process, num_proc= 4, load_from_cache_file=False)

ds_train = chunked_dataset['train'].train_test_split(test_size=0.2, seed=42)

ds_splits = DatasetDict({
    'train': ds_train['train'],
    'test': ds_train['test']
})


ds_train["train"].to_json("train_dataset.json", orient="records", force_ascii=False)
ds_train["test"].to_json("test_dataset.json", orient="records", force_ascii=False)