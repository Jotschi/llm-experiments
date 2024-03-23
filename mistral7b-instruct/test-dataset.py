from datasets import load_dataset

dataset_name = "Jotschi/german-news-titles"
llm_dataset = load_dataset(dataset_name)
#split_dataset = llm_dataset["train"].train_test_split(test_size=0.1)

def prepare_dialogue(text, title, eos_token):
    sample = ""
    sample += f"<|user|>\n{text}{eos_token}\n"
    sample += f"<|assistant|>\n{title}{eos_token}\n"
    return sample

def chunk_examples(batch):
    all_samples = []
    eos_token="<EOS>"
    batched_text = batch["text"]
    batched_titles = batch["titles"]
    for i in range(len(batched_text)):
        text = batched_text[i]
        titles = batched_titles[i]
        for title in titles:
            all_samples += [ prepare_dialogue(text, title, eos_token) ]
    return {"text": all_samples}

chunked_dataset = llm_dataset.map(chunk_examples, batched=True, num_proc=4,
                      remove_columns=["titles", "text"])

print(llm_dataset)
chunked_dataset = llm_dataset.map(chunk_examples, batched=True, num_proc=4,
                                   remove_columns=[ "text", "titles"])
print(chunked_dataset)
print(len(chunked_dataset['train']))

print(chunked_dataset['train'][0])
print(chunked_dataset['train'][1])
