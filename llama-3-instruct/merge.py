del trainer, model
gc.collect()
torch.cuda.empty_cache()

#reload tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    return_dict = True,
    torch_dtype=torch.float16,
    device_map="auto"
)

model, tokenizer = setup_chat_format(model, tokenizer)

#Merge
model = PeftModel.from_pretrained(model, new_model)
model = model.merge_and_unload()

model.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model, use_temp_dir=False)