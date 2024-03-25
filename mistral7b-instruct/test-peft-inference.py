from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

device = "cuda" # the device to load the model onto

model_path = "./r256-b4-a16-seq512-l2.0e-5_reformat/checkpoint-1000"
base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=False, load_in_4bit=True
)


base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)

query = "Albert Einstein (* 14. März 1879 in Ulm; † 18. April 1955 in Princeton, New Jersey) war ein schweizerisch-US-amerikanischer theoretischer Physiker deutscher Herkunft. Der Wissenschaftler jüdischer Abstammung hatte bis 1896 die württembergische Staatsbürgerschaft, ab 1901 die Schweizer Staatsbürgerschaft und ab 1940 zusätzlich die US-amerikanische. Preußischer Staatsangehöriger war er von 1914 bis 1934."
eval_prompt = """Patient's Query:\n\n {} ###\n\n""".format(query)
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

ft_model = PeftModel.from_pretrained(base_model, "mistral7b-instruct/r256-b4-a16-seq512-l2.0e-5_prefix/checkpoint-1000")
output = ft_model.generate(input_ids=model_input["input_ids"].to(device),
                           attention_mask=model_input["attention_mask"], 
                           max_new_tokens=125, repetition_penalty=1.15)
result = tokenizer.decode(output[0], skip_special_tokens=True).replace(eval_prompt, "")


print(result)