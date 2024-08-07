from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import sys

device = "cuda" # the device to load the model onto

model_path = sys.argv[1]
model_path = "./" + model_path

base_model_id = "mistralai/Mistral-7B-Instruct-v0.3"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant=False
)


base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=False
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
#tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token
tokenizer.padding_side = 'right'

#query = "Albert Einstein (* 14. März 1879 in Ulm; † 18. April 1955 in Princeton, New Jersey) war ein schweizerisch-US-amerikanischer theoretischer Physiker deutscher Herkunft. Der Wissenschaftler jüdischer Abstammung hatte bis 1896 die württembergische Staatsbürgerschaft, ab 1901 die Schweizer Staatsbürgerschaft und ab 1940 zusätzlich die US-amerikanische. Preußischer Staatsangehöriger war er von 1914 bis 1934."

text  = "Einstein gilt als einer der bedeutendsten Physiker der Wissenschaftsgeschichte und weltweit als einer der bekanntesten Wissenschaftler der Neuzeit."
query = "Erstelle einen Titelvorschlag für folgenden Artikel:\n" + text


messages = [
    {"role": "user", "content": query}
]

model_input = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

ft_model = PeftModel.from_pretrained(base_model, model_path)
output = ft_model.generate(model_input, max_new_tokens=1000, do_sample=True, repetition_penalty=1.15)

decoded = tokenizer.batch_decode(output)
print(decoded[0])