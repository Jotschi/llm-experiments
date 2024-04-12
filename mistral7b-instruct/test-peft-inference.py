from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

device = "cuda" # the device to load the model onto

model_path = "./newstitle-r64-2k"
base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
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
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token

#query = "Albert Einstein (* 14. März 1879 in Ulm; † 18. April 1955 in Princeton, New Jersey) war ein schweizerisch-US-amerikanischer theoretischer Physiker deutscher Herkunft. Der Wissenschaftler jüdischer Abstammung hatte bis 1896 die württembergische Staatsbürgerschaft, ab 1901 die Schweizer Staatsbürgerschaft und ab 1940 zusätzlich die US-amerikanische. Preußischer Staatsangehöriger war er von 1914 bis 1934."
query = """In Puchberg am Schneeberg ist es am Samstagnachmittag zu einem Waldbrand in unwegsamem Gelände gekommen. Zehn Feuerwehren waren im Einsatz. Die Rauchsäule wurde von einem Flugzeug aus entdeckt und gemeldet.

Online seit gestern, 19.20 Uhr
Gegen 14.30 Uhr wurde die Austro Control darüber informiert, dass von einem Flugzeug aus eine Rauchsäule bei Puchberg am Schneeberg (Bezirk Neunkirchen) gesichtet wurde. Die Koordinaten wurden übermittelt, sodass der Brand im Bereich Rohrbach im Graben lokalisiert werden konnte.

Die Zufahrt gestaltete sich schwierig, da das Waldstück nur über einen schmalen Weg erreichbar ist. „Die schmale Forststraße war nur in eine Richtung befahrbar, und es dauerte rund 25 Minuten, bis wir den Brandherd erreicht hatten“, so Einsatzleiter Gerhard Duchan."""


messages = [
    {"role": "user", "content": "@Title. " + query}
]

model_input = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

ft_model = PeftModel.from_pretrained(base_model, model_path)
output = ft_model.generate(model_input, max_new_tokens=1000, do_sample=True, repetition_penalty=1.15)

decoded = tokenizer.batch_decode(output)
print(decoded[0])