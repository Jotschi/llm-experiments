from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

device = "cuda" # the device to load the model onto

model_path = "./r256-b2-a16-seq4096-l2.0e-5_reformat-v1/checkpoint-500"
base_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=False, load_in_4bit=True
)


base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=False
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)

#query = "Albert Einstein (* 14. März 1879 in Ulm; † 18. April 1955 in Princeton, New Jersey) war ein schweizerisch-US-amerikanischer theoretischer Physiker deutscher Herkunft. Der Wissenschaftler jüdischer Abstammung hatte bis 1896 die württembergische Staatsbürgerschaft, ab 1901 die Schweizer Staatsbürgerschaft und ab 1940 zusätzlich die US-amerikanische. Preußischer Staatsangehöriger war er von 1914 bis 1934."
query = """In Puchberg am Schneeberg ist es am Samstagnachmittag zu einem Waldbrand in unwegsamem Gelände gekommen. Zehn Feuerwehren waren im Einsatz. Die Rauchsäule wurde von einem Flugzeug aus entdeckt und gemeldet.

Online seit gestern, 19.20 Uhr
Gegen 14.30 Uhr wurde die Austro Control darüber informiert, dass von einem Flugzeug aus eine Rauchsäule bei Puchberg am Schneeberg (Bezirk Neunkirchen) gesichtet wurde. Die Koordinaten wurden übermittelt, sodass der Brand im Bereich Rohrbach im Graben lokalisiert werden konnte.

Die Zufahrt gestaltete sich schwierig, da das Waldstück nur über einen schmalen Weg erreichbar ist. „Die schmale Forststraße war nur in eine Richtung befahrbar, und es dauerte rund 25 Minuten, bis wir den Brandherd erreicht hatten“, so Einsatzleiter Gerhard Duchan."""

eval_prompt = """Query:\n\n {} ###\n\n""".format(query)
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

ft_model = PeftModel.from_pretrained(base_model, model_path)
output = ft_model.generate(input_ids=model_input["input_ids"].to(device),
                           attention_mask=model_input["attention_mask"], 
                           max_new_tokens=555, repetition_penalty=1.15)
result = tokenizer.decode(output[0], skip_special_tokens=True).replace(eval_prompt, "")


print(result)