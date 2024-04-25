import ollama
import json
import uuid
import random
import re

model="phi3"
#model="dolphin-mixtral:v2.7"


# Gib 20 deutsche Wörter für einen kindgerechte Abenteuer Geschichte von einem kleinen Astronauten aus. Beispiele: Weltraum, Raumschiff, Planet, Orbit, Außerirdischer. Gib nur das Wort aus und keine Beschreibung.
# Gib 20 deutsche Wörter für einen kindgerechte Abenteuer Geschichte aus. Gib nur das Wort aus und keine Beschreibung.
# Gib 20 deutsche Adjektive für eine kindgerechte Abenteuer Geschichte von kleinen Astronauten aus. Umschreibe komplexe Wörter in ein für Kinder verständliches weise. Gib sie im JSON Format aus.
# Gib 40 deutsche Verben für eine kindgerechte Abenteuer Geschichte von kleinen Astronauten aus. Umschreibe komplexe Wörter in ein für Kinder verständliches weise. Gib sie im JSON Format aus.
# Zähle mindestens 40 deutsche Themen auf die man für eine kindgerechte Abenteuer Geschichte von einem kleinen Astronauten nutzen könnte. Gib nur ein Wort pro Thema aus. Umschreibe komplexe Themen in ein für Kinder verständliches Format. Gib sie im JSON Format aus.


with open('input/entities_space.json', 'r') as f:
  entities_space = json.load(f)
  
with open('input/entities_no_space.json', 'r') as f:
  entities_no_space = json.load(f)

with open('input/adjectives.json', 'r') as f:
  adjectives = json.load(f)

with open('input/verbs.json', 'r') as f:
  verbs = json.load(f)

with open('input/topics.json', 'r') as f:
  topics = json.load(f)



def inferStory(topic, word, space_word, verb, adjective_1, adjective_2):
    prompt =  'Du bist ein Author von Kindergeschichten vom kleinen Astronauten. Schreibe eine Abendetuer Geschichte die das Theme ' + topic + ' enthält. Verzichte auf eine Einleitung, Titlel am Beginn der Geschichte.\n'
    prompt += 'Gib nur die Geschichte aus. Gib nicht den Author aus. Erwähne dich nicht. Keine Überschrift. Schreibe eine abendteuer Weltraum Geschichte.\n' 
    prompt += 'Umschreibe komplizierte Wörter so das sie von Kindern verstanden werden können.\n'
    prompt += 'Hier eine Inspirationshilfe: ' + adjective_1 + " " + word + " " + verb + " " + adjective_2 + " " + space_word 
    #print(prompt)
    #print("----------------")
    response = ollama.chat(model=model, messages=[
    {
        'role': 'user',
        'content': prompt,
    },
    ])
    return response['message']['content'].strip()

def generate():
    verb  = random.choice(verbs)
    topic = random.choice(topics)
    word  = random.choice(entities_no_space)
    space_word  = random.choice(entities_space)
    adjective_1   = random.choice(adjectives)
    adjective_2   = random.choice(adjectives)   
    story = inferStory(topic, word, space_word, verb, adjective_1, adjective_2)
    #print(story)
    dic = {
        "text": story,
        "topic": topic,
        "word_1": space_word,
        "adjective_1": adjective_1,
        "verb": verb,
        "word_2": word,
        "adjective_2": adjective_2
    }
    
    json_object = json.dumps(dic, indent=4)
    story_uuid = str(uuid.uuid4())
    filename = story_uuid + ".json"
    
    with open("json/" + filename, "w") as outfile:
        outfile.write(json_object)

if __name__ == "__main__":
    total=10_000
    for x in range(total):
        print(str(x) + " of " + str(total))
        generate()