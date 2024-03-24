import ollama
import json
import uuid
import random
import re

#print(ollama.list())
model="dolphin-mixtral:v2.7"

# Gib 20 nomen für einen nachrichten artikel mit dem thema Luftverschmutzung aus.
# Gib 20 adjektive für einen nachrichten artikel mit dem thema Luftverschmutzung aus.


# Topics: zähle mindestens 50 themen bereiche auf die in nachrichten abgebildet werden. Verzichte auf eine beschreibung.
news_topics = [
    "Politik",
    "Wirtschaft",
    "Gesellschaft",
    "Wissenschaft und Technologie",
    "Umwelt",
    "Gesundheit",
    "Sport",
    "Medien und Unterhaltung",
    "Reisen und Tourismus",
    "Wetter und Naturkatastrophen",
    "Bildung",
    "Religion",
    "Justiz und Recht",
    "Landwirtschaft",
    "Krieg und Frieden",
    "Architektur und Stadtentwicklung",
    "Kunst und Kultur",
    "Arbeitsmarkt",
    "Verkehr und Infrastruktur",
    "Energie und Rohstoffe",
    "Immobilien",
    "Kriminalität",
    "Verbraucherschutz",
    "Raumfahrt",
    "Politische Parteien",
    "Internationale Beziehungen",
    "Lokalnachrichten",
    "Nationalnachrichten",
    "Auslandsnachrichten",
    "Politische Entscheidungen",
    "Steuern und Abgaben",
    "Soziale Medien",
    "Datenschutz",
    "Geopolitik",
    "Menschenrechte",
    "Tierschutz",
    "Bildungspolitik",
    "Wissenschaftspolitik",
    "Forschung und Entwicklung",
    "Innovation und Start-ups",
    "Klimawandel",
    "Nachhaltigkeit",
    "Umweltschutz",
    "Abfallwirtschaft",
    "Wasserwirtschaft",
    "Luftverschmutzung",
    "Arzneimittel",
    "Impfungen",
    "Öffentliche Gesundheit",
    "Sucht und Drogen"
]

def inferArtikel(topic):
    response = ollama.chat(model=model, messages=[
    {
        'role': 'user',
        'content': 'Du bist ein Nachrichten Redakteur. Schreibe einen Artikel für eine Deutsche Zeitung für das Thema ' + topic + '. Verzichte auf eine Einleitung, Titlel Zeile am Beginn des Artikels. Gib nur den Arikel aus und keine Antwort oder einen Hinweis Ende. Gib nicht den Author aus. Erwähne dich nicht. Keine Überschrift.',
    },
    ])
    return response['message']['content'].strip()

def inferTitles(topic, article):
    response = ollama.chat(model=model, messages=[
    {
        'role': 'user',
        'content': 'Du bist ein Nachrichten Redakteur. Schreibe einen Artikel für eine Deutsche Zeitung für das Thema ' + topic + '. Verzichte auf eine Einleitung, Titlel Zeile am Beginn des Artikels. Gib nur den Arikel aus und keine Antwort oder einen Hinweis Ende. Gib nicht den Author aus. Erwähne dich nicht. Keine Überschrift.',
    },
    {
        'role': 'assistant',
        'content': article,
    },
    {
        'role': 'user',
        'content': 'Schreibe 20 Titelvorschläge unterschiedlicher länge für den Aritkel.',
    },
    ])
    
    text = response['message']['content'].strip()
    array = re.split("\n\d+\.", text)
    titles = [line.strip().strip('"').strip("1. ") for line in array]
    return titles


def main():
    topic = random.choice(news_topics)
    print("Using: " + topic)
    article = inferArtikel(topic)
    titles = inferTitles(topic, article)
    dic = {
        "text": article,
        "titles": titles
    }

    json_object = json.dumps(dic, indent=4)
    article_uuid = str(uuid.uuid4())
    filename = article_uuid + ".json"

#    with open(article_uuid +"_article.txt", "w") as text_file:
#        text_file.write(article)
#
#    with open(article_uuid +"_titles.txt", "w") as text_file:
#        text_file.write(titles)

    with open("json/" + filename, "w") as outfile:
        outfile.write(json_object)

if __name__ == "__main__":
    main()