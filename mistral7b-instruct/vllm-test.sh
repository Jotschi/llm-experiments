#!/bin/bash

#TEXT="Einstein gilt als einer der bedeutendsten Physiker der Wissenschaftsgeschichte und weltweit als einer der bekanntesten Wissenschaftler der Neuzeit."
TEXT="Die Vorbereitungen für das Frequency Festival, das von 14. bis 17. August in St. Pölten über die Bühne gehen wird, laufen auf Hochtouren. Das Rote Kreuz wird in Spitzenzeiten mit bis zu 120 Mitarbeitenden im Einsatz sein, die ÖBB bieten Sonderzüge an. Online seit heute, 9.00 Uhr Laut dem Veranstalter wird das Festival „wie geplant“ stattfinden, allerdings müssen Besucher und Besucherinnen mit längeren Wartezeiten aufgrund genauerer Kontrollen rechnen. Den Auftakt macht Ed Sheeran am 14. August, einem Zusatztag. An den folgenden Tagen werden an der Traisen u. a. Apache 207, The Offspring, RAF Camora, Peter Fox und Cro auftreten. Pro Tag werden um die 50.000 Besucher auf dem VAZ-Gelände in der niederösterreichischen Landeshauptstadt erwartet. Anders als in Vorjahren gibt es (laut Stand von Samstagfrüh) noch Viertagespässe und Tagestickets zu kaufen."

QUERY="Erstelle einen Titelvorschlag für folgenden Artikel:\n$TEXT" 

MODEL_NAME=$(basename $1)

MODEL_CONTAINER_PATH="/models/$MODEL_NAME"

curl http://localhost:10300/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d "{
         \"model\": \"${MODEL_CONTAINER_PATH}\",
         \"messages\": [
             {\"role\": \"user\", \"content\": \"${QUERY}\"}
         ]
     }"