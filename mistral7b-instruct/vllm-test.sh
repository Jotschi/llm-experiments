#!/bin/bash

TEXT="Einstein gilt als einer der bedeutendsten Physiker der Wissenschaftsgeschichte und weltweit als einer der bekanntesten Wissenschaftler der Neuzeit."
QUERY="Erstelle einen Titelvorschlag für folgenden Artikel:\n$TEXT" 

MODEL="/models/news"

curl http://localhost:10300/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d "{
         \"model\": \"${MODEL}\",
         \"messages\": [
             {\"role\": \"user\", \"content\": \"${QUERY}\"}
         ]
     }"