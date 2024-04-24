#!/bin/bash

PORT=10300
curl 127.0.0.1:$PORT/generate_stream \
    -X POST \
    -d '{"inputs":"Erstelle einen Titelvorschlag für folgenden Artikel: Auf dem Rathausplatz dreht sich dieses Wochenende alles ums Rad. Wie jedes Jahr im April findet wieder das ARGUS Bikefestival statt. Das nach eigenen Angaben größte Fahrradfestival Europas gibt etwa einen Überblick über Trends. Auch die Dirt-Jumper zeigen ihr Können.\nOnline seit heute, 0.01 Uhr Das Bikefestival auf dem Rathausplatz gibt es seit mittlerweile 25 Jahren. Bei freiem Eintritt locken weit über hundert Aussteller und viele Programmpunkte. So können Fahrräder getestet oder mitgebrachte Räder bei einer Waschstation gewaschen werden.\nTitelvorschlag:","parameters":{"max_new_tokens":50}}' \
    -H 'Content-Type: application/json'