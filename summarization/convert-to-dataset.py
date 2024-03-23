import json
import os

json_dir_path = 'json'
dataset = []

# Iterate through all files in the directory
for filename in os.listdir(json_dir_path):
    if filename.endswith('.json'):
        with open(os.path.join(json_dir_path, filename), 'r') as f:
            data = json.load(f)
            dataset.append(data)



# Write the combined JSON array to a file
with open('combined.json', 'w') as f:
    json.dump(dataset, f, indent=4)