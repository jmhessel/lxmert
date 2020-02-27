import json

with open('minival_predict.json') as f:
    data = json.load(f)

print(len(data))
