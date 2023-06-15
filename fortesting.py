import json

with open("landmarks.json") as file:
    data = json.load(file)

print(type(data))
print(data)