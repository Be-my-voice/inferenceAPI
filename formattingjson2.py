# formats json

import json
import re

with open("Untitled-1.json") as file:
    data = json.load(file)


# print(data['results'][1]['landmarks'][0])
# print(type(data['results'][1]['landmarks'][0]))
data2 = data['results']
finalRes = []
for each in data2:
    finalRes.append(each['landmarks'][0])

with open('formattedMobileLandmarks2.json', 'w') as json_file:
    json_file.write(json.dumps(finalRes))

print(len(data2))
print(type(data2[0]))