import re
import json

# Load the contents of the JSON file
with open('mobileLandmarks.json', 'r') as file:
    json_text = file.read()

# Use regular expressions to extract data from angle brackets
pattern = r'<(.*?)>'
matches = re.findall(pattern, json_text)
# print(matches)
# print(type(matches))

values = []

for match in matches:
    l1 = match[0]
    if(l1 == 'N'):
        pattern2 = r'\((.*?)\)'
        m = re.findall(pattern2, match)
        values.append(m[0].split(' '))
        # print(m, len(m))
        # print(match)

# print(values)

farray = []

for v in values:
    sfarray = {}
    sfarray["x"] = float(v[0].split('=')[1])
    sfarray["y"] = float(v[1].split('=')[1])
    sfarray["z"] = float(v[2].split('=')[1])
    farray.append(sfarray)
    # print(sfarray)

# print(farray)
# print(len(farray))

gfarray = []
j = 1
temp = []
for i in farray:
    temp.append(i)
    if j % 33 == 0:
        print(temp, len(temp))
        gfarray.append(temp)
        temp = []
    j += 1

print(gfarray)
print(len(gfarray))

jData = json.dumps(gfarray)
with open('formattedMobileLandmarks.json', 'w') as json_file:
    json_file.write(jData)

# Create a dictionary from the extracted data
# data_dict = {}
# for match in matches:
#     key, value = match.split(':')  # Assuming data is in key:value format
#     data_dict[key.strip()] = value.strip()

# # Convert the dictionary to a JSON string
# json_data = json.dumps(data_dict, indent=4)

# # Print the resulting JSON data
# print(json_data)