import json,os

with open('./secret.json', 'r') as f:
    json_data = json.load(f)
apikey = json_data['openAI']

print(apikey)
