
from full_solver import full_solver
import json

# Replace 'file.json' with the path to your JSON file

data=[]
my_json_file='pre_process/slinklpy_input_clean.json'
with open(my_json_file, 'r') as json_file:
    data = json.load(json_file)
print('data.keys()')
print(data.keys())

for h in data['initGraphNode2AggNode']:
    data['initGraphNode2AggNode'][h] = {k: str(v) for k, v in data['initGraphNode2AggNode'][h].items()}
    
    data['hij2P'][h] = {tuple(k.split('|')): v for k, v in data['hij2P'][h].items()}

my_solver=full_solver(data)


