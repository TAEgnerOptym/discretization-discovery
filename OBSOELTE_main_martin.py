
from full_solver import full_solver
import json

# Replace 'file.json' with the path to your JSON file

data=[]
my_json_file='test.json'
with open(my_json_file, 'r') as json_file:
    data = json.load(json_file)
for h in data['initGraphNode2AggNode']:
    data['initGraphNode2AggNode'][h] = {k: str(v) for k, v in data['initGraphNode2AggNode'][h].items()}

print('data.keys()')
print(data.keys())
print('self.hij_2_P.keys()')
print(data['hij2P'].keys())
my_solver=full_solver(data)


