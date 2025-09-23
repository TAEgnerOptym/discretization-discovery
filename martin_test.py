import gurobipy as gp
from gurobipy import GRB
import time
# Set desired solver options
options = {
        "WLSACCESSID": "b7836a23-3df1-40ac-be4d-310282e2178e",
        "WLSSECRET": "8dd2c11c-cb9b-46f3-b072-4887712ea0c9",
        "LICENSEID": 2690165
    }

model_path="model_name.mps"
with gp.Env(params=options) as env:
    with gp.read(model_path, env=env) as model:
        model.optimize()    
        

import pulp

# Assuming 'my_model.mps' is in the same directory as your script
file_path = model_path

# Read the MPS file, assuming a minimization problem
variables_dict, model = pulp.LpProblem.fromMPS(file_path, sense=pulp.LpMinimize)

# Solve the problem
model.solve()

# Print the solution status and objective value
print(f"Status: {pulp.LpStatus[model.status]}")
print(f"Objective Value: {model.objective.value()}")

# Print the optimal values of the variables
print("Variable Values:")
for var_name, var_obj in variables_dict.items():
    print(f"  {var_name}: {var_obj.value()}")