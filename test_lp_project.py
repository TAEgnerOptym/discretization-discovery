import gurobipy as gp
from gurobipy import GRB
import time
# Set desired solver options
options = {
        "WLSACCESSID": "8f7bb9d6-8fe5-4349-9dd3-6abbaa9199a0",
        "WLSSECRET": "cb02810a-e0e2-4a1f-8fc0-fd375f65fc65",
        "LICENSEID": 2660300
    }
#model_path="../Optym_gurobi_files_R107_LP/phase_1_file_num8417073.mps"
#model_path="../optym_gurobi_file_ILP/C_104_50_458.mps"
#model_path="../ALL_JSON_BIG/QQbig_30.mps"
model_path="LONG_proj.mps"
#model_path="tryMe.mps"
#model_path="R_104_fany.mps"
#model_path="R_104_just_flip.mps"
#model_path="../optym_gurobi_file_ILP/C_104_50_25_compress.mps"
with gp.Env(params=options) as env:
    with gp.read(model_path, env=env) as model:
        
        delta_vars = [v for v in model.getVars() if "Proj" in v.VarName]
       
        #model.setParam("Method", 1)         # Use dual simplex
        #model.setParam("Crossover", 0)      # Skip crossover if using barrier
        model.setParam("Presolve", 2)       # Aggressive presolve
        #model.setParam("ScaleFlag", 1)      # Enable scaling
        model.optimize()
        print('nothing')
        for var in model.getVars():
            if var.X > 0:
                print(f"{var.VarName} = {var.X}")