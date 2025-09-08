import gurobipy as gp
from gurobipy import GRB
import time
import numpy as np
# Set desired solver options
options = {
        "WLSACCESSID": "b7836a23-3df1-40ac-be4d-310282e2178e",
        "WLSSECRET": "8dd2c11c-cb9b-46f3-b072-4887712ea0c9",
        "LICENSEID": 2690165
    }
#model_path="../Optym_gurobi_files_R107_LP/phase_1_file_num8417073.mps"
#model_path="../optym_gurobi_file_ILP/C_104_50_458.mps"
#model_path="../ALL_JSON_BIG/QQbig_30.mps"
#model_path="errHere.mps"
#model_path="tryMe.mps"
#model_path="R_104_fany.mps"
#model_path="R_104_just_flip.mps"
#model_path="../optym_gurobi_file_ILP/C_104_50_25_compress.mps"
model_path="model_name.mps"
with gp.Env(params=options) as env:
    with gp.read(model_path, env=env) as model:
        

        #model.setParam("Method" , 2)
        #model.setParam("Crossover" , 0)        # 2 = Barrier (interior-point)
        #model.setParam("BarConvTol", 1e-4)

        for var in model.getVars():
            if var.Obj!=0:
                var.VType=GRB.BINARY
        model.optimize()
        input('looking here')

        constrs = model.getConstrs()
        # filter to only inequalities (<= or >=)
        ineq_constrs = [c for c in constrs if c.Sense in ("<", ">")]

        # take the last 23
        constrs = model.getConstrs()
    
        # Filter inequality constraints (Sense is '<' or '>')
        ineq_constrs = [c for c in constrs if c.Sense in ("<", ">")]
        K=30
        z=0.001

        # Take last K
        #last_k = ineq_constrs[-K:]
        
        for c in ineq_constrs:
            if abs(c.RHS-np.floor(c.RHS))>0:
                old_rhs = c.RHS
                c.RHS = old_rhs - z
                print(f"Adjusted {c.ConstrName}: {old_rhs} -> {c.RHS}")
        
        # Update the model
        model.update()
        input('hold')
        model.optimize()