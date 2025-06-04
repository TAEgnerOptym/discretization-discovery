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
model_path="../optym_gurobi_file_ILP/C_104_25_10.mps"
model_path="../optym_gurobi_file_ILP/C_104_50_25.mps"
model_path="../optym_gurobi_file_ILP/R_104_50.mps"
model_path="../optym_gurobi_file_ILP/R_104_50.mps"
model_path="../optym_gurobi_file_ILP/C_104_100_10.mps"
model_path="FANCY_COMPRESS_model_name.mps"
model_path="NO_FANCY_COMPRESS_model_name.mps"
model_path="R_104_MPS_NO_FANCY.mps"
model_path="R_104_NO_FancY_2.mps"
#model_path="R_104_MPS_NO_FANCY_2.mps"
model_path="R104_super_fine.mps"
#model_path="tryMe.mps"
model_path="LOOK_ME_model_name.mps"
#model_path="R_104_fany.mps"
#model_path="R_104_just_flip.mps"
#model_path="../optym_gurobi_file_ILP/C_104_50_25_compress.mps"
with gp.Env(params=options) as env:
    with gp.read(model_path, env=env) as model:
        model.setParam("DisplayInterval", 1)
        bin_set=[]
        counter=0
        for var in model.getVars():

            if abs(var.Obj)<0.01:# and  == GRB.BINARY:
                var.VType = GRB.CONTINUOUS
                print(var.Obj)
                counter=counter+1
                #bin_set.append(var)
        print('counter')
        print(counter)
        input('---')
        print('len(bin_set)')
        print(len(bin_set))
        model.update()
        if 1<0: 
            print('changing options')
            model.setParam("Cuts", 0)                # Disable all cutting planes
            #model.setParam("Heuristics", 0.2)          # Disable primal heuristics
            #model.setParam("CutPasses", 0)           # No passes even beyond root
            model.setParam("MIPFocus", 3)
            
            model.setParam("Cuts", 0)

            model.setParam("MIRCuts", 0)
            model.setParam("FlowCoverCuts", 0)
            model.setParam("ZeroHalfCuts", 2)


            model.update()
        
         

        model.optimize()