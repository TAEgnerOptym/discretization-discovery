import gurobipy as gp
from gurobipy import GRB
import time
# Set desired solver options
options = {
        "WLSACCESSID": "b7836a23-3df1-40ac-be4d-310282e2178e",
        "WLSSECRET": "8dd2c11c-cb9b-46f3-b072-4887712ea0c9",
        "LICENSEID": 2690165
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
#model_path="LOOK_ME_model_name.mps"
#model_path="R_104_fany.mps"
#model_path="R_104_just_flip.mps"
#model_path="../optym_gurobi_file_ILP/C_104_50_25_compress.mps"
model_path="Z7_model_name.mps"
model_path="model_name.mps"
with gp.Env(params=options) as env:
    with gp.read(model_path, env=env) as model:
        model.setParam("DisplayInterval", 1)
        bin_set=[]
        counter=0
        if 1>0:
            var_to_orig_type=dict()
            for var in model.getVars():
                var_to_orig_type[var]=var.VType
                var.VType = GRB.CONTINUOUS
            model.optimize()
            obj_target=822.95
            obj_val=model.ObjVal
            min_rc_remove=0.001+(obj_target-model.ObjBound)
            num_count=0
            for var in model.getVars():
                if min_rc_remove<var.RC:
                    var.UB=0
                    num_count=num_count+1
            for var in model.getVars():
                var.VType = var_to_orig_type[var]
            print('num_count')
            print(num_count)
            model.update()
        model.optimize()
        