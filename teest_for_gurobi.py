import gurobipy as gp
import xpress as xp

options = {
        "WLSACCESSID": "b7836a23-3df1-40ac-be4d-310282e2178e",
        "WLSSECRET": "8dd2c11c-cb9b-46f3-b072-4887712ea0c9",
        "LICENSEID": 2690165
    }
model_path="C_104_model_name.mps"
branch_file="C_104_branch_priorities.txt"


model_path="C104_100_model_name.mps"
branch_file="C104_100_branch_priorities.txt"


model_path="NEW_model_name.mps"
branch_file="NEW_branch_priorities_2.txt"

vars_name_keep_integer=['v60799','v60800','v60801','v60802','v60803','v60804','v60805','v60806','v60807','v60808']

mode_test=2
with gp.Env(params=options) as env:
    with gp.read(model_path, env=env) as model:
        if mode_test>-0.5:
            dict_var_2_type=dict()
            vars_keep_integer=[]
            for k in range(0,mode_test):
                new_var=model.getVarByName(vars_name_keep_integer[k])
                vars_keep_integer.append(new_var)
            set_flip=set(model.getVars())-set(vars_keep_integer)

            for var in set_flip:
                if var.vType==gp.GRB.BINARY:
                    var.Ub=1
                var.vType=gp.GRB.CONTINUOUS
        model.update()
        model.optimize()