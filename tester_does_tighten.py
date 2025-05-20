import gurobipy as gp
from gurobipy import GRB
import time
import random
# Set desired solver options
options = {
        "WLSACCESSID": "8f7bb9d6-8fe5-4349-9dd3-6abbaa9199a0",
        "WLSSECRET": "cb02810a-e0e2-4a1f-8fc0-fd375f65fc65",
        "LICENSEID": 2660300
    }
file_1='T1_no_fancy.mps'
file_2='T1_fancy.mps'
file_2='idea_fancy_with_reset.mps'
#file_2='much_denser.mps'
#file_2='NG_25.mps'
#file_2='../fancy_compare/NO_FANCY_COMPRESS_model_name.mps'

K=2


my_hist=[]
my_hist_time=[]
for iter in range(0,20):
    M1_var_name_2_to_val=dict()
    M1_var_name_2_to_obj_coeff=dict()
    M2_var_name_2_to_obj_coeff=dict()
    val1=-1
    val2=-1
    selected_names_1=[]
    selected_names_2=[]
    fractional_var_names_1=[]
    fractional_var_names_2=[]
    t1=0
    t2=0
    t3=0
    t4=0
    with gp.Env(params=options) as env:
        with gp.read(file_1, env=env) as model:
            act_vars = [v for v in model.getVars() if v.VarName.startswith("act_")]
            for v in model.getVars():
                if v.VType == GRB.BINARY:
                    v.VType = GRB.CONTINUOUS
            t1=time.time()
            model.optimize()
            t1=time.time()-t1
            M1_var_name_2_to_val = {v.VarName: v.X for v in act_vars}
            M1_var_name_2_to_obj_coeff = {v.VarName: v.Obj for v in act_vars}
            val1=model.ObjVal

            fractional_var_names_1 = [v.VarName for v in act_vars if 0.01 < v.X < 0.99]

            selected_names_1 = random.sample(fractional_var_names_1, min(K, len(fractional_var_names_1)))

            print('len(act_vars)')
            print(len(act_vars))

    M2_var_name_2_to_val=dict()

    with gp.Env(params=options) as env:
        with gp.read(file_2, env=env) as model:
            act_vars = [v for v in model.getVars() if v.VarName.startswith("act_")]
            for v in model.getVars():
                if v.VType == GRB.BINARY:
                    v.VType = GRB.CONTINUOUS
            t2=time.time()
            model.optimize()
            t2=time.time()-t2
            val2=model.ObjVal
            M2_var_name_2_to_val = {v.VarName: v.X for v in act_vars}
            M2_var_name_2_to_obj_coeff = {v.VarName: v.Obj for v in act_vars}

            fractional_var_names_2 = [v.VarName for v in act_vars if 0.01 < v.X < 0.99]

            selected_names_2 = random.sample(fractional_var_names_2, min(K, len(fractional_var_names_1)))

            print('len(act_vars)')
            print(len(act_vars))

    print('val1:  '+str(val1))
    print('val2:  '+str(val2))

    print('selected_names_2')
    print(selected_names_2)
    print('selected_names_1')
    print(selected_names_1)

    counter_1=0
    with gp.Env(params=options) as env:
        with gp.read(file_1, env=env) as model:
            act_vars_3 = [v for v in model.getVars() if v.VarName.startswith("act_")]
            for v in model.getVars():
                if v.VType == GRB.BINARY and v.VarName not in selected_names_1 and v.VarName not in selected_names_2:
                    v.VType = GRB.CONTINUOUS
                    counter_1=counter_1+1
            t3=time.time()
            model.optimize()
            t3=time.time()-t3
            val3=model.ObjVal

    val4=0
    counter_2=0
    with gp.Env(params=options) as env:
        with gp.read(file_2, env=env) as model:
            act_vars_4 = [v for v in model.getVars() if v.VarName.startswith("act_")]
            for v in model.getVars():
                if v.VType == GRB.BINARY and v.VarName not in selected_names_1 and v.VarName not in selected_names_2:
                    v.VType = GRB.CONTINUOUS
                    counter_2=counter_2+1
            t4=time.time()
            model.optimize()
            t4=time.time()-t4
            val4=model.ObjVal

        if counter_2!=counter_1:
            input('error ')
    print()
    print('val3')
    print(val3)
    print('val4')
    print(val4)

    my_hist.append(tuple([val3,val4]))
    my_hist_time.append(tuple([t1,t2,t3,t4]))
    print('my_hist')
    print(my_hist)
    print('my_hist_time')
    print(my_hist_time)

    print('---')