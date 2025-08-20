import gurobipy as gp

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
branch_file="NEW_branch_priorities.txt"

#model_path="C104_50_model_name.mps"
#branch_file="C104_50_branch_priorities.txt"


#model_path="model_name.mps"
#branch_file="branch_priorities.txt"

operator_0=True

with gp.Env(params=options) as env:
    with gp.read(model_path, env=env) as model:
        dict_var_2_type=dict()
        set_terms_kept_binary_or_int=set([])
        model.setParam("Cuts", 0)                # Disable all cutting planes
        #model.setParam("Heuristics", 0.2)          # Disable primal heuristics
        model.setParam("CutPasses", 0)           # No passes even beyond root
        #model.setParam("MIPFocus", 3)
        model.setParam("Presolve", 0)

        with open(branch_file, "r") as f:
            num_keep=3
            orig_keep=num_keep
            for line in f:
                name, bp = line.strip().split()
                var = model.getVarByName(name)
                var.LB=0
                #if int(bp)<10:
                #    var.VType=gp.GRB.CONTINUOUS
                if (num_keep<=0 or  int(bp)<60) and var.VType!=gp.GRB.CONTINUOUS:
                    dict_var_2_type[var]=var.VType
                    if var.VType == gp.GRB.BINARY:
                         var.UB=1
                    var.VType=gp.GRB.CONTINUOUS
                if int(bp)>60 and  var.VType!=gp.GRB.CONTINUOUS and num_keep>0 :
                    var.BranchPriority=int(bp)
                    var.Start=1

                    var.BranchPriority=1000
                    var.VarHintPri=int(1000*(num_keep/orig_keep))
                    var.VarHintVal=1 
                    set_terms_kept_binary_or_int.add(var)
                    num_keep=num_keep-1

        model.update()

        model.optimize()
        input('DONE')
        solution_values = {var.VarName: var.X for var in model.getVars()}
        vars_change=set(model.getVars())-set(dict_var_2_type.keys())

        for var in dict_var_2_type:
            var.VType=dict_var_2_type[var]
        model.update()
        #for var in set_terms_kept_binary_or_int:
        #    var.VarHintPri=1000 #=solution_values[var.VarName]
        #    var.VarHintVal=solution_values[var.VarName]
        #    var.Start=solution_values[var.VarName]
        for var in vars_change:
            
            if var.VType!=gp.GRB.CONTINUOUS:
                #if var.Obj>0:
                #    var.Start=solution_values[var.VarName]

                #var.VarHintPri=0 #=solution_values[var.VarName]
                #var.VarHintVal=round(solution_values[var.VarName])
                print('fixing')
                print(var.VarName)
                print('solution_values[var.VarName]')
                print(solution_values[var.VarName])
                print('moo')
            #model.update()


        print('PART @')

        model.update()
        model.optimize()

        solution_values = {var.VarName: var.X for var in model.getVars()}

        model.reset()
        for var in model.getVars():
            var.Start=solution_values[var.VarName]
        print('resetting ')
        model.update()

        model.optimize()
