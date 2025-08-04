import gurobipy as gp

options = {
        "WLSACCESSID": "b7836a23-3df1-40ac-be4d-310282e2178e",
        "WLSSECRET": "8dd2c11c-cb9b-46f3-b072-4887712ea0c9",
        "LICENSEID": 2690165
    }
#model_path="RC103_model_name.mps"
#branch_file="RC103_branch_priorities.txt"
#model_path="C104_model_name.mps"
#branch_file="C104_branch_priorities.txt"
model_path="C104_new_model_name.mps"
branch_file="C104_new_branch_priorities.txt"
mode_1=1
with gp.Env(params=options) as env:
    with gp.read(model_path, env=env) as model:
        #model.setParam("Method", 3)
        #model.setParam("VarBranch", -1)
        #model.setParam("BranchDir", 1)
        #model.setParam("Presolve", 0)
        #model.setParam("Heuristics", 0)
        #model.setParam("Cuts", 0)
        model.setParam("LogFile", "mode3.log")

        # Step 1: Identify variables to remove
        vars_to_remove_jy=[]
        var_names_to_remove=[]
        if 1>0:
            with open(branch_file, "r") as f:
                my_count=0
                did_find=dict()
                if mode_1==1:
                    for line in f:
                        name, bp = line.strip().split()
                        var = model.getVarByName(name)
                        did_find[var]=True
                        if int(bp)<60 or my_count>=1:
                            var.VType=gp.GRB.CONTINUOUS
                            #var.setAttr("BranchPriority",0)
                        else:
                            var.setAttr("BranchPriority",1000)
                            my_count=my_count+1
                    print('my_count')
                    print(my_count)
                    model.update()
                    
                if mode_1==2:
                    for line in f:
                        name, bp = line.strip().split()
                        var = model.getVarByName(name)
                        var.setAttr("BranchPriority", int(bp))
                if mode_1==3:
                    for line in f:
                        name, bp = line.strip().split()
                        var = model.getVarByName(name)
                        
                        if int(bp)<60: #or my_count>3:
                            var.setAttr("BranchPriority", int(0))
                        else:
                            var.setAttr("BranchPriority", int(1000))
                            my_count=my_count+1
                    print('my_count')
                    print(my_count)
                    model.update()
        
        #for var in model.getVars():
        #    if var.VType!=gp.GRB.CONTINUOUS:
        #        print('var.VarName')
        #        print(var.VarName)
        ##        print('var.BranchPriority')
         #       print(var.BranchPriority)
         #       print('var.VType')
         #       print(var.VType)
         #       input('--')
        # Step 4: Solve the MILP
        model.optimize()

        if model.status == gp.GRB.OPTIMAL:
            print(f"Optimal objective: {model.ObjVal}")
        else:
            print(f"Optimization ended with status {model.status}")

