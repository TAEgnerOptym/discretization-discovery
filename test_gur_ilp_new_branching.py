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
model_path="model_name.mps"
branch_file="branch_priorities.txt"
mode_1=1
num_keep_given_mode_1=10

model_path="C104_100_model_name.mps"
branch_file="C104_100_branch_priorities.txt"

with gp.Env(params=options) as env:
    with gp.read(model_path, env=env) as model:
        model.setParam("Method", 3)
        model.setParam("MIPFocus",3)
        #model.setParam("VarBranch", -1)
        #model.setParam("BranchDir", 1)
        #model.setParam("Presolve", 0)
        #model.setParam("Heuristics", 0)
        #model.setParam("Cuts", 0)
        model.setParam("LogFile", "mode3.log")

        # Step 1: Identify variables to remove
        #mode 6:  FAST MODE is an LP with a given varible CONTINOUS 1<=x_{g}<=1 
        #mode 7:  FAST MODE is an ILP with a given varible BINARY 1<=x_{g}<=1 
        vars_to_remove_jy=[]
        var_names_to_remove=[]
        if 1>0:
            with open(branch_file, "r") as f:
                my_count=0
                did_find=dict()

                if mode_1 ==6:
                    #set lower bounds for first num_keep_given_mode_1 
                    #   groups to one 
                    # using everything as fractional
                    for line in f:
                        name, bp = line.strip().split()
                        var = model.getVarByName(name)
                        did_find[var]=True
                        var.VType=gp.GRB.CONTINUOUS
                        if int(bp)>60 and my_count<=num_keep_given_mode_1:
                            var.LB=1
                            var.UB=1
                            my_count=my_count+1
                if mode_1 ==7:
                    #set lower bounds for first num_keep_given_mode_1
                    #  groups to one using everything ELSE fractional but keeeping that term as binary
                    #we still one binary varaible butits LB is one so not really binary
                    for line in f:
                        name, bp = line.strip().split()
                        var = model.getVarByName(name)
                        did_find[var]=True
                        if int(bp)>60 and my_count<=num_keep_given_mode_1:
                            var.LB=1
                            var.UB=1
                            my_count=my_count+1
                        else:
                            var.VType=gp.GRB.CONTINUOUS

                if mode_1==0: 
                    for var in model.getVars():
                        var.VType=gp.GRB.CONTINUOUS
                if mode_1==1:
                    for line in f:
                        name, bp = line.strip().split()
                        var = model.getVarByName(name)
                        did_find[var]=True
                        #var.VType=gp.GRB.CONTINUOUS

                        if int(bp)<60 or my_count>=num_keep_given_mode_1:
                            var.VType=gp.GRB.CONTINUOUS
                            #var.setAttr("BranchPriority",0)
                            
                        else:
                            var.setAttr("BranchPriority",1000)
                            my_count=my_count+1
                            #var.UB=0
                            #var.LB=1
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

