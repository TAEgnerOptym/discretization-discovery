import gurobipy as gp
options = {
        "WLSACCESSID": "8f7bb9d6-8fe5-4349-9dd3-6abbaa9199a0",
        "WLSSECRET": "cb02810a-e0e2-4a1f-8fc0-fd375f65fc65",
        "LICENSEID": 2660300
    }
    
prune_by_branch_priority=True
model_path="model_name.mps"
with gp.Env(params=options) as env:
    with gp.read(model_path, env=env) as model:
    

        if prune_by_branch_priority: #options.get("prune_by_branch_priority", False):
            # Step 1: Identify variables to remove
            vars_to_remove_jy=[]
            var_names_to_remove=[]
            with open("branch_priorities.txt", "r") as f:
                for line in f:
                    name, bp = line.strip().split()
                    var = model.getVarByName(name)
                    if var:
                        var.setAttr("BranchPriority", int(bp))
                        if int(bp)>52 and var.Obj==0:
                            vars_to_remove_jy.append(var)
                            var_names_to_remove.append(var.VarName)
                        #if int(bp)>1:
                        #    print(['hi+.   '+str(bp)])
            vars_to_remove=vars_to_remove_jy
            # Step 2: Identify constraints to remove (those involving any of the above vars)
            cons_to_remove = []
            for con in model.getConstrs():
                row = model.getRow(con)
                for i in range(row.size()):
                    if row.getVar(i).VarName in var_names_to_remove:
                        cons_to_remove.append(con)
                        break  # no need to check further vars in this constraint

            # Step 3: Remove variables and constraints
            for con in cons_to_remove:
                model.remove(con)
            model.update()

            for var in vars_to_remove:
                model.remove(var)
            model.update()
            print('len(cons_to_remove)')
            print(len(cons_to_remove))
            print('len(vars_to_remove)')
            print(len(vars_to_remove))

        # Step 4: Solve the MILP
        model.optimize()

        if model.status == gp.GRB.OPTIMAL:
            print(f"Optimal objective: {model.ObjVal}")
        else:
            print(f"Optimization ended with status {model.status}")