import gurobipy as gp
from gurobipy import GRB

# Set desired solver options
options = {
        "WLSACCESSID": "8f7bb9d6-8fe5-4349-9dd3-6abbaa9199a0",
        "WLSSECRET": "cb02810a-e0e2-4a1f-8fc0-fd375f65fc65",
        "LICENSEID": 2660300
    }


# Path to your model file (.lp, .mps, etc.)
model_path = "all_binmodel_name.mps"  # change as needed
model_path = "9000_bin_model_name.mps"  # change as needed
#model_path = "model_name.mps"  # change as needed

# Load model inside customized environment
with gp.Env(params=options) as env:
    with gp.read(model_path, env=env) as model:

        bin_vars=[]
        for var in model.getVars():
            if var.VType == GRB.BINARY:
                bin_vars.append(var)
                var.VType = GRB.CONTINUOUS
                var.LB = 0.0
                var.UB = GRB.INFINITY
        model.update()
        if 1>0:
            model.setParam("OutputFlag", 1)  # Show solver log (optional)
            #model.setParam("SimplexPricing", )
            model.setParam("Cuts", 0)                # Disable all cutting planes
            model.setParam("Heuristics", 0.05)          # Disable primal heuristics
            model.setParam("CutPasses", 0)           # No passes even beyond root
            #model.setParam("Presolve", 2)            # Leave presolve on (it's cheap and useful)
            #model.setParam("NodeMethod", 2)          # Use dual simplex in nodes
            #model.setParam("Method", 1)              # Use dual simplex for LPs
            model.setParam("StartNodeLimit", 1)  # Leave presolve on, it's helpful
            model.setParam("VarBranch", 1) 
        print('LP done')
        model.optimize()
        #input('optimal LP done')
        cur_val=dict()
        #for var in bin_vars:
        #    var.VType = GRB.BINARY
            
        lp_primal_solution = {
            v: v.X for v in bin_vars
        }
        my_counter=1
        for v in bin_vars:
            if  lp_primal_solution[v]>.01 and lp_primal_solution[v]<.99:
                #print(lp_primal_solution[v])
                v.VType = GRB.BINARY
                my_counter=my_counter+1
        #print('len(bin_vars)')
        #print(len(bin_vars))
        #input('--')
        model.update()
        model.optimize()

        # Check result
        if model.Status == GRB.OPTIMAL:
            print(f"Optimal objective: {model.ObjVal}")
        else:
            print(f"Solve ended with status {model.Status}")
