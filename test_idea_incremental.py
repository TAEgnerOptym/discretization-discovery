import gurobipy as gp
import xpress as xp


def get_dict(set_terms_kept_binary_or_int):

    dual_lb_dict=dict()
    for var in set_terms_kept_binary_or_int:
        # Check if variable is at lower bound
        if abs(var.X - var.LB) < 1e-6:
            # Then reduced cost is dual on LB (if > 0)
            dual_lb = var.RC
        else:
            # Not at lower bound — dual on LB is zero
            dual_lb = 0.0

        dual_lb_dict[var.VarName] = dual_lb
    return dual_lb_dict

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

        
        #model.setParam("Cuts", 0)                # Disable all cutting planes
        #model.setParam("Heuristics", 0.2)          # Disable primal heuristics
        #model.setParam("CutPasses", 0)           # No passes even beyond root
        #model.setParam("MIPFocus", 3)
        #model.setParam("Presolve", 0)
        #model.setParam("Method", 2)

        with open(branch_file, "r") as f:
            num_keep=100
            orig_keep=num_keep
            for line in f:
                name, bp = line.strip().split()
                var = model.getVarByName(name)
                var.LB=0
                
                if (num_keep<=0 or int(bp)<60) and var.VType!=gp.GRB.CONTINUOUS:
                    dict_var_2_type[var]=var.VType
                    if var.VType == gp.GRB.BINARY:
                         var.UB=1
                    var.VType=gp.GRB.CONTINUOUS
                if int(bp)>60 and  var.VType!=gp.GRB.CONTINUOUS and num_keep>0 :
                    #var.BranchPriority=int(bp)
                    var.Start=1
                    set_terms_kept_binary_or_int.add(var)
                    num_keep=num_keep-1
                    var.LB=1
                    var.UB=1
                    var.VType=gp.GRB.CONTINUOUS #CHATGPT FOCUS:  If I remove this line the code runs much much slower.  but note above that its LB 
        print('step0')
        model.update()
        model.optimize()
        dual_lb_dict=get_dict(set_terms_kept_binary_or_int)
        print('dual_lb_dict')
        print(dual_lb_dict)
        for var in set_terms_kept_binary_or_int:
            var.LB=0
        k=4
        top_k_varnames = sorted(
            dual_lb_dict,
            key=lambda name: dual_lb_dict[name],
            reverse=True
        )[:k]

        for name in top_k_varnames:
            var = model.getVarByName(name)
            if var is None:
                raise ValueError(f"Variable {name} not found in model.")

            print(f"Updating variable {name} with dual value {dual_lb_dict[name]}")
            var.VType = gp.GRB.BINARY
            var.LB = 0
            var.UB = 1

        # Apply all changes
        model.update()
        model.optimize()