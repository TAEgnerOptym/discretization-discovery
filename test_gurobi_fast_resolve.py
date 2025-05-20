import gurobipy as gp
from gurobipy import GRB
import time
# Set desired solver options

import gurobipy as gp
import gurobipy as gp

def force_integer(model):
    for var in model.getVars():
        if  var.VType != gp.GRB.BINARY and "delta" not in var.VarName:
            var.VType = gp.GRB.INTEGER
    model.update()


import gurobipy as gp
def solve_with_lazy_delta_constraints(model: gp.Model,do_remove,do_bring_back,do_force_integer):
    """
    Solves a MIP where delta constraints are treated as lazy constraints at the root node.
    These are activated after root node processing to preserve cuts and warm start.

    Steps:
    1. Identify and store delta constraints.
    2. Remove them before solving root node.
    3. Re-add them as lazy constraints and solve full MIP.
    """
    #force_integer(model)  # Assumes this function sets VType correctly
    delta_constraints = []
    delta_vars = [v for v in model.getVars() if "delta" in v.VarName]

    # Step 1: Identify delta constraints and store their data
    for constr in model.getConstrs():
        row = model.getRow(constr)
        for i in range(row.size()):
            if "delta" in row.getVar(i).VarName:
                delta_constraints.append(constr)
                constr.Lazy=2
                break
    
    if do_force_integer==True:
        force_integer(model)
    if do_remove==True:
        delta_con_data = []
        for constr in delta_constraints:
            expr = model.getRow(constr)
            rhs = constr.RHS
            sense = constr.Sense
            name = constr.ConstrName
            delta_con_data.append((expr, sense, rhs, name))
            model.remove(constr)  # Must remove *after* storing data
        if do_bring_back==False:
            for var in delta_vars:
                model.remove(var)
    print(f"Marked {len(delta_constraints)} constraints for lazy re-addition.")

    # Step 2: Solve root node only (cutting planes allowed)
    #cut_params = {
    #    "Cuts":             -1,   # –1=auto (default), 0=off, 1=moderate, 2=aggressive
        #"LiftProjectCuts":  2,
        #"ImpliedCuts":      2,
        #"CliqueCuts":       2,
        #"MIRCuts":          2,
        #"StrongCGCuts":     2,
        #"ZeroHalfCuts":     2,
        #"RLTCuts":          2,
        #"RelaxAndLiftCuts": 2,
        #"PSDCuts":          2,
    #}

    #for name, value in cut_params.items():
    #    model.setParam(name, value)

        
    #model.setParam("MIPFocus", 2)
    #model.setParam("Cuts", 0)
    model.setParam("NodeLimit", 100000)        # Only solve root node
    model.setParam("OutputFlag", 1)
    model.update()
    model.optimize()
    input('Done first pass')
    if do_remove==True and do_bring_back==True:
        # Step 3: Re-add delta constraints as LAZY=3
        for expr, sense, rhs, name in delta_con_data:
            if sense == gp.GRB.LESS_EQUAL:
                c = model.addConstr(expr <= rhs, name=name)
            elif sense == gp.GRB.EQUAL:
                c = model.addConstr(expr == rhs, name=name)
            elif sense == gp.GRB.GREATER_EQUAL:
                c = model.addConstr(expr >= rhs, name=name)
            c.Lazy = 3
    model.update()


    # Step 4: Solve full MIP with delta constraints active
    model.setParam("NodeLimit", 100000)
    model.setParam("Cuts", 0)
    model.setParam("ZeroHalfCuts", 0)
    model.setParam("Heuristics", 1)
    model.setParam("MIPFocus", 1)
    model.update()
    model.optimize()

    return model

options = {
        "WLSACCESSID": "8f7bb9d6-8fe5-4349-9dd3-6abbaa9199a0",
        "WLSSECRET": "cb02810a-e0e2-4a1f-8fc0-fd375f65fc65",
        "LICENSEID": 2660300
    }


model_path="R104_super_fine.mps"
#model_path="NO_FANCY_COMPRESS_model_name.mps"
#model_path="../optym_gurobi_file_ILP/C_104_100_10.mps"
#model_path="model_name.mps"

do_remove=True
do_bring_back=False
do_force_integer=False
with gp.Env(params=options) as env:
    with gp.read(model_path, env=env) as model:
        model.setParam("DisplayInterval", 1)
        print('changing options')
        #model.optimize()
        #input('---')
        #model.update()
        solve_with_lazy_delta_constraints(model,do_remove,do_bring_back,do_force_integer)
         

        #model.optimize()