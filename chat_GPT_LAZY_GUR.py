import gurobipy as gp
from gurobipy import GRB

# ---------------------------------------
# Step 1: Lazy constraint callback
# ---------------------------------------
def force_integer(model):
    for var in model.getVars():
        if  var.VType != gp.GRB.BINARY and "delta" not in var.VarName:
            var.VType = gp.GRB.INTEGER
    model.update()
def lazy_delta_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        print('CHECKING')
        input('---')
        for expr, sense, rhs in model._delta_constraints:
            val = model.cbGetSolution(expr)
            violated = (
                (sense == GRB.LESS_EQUAL and val > rhs + 1e-6) or
                (sense == GRB.EQUAL and abs(val - rhs) > 1e-6) or
                (sense == GRB.GREATER_EQUAL and val < rhs - 1e-6)
            )
            if violated:
                print('Adding')
                input('---')
                if sense == GRB.LESS_EQUAL:
                    model.cbLazy(expr <= rhs)
                elif sense == GRB.EQUAL:
                    model.cbLazy(expr == rhs)
                elif sense == GRB.GREATER_EQUAL:
                    model.cbLazy(expr >= rhs)

# ---------------------------------------
# Step 2: Solve function
# ---------------------------------------
def solve_with_lazy_delta_constraints(model: gp.Model):
    """
    Solves the MILP in two phases:
    1. Solve root node LP with aggressive cuts (e.g. zero-half), skipping delta constraints.
    2. Solve full MILP with delta constraints enforced lazily via callback.
    """
    delta_con_data = []
    force_integer(model)
    # Mark delta constraints as Lazy=1

    delta_cons=[]
    for constr in model.getConstrs():
        row = model.getRow(constr)
        for i in range(row.size()):
            if "delta" in row.getVar(i).VarName:
                expr = model.getRow(constr)
                delta_con_data.append((expr, constr.Sense, constr.RHS))
                constr.Lazy = 1
                delta_cons.append(constr)
                print('FOUND')
                break
    do_remove=False
    if do_remove==True:
        for c in delta_cons:
            model.remove(c)

    model._delta_constraints = delta_con_data
    model.setParam("LazyConstraints", 1)

    print(f"Marked {len(delta_con_data)} delta constraints as lazy.")

    # --- Phase 1: Root node solve with aggressive cuts ---
    model.setParam("Cuts", 2)                 # Disable general cuts to isolate zero-half
    #model.setParam("ZeroHalfCuts", 2)         # Aggressive zero-half cut generation
    
    model.setParam("LiftProjectCuts", 2)
    model.setParam("MIRCuts", 2)
    model.setParam("FlowCoverCuts", 2)
    model.setParam("ZeroHalfCuts", 2)

    #model.setParam("ZeroHalfCuts", 2)         # Aggressive zero-half cut generation
    model.setParam("MIPfocus",3)
    model.setParam("NodeLimit", 1)            # Solve only root node
    model.setParam("OutputFlag", 1)
    model.update()
    model.optimize()

    print(f"\nRoot LP relaxation completed. Root bound: {model.ObjBound}")
    input("Press Enter to continue to full MIP solve with lazy delta enforcement...")

    # --- Phase 2: Full MIP solve with all nodes ---
    model.setParam("NodeLimit", 1e9)          # Reset node limit
    model.setParam("MIPfocus",1)
    #model.setParam("Heuristics",0.2)
    model.setParam("Cuts", 0)  
    model.setParam("LiftProjectCuts", 0)
    model.setParam("MIRCuts", 0)
    model.setParam("FlowCoverCuts", 0)
    model.setParam("ZeroHalfCuts", 0)
                   # Disable general cuts to isolate zero-half

    model.optimize(lazy_delta_callback)

    if model.Status == GRB.OPTIMAL:
        print(f"\nFinal MILP solution: Obj = {model.ObjVal}")
    else:
        print(f"\nMILP solve ended with status {model.Status}")

    return model


# ---------------------------------------
# Step 3: Main run with your cloud config
# ---------------------------------------
options = {
    "WLSACCESSID": "8f7bb9d6-8fe5-4349-9dd3-6abbaa9199a0",
    "WLSSECRET": "cb02810a-e0e2-4a1f-8fc0-fd375f65fc65",
    "LICENSEID": 2660300
}

model_path = "R104_super_fine.mps"

with gp.Env(params=options) as env:
    with gp.read(model_path, env=env) as model:
        model.setParam("DisplayInterval", 1)
        print("Solving with lazy delta constraints...")
        solve_with_lazy_delta_constraints(model)
