import gurobipy as gp
from gurobipy import GRB
options = {
"WLSACCESSID":"8f7bb9d6-8fe5-4349-9dd3-6abbaa9199a0",
"WLSSECRET":"cb02810a-e0e2-4a1f-8fc0-fd375f65fc65",
"LICENSEID":2660300
}
with gp.Env(params=options) as env:
    # Pass this ENV symbol above to any function that needs to construct an LP
    # Any time you're building and solving an LP model, it needs to be within a
    # context manager of the form below
    with gp.Model(env=env) as model:
        # Do stuff

        # Add variables
        x = model.addVar(vtype=GRB.INTEGER, name="x")
        y = model.addVar(vtype=GRB.INTEGER, name="y")

        # Set objective: Maximize 3x + 2y
        model.setObjective(3 * x + 2 * y, GRB.MAXIMIZE)

        # Add constraints
        model.addConstr(2 * x + y <= 4, "c1")
        model.addConstr(x + 2 * y <= 5, "c2")

        # Optimize the model
        model.optimize()

        # Print solution
        if model.status == GRB.OPTIMAL:
            print(f"Optimal objective value: {model.objVal}")
            print(f"x = {x.X}, y = {y.X}")
        else:
            print("No optimal solution found.")