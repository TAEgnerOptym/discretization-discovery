import gurobipy as gp
from gurobipy import GRB
from itertools import combinations

def build_lp_with_y_vars():
    model = gp.Model("zero_half_with_y")
    model.setParam("OutputFlag", 0)

    nodes = ['a', 'b', 'c', 'd']

    # Binary x variables relaxed to continuous [0,1]
    x = {(i, j): model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name=f"x_{i}{j}")
         for i in nodes for j in nodes if i != j}

    # y variables bounded between 1 and 3
    y = {i: model.addVar(lb=1.0, ub=3.0, name=f"y_{i}") for i in ['a', 'b', 'c']}

    # Objective: minimize transitions involving node d
    obj_terms = ['da', 'db', 'dc', 'ad', 'bd', 'cd']
    model.setObjective(gp.quicksum(x[i, j] for (i, j) in x if f"{i}{j}" in obj_terms), GRB.MINIMIZE)

    # Flow constraints
    model.addConstr(x['a', 'b'] + x['a', 'c'] + x['a', 'd'] == 1, "row_a")
    model.addConstr(x['b', 'a'] + x['b', 'c'] + x['b', 'd'] == 1, "row_b")
    model.addConstr(x['c', 'a'] + x['c', 'b'] + x['c', 'd'] == 1, "row_c")
    model.addConstr(x['b', 'a'] + x['c', 'a'] + x['d', 'a'] == 1, "col_a")
    model.addConstr(x['a', 'b'] + x['c', 'b'] + x['d', 'b'] == 1, "col_b")
    model.addConstr(x['a', 'c'] + x['b', 'c'] + x['d', 'c'] == 1, "col_c")
    model.addConstr(x['d', 'a'] + x['d', 'b'] + x['d', 'c'] == x['a', 'd'] + x['b', 'd'] + x['c', 'd'], "d_balance")

    # Time constraints
    model.addConstr(y['a'] - 1 + 3 * (1 - x['a', 'b']) >= y['b'], "time_ab")
    model.addConstr(y['a'] - 1 + 3 * (1 - x['a', 'c']) >= y['c'], "time_ac")
    model.addConstr(y['b'] - 1 + 3 * (1 - x['b', 'a']) >= y['a'], "time_ba")
    model.addConstr(y['b'] - 1 + 3 * (1 - x['b', 'c']) >= y['c'], "time_bc")
    model.addConstr(y['c'] - 1 + 3 * (1 - x['c', 'a']) >= y['a'], "time_ca")
    model.addConstr(y['c'] - 1 + 3 * (1 - x['c', 'b']) >= y['b'], "time_cb")

    model.update()
    return model

def find_violated_zero_half_cuts(model):
    violated = []
    constrs = model.getConstrs()
    rows = [model.getRow(c) for c in constrs]
    rhs = [c.RHS for c in constrs]
    names = [c.ConstrName for c in constrs]

    for (i, j) in combinations(range(len(rows)), 2):
        row_sum = rows[i] + rows[j]
        new_expr = gp.LinExpr()
        valid = True
        for k in range(row_sum.size()):
            coeff = row_sum.getCoeff(k)
            if coeff in {0.0, 1.0, 2.0}:
                new_expr.addTerms(0.5 * coeff, row_sum.getVar(k))
            else:
                valid = False
                break
        if not valid:
            continue

        cut_rhs = int((rhs[i] + rhs[j]) // 2)
        lhs_val = new_expr.getValue()
        if lhs_val > cut_rhs + 1e-6:
            violated.append((f"{names[i]} + {names[j]}", lhs_val, cut_rhs))

    return violated

def run_cutting_plane_test():
    model = build_lp_with_y_vars()
    model.optimize()

    print(f"\nInitial LP Objective: {model.ObjVal:.4f}\n")
    violated = find_violated_zero_half_cuts(model)

    if not violated:
        print("No violated zero-half cuts found.")
    else:
        print("Violated zero-half cuts:")
        for name, lhs, rhs in violated:
            print(f"  {name}: LHS = {lhs:.3f} > RHS = {rhs}")

if __name__ == "__main__":
    run_cutting_plane_test()
