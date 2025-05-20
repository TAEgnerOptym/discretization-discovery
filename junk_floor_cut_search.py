import gurobipy as gp
from gurobipy import GRB
from itertools import combinations, product
import math

def build_lp_with_y_vars():
    model = gp.Model("floor_cut_LP")
    model.setParam("OutputFlag", 0)

    nodes = ['a', 'b', 'c', 'd']

    # Relaxed binary variables
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

    # Time window constraints
    model.addConstr(y['a'] - 1 + 3 * (1 - x['a', 'b']) >= y['b'], "time_ab")
    model.addConstr(y['a'] - 1 + 3 * (1 - x['a', 'c']) >= y['c'], "time_ac")
    model.addConstr(y['b'] - 1 + 3 * (1 - x['b', 'a']) >= y['a'], "time_ba")
    model.addConstr(y['b'] - 1 + 3 * (1 - x['b', 'c']) >= y['c'], "time_bc")
    model.addConstr(y['c'] - 1 + 3 * (1 - x['c', 'a']) >= y['a'], "time_ca")
    model.addConstr(y['c'] - 1 + 3 * (1 - x['c', 'b']) >= y['b'], "time_cb")

    model.update()
    return model

def generate_and_check_floor_cuts(model, max_k=4):
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        print("Model did not solve to optimality.")
        return []

    violated = []
    constrs = model.getConstrs()
    rows = [model.getRow(c) for c in constrs]
    rhs = [c.RHS for c in constrs]
    names = [c.ConstrName for c in constrs]

    for size in range(2, max_k + 1):
        for idxs in combinations(range(len(rows)), size):
            for signs in product([-0.5, 0.5], repeat=size):
                weighted_row = gp.LinExpr()
                total_rhs = 0
                for sign, idx in zip(signs, idxs):
                    weighted_row += sign * rows[idx]
                    total_rhs += sign * rhs[idx]

                # Floor coefficients and RHS
                floored_expr = gp.LinExpr()
                coeffs = {}
                for i in range(weighted_row.size()):
                    var = weighted_row.getVar(i)
                    coeff = weighted_row.getCoeff(i)
                    floored_coeff = math.floor(coeff + 1e-6)
                    if abs(floored_coeff) > 1e-6:
                        coeffs[var.VarName] = floored_coeff
                        floored_expr.addTerms(floored_coeff, var)

                floored_rhs = math.floor(total_rhs + 1e-6)
                lhs_val = floored_expr.getValue()

                if lhs_val > floored_rhs + 1e-6:
                    violated.append({
                        "Constraint Combo": " + ".join([f"{s:+.1f}*{names[i]}" for s, i in zip(signs, idxs)]),
                        "Expression": " + ".join([f"{c:+d}*{v}" for v, c in coeffs.items()]),
                        "RHS": floored_rhs,
                        "LHS": round(lhs_val, 6)
                    })

    return violated

def main():
    model = build_lp_with_y_vars()
    cuts = generate_and_check_floor_cuts(model, max_k=4)

    if not cuts:
        print("No violated floor-based cuts found.")
    else:
        print(f"\nFound {len(cuts)} violated floor-based cuts:")
        for cut in cuts:
            print(f"\n---")
            print(f"From combination: {cut['Constraint Combo']}")
            print(f"Expression: {cut['Expression']}")
            print(f"LHS = {cut['LHS']} > RHS = {cut['RHS']}")

if __name__ == "__main__":
    main()
