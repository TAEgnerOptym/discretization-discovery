import gurobipy as gp
from gurobipy import GRB
from itertools import combinations, product
import math

# Gurobi WLS options
options = {
    "WLSACCESSID": "8f7bb9d6-8fe5-4349-9dd3-6abbaa9199a0",
    "WLSSECRET": "cb02810a-e0e2-4a1f-8fc0-fd375f65fc65",
    "LICENSEID": 2660300
}

def build_lp_with_y_vars(model):
    nodes = ['a', 'b', 'c', 'd']

    # Relaxed binary variables
    x = {(i, j): model.addVar(vtype=GRB.BINARY, lb=0.0, ub=1.0, name=f"x_{i}{j}")
         for i in nodes for j in nodes if i != j}

    # y variables bounded between 1 and 3
    y = {i: model.addVar(vtype=GRB.INTEGER,lb=1.0, ub=3.0, name=f"y_{i}") for i in ['a', 'b', 'c']}

    # Objective: minimize transitions involving node d
    obj_terms = ['da', 'db', 'dc', 'ad', 'bd', 'cd']
    model.setObjective(gp.quicksum(x[i, j] for (i, j) in x if f"{i}{j}" in obj_terms), GRB.MINIMIZE)

    # Flow constraints
    model.addConstr(-x['a', 'b'] - x['a', 'c'] - x['a', 'd'] <= -1, "row_a_pos")
    model.addConstr(-x['b', 'a'] - x['b', 'c'] - x['b', 'd'] <= -1, "row_b_pos")
    model.addConstr(-x['c', 'a'] - x['c', 'b'] - x['c', 'd'] <= -1, "row_c_pos")
    model.addConstr(-x['b', 'a'] - x['c', 'a'] - x['d', 'a'] <= -1, "col_a_pos")
    model.addConstr(-x['a', 'b'] - x['c', 'b'] - x['d', 'b'] <= -1, "col_b_pos")
    model.addConstr(-x['a', 'c'] - x['b', 'c'] - x['d', 'c'] <= -1, "col_c_pos")
    model.addConstr(-x['d', 'a'] - x['d', 'b'] - x['d', 'c'] +x['a', 'd'] + x['b', 'd'] +x['c', 'd']<= 0, "d_balance_pos")

    model.addConstr(x['a', 'b'] + x['a', 'c'] + x['a', 'd'] <= 1, "row_a")
    model.addConstr(x['b', 'a'] + x['b', 'c'] + x['b', 'd'] <= 1, "row_b")
    model.addConstr(x['c', 'a'] + x['c', 'b'] + x['c', 'd'] <= 1, "row_c")
    model.addConstr(x['b', 'a'] + x['c', 'a'] + x['d', 'a'] <= 1, "col_a")
    model.addConstr(x['a', 'b'] + x['c', 'b'] + x['d', 'b'] <= 1, "col_b")
    model.addConstr(x['a', 'c'] + x['b', 'c'] + x['d', 'c'] <= 1, "col_c")
    model.addConstr(x['d', 'a'] + x['d', 'b'] + x['d', 'c'] -x['a', 'd'] - x['b', 'd'] -x['c', 'd']<= 0, "d_balance_neg")


    # Time constraints
    #model.addConstr(y['a'] - 1 + 3 * (1 - x['a', 'b']) >= y['b'], "time_ab")
    model.addConstr(y['b']-y['a']+3*x['a', 'b'] <= 2, "time_ab")
    model.addConstr(y['c']-y['a']+3*x['a', 'c'] <= 2, "time_ac")
    model.addConstr(y['c']-y['b']+3*x['b', 'c'] <= 2, "time_bc")
    model.addConstr(y['a']-y['b']+3*x['b', 'a'] <= 2, "time_ba")
    model.addConstr(y['a']-y['c']+3*x['c', 'a'] <= 2, "time_ca")
    model.addConstr(y['b']-y['c']+3*x['c', 'b'] <= 2, "time_cb")

    model.update()

def generate_and_check_floor_cuts(model, max_k=20):
    model.setParam("MIPFocus", 3)
    model.setParam("Cuts", 0)
    model.setParam("MIRCuts", 0)
    model.setParam("FlowCoverCuts", 0)
    model.setParam("ZeroHalfCuts", 2)
    model.setParam("Presolve", 0) 
    model.setParam("NodeLimit", 1) 
    print('below I have teh output with the cutting planes on to solve the MILP')
    model.optimize()
    input('done lp basic ') 
    for v in model.getVars():
        v.VType=GRB.CONTINUOUS
    model.update()
    print('below I turn off integrality enforcment and the cuts so that I can try to find them myself')

    model.optimize()
    #input('hold')
    #if model.Status != GRB.OPTIMAL:
    #    print("Model did not solve to optimality.")
    #    return []

    violated = []
    non_violated=[]
    constrs = model.getConstrs()
    rows = [model.getRow(c) for c in constrs]
    rhs = [c.RHS for c in constrs]
    names = [c.ConstrName for c in constrs]

    for size in range(2, max_k + 1):
        for idxs in combinations(range(len(rows)), size):
            #for signs in product([-0.5, 0.5], repeat=size):
                weighted_row = gp.LinExpr()
                total_rhs = 0
                for idx in idxs:
                    weighted_row += 0.5 * rows[idx]
                    total_rhs += 0.5 * rhs[idx]
                floored_rhs = math.floor(total_rhs + 1e-6)
                #if floored_rhs==total_rhs:
                #    continue
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
                
                #print('floored_expr')
                #print(floored_expr)
                #print('weighted_row')
                #print(weighted_row)
                #print('total_rhs')
                #print(total_rhs)
                #print('floored_rhs')
                #print(floored_rhs)
                #input('---')
                #if floored_rhs<total_rhs:
                #    input('ok thats ogod i want that')

                if lhs_val > floored_rhs + 1e-6:
                    signs=[0.5]*len(idxs)
                    violated.append({
                        "Constraint Combo": " + ".join([f"{s:+.1f}*{names[i]}" for s, i in zip(signs, idxs)]),
                        "Expression": " + ".join([f"{c:+d}*{v}" for v, c in coeffs.items()]),
                        "RHS": floored_rhs,
                        "LHS": round(lhs_val, 6)
                    })
                else:
                    signs=[0.5]*len(idxs)

                    new_term={
                        "Constraint Combo": " + ".join([f"{s:+.1f}*{names[i]}" for s, i in zip(signs, idxs)]),
                        "Expression": " + ".join([f"{c:+d}*{v}" for v, c in coeffs.items()]),
                        "RHS": floored_rhs,
                        "LHS": round(lhs_val, 6)
                    }
                    non_violated.append(new_term)
                    #print('new_term')
                    #print(new_term)
                    #input('--')
    print('len(non_violated)')
    print(len(non_violated))
    print('len(violated)')
    print(len(violated))
    return violated

# Main execution inside the Gurobi Cloud environment
if __name__ == "__main__":
    with gp.Env(params=options) as env:
        with gp.Model("converted_LP", env=env) as model:
            build_lp_with_y_vars(model)
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
