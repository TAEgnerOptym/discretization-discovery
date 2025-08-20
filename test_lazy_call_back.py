import gurobipy as gp
from collections import defaultdict

def combined_callback(model, where):
    # --- At fractional LP solutions ---
    if where == gp.GRB.Callback.MIPNODE:
        status = model.cbGet(gp.GRB.Callback.MIPNODE_STATUS)
        if status != gp.GRB.OPTIMAL:
            return

        for var in var_objs:
            for (lhs, sense, rhs, name) in lazy_constr_data[var]:
                val = sum(model.cbGetNodeRel(lhs.getVar(i)) * lhs.getCoeff(i)
                          for i in range(lhs.size()))

                violated = (
                    sense == gp.GRB.LESS_EQUAL and val > rhs + 1e-5 or
                    sense == gp.GRB.GREATER_EQUAL and val < rhs - 1e-5 or
                    sense == gp.GRB.EQUAL and abs(val - rhs) > 1e-5
                )

                if violated:
                    print(f"[MIPNODE] Cutting: {name} due to {var.VarName}")
                    model.cbCut(lhs >= rhs if sense == gp.GRB.GREATER_EQUAL else
                                lhs <= rhs if sense == gp.GRB.LESS_EQUAL else
                                lhs == rhs)
                    return

    # --- At integer-feasible solutions ---
    elif where == gp.GRB.Callback.MIPSOL:
        for var in var_objs:
            if var not in lazy_constr_data:
                continue

            for (lhs, sense, rhs, name) in lazy_constr_data[var]:
                lhs_val = sum(
                    model.cbGetSolution(lhs.getVar(i)) * lhs.getCoeff(i)
                    for i in range(lhs.size())
                )

                violated = (
                    sense == gp.GRB.EQUAL and abs(lhs_val - rhs) > 1e-6 or
                    sense == gp.GRB.LESS_EQUAL and lhs_val > rhs + 1e-6 or
                    sense == gp.GRB.GREATER_EQUAL and lhs_val < rhs - 1e-6
                )

                if violated:
                    print(f"[MIPSOL] Adding lazy constraint '{name}' due to {var.VarName}")
                    model.cbLazy(lhs == rhs if sense == gp.GRB.EQUAL else
                                 lhs <= rhs if sense == gp.GRB.LESS_EQUAL else
                                 lhs >= rhs)
                    return

def aggressive_callback(model, where):
    if where == gp.GRB.Callback.MIPNODE:
        status = model.cbGet(gp.GRB.Callback.MIPNODE_STATUS)
        if status != gp.GRB.OPTIMAL:
            return

        for var in var_objs:
            for (lhs, sense, rhs, name) in lazy_constr_data[var]:
                val = sum(model.cbGetNodeRel(lhs.getVar(i)) * lhs.getCoeff(i)
                          for i in range(lhs.size()))

                violated = (
                    sense == gp.GRB.LESS_EQUAL and val > rhs + 1e-5 or
                    sense == gp.GRB.GREATER_EQUAL and val < rhs - 1e-5 or
                    sense == gp.GRB.EQUAL and abs(val - rhs) > 1e-5
                )

                if violated:
                    model.cbCut(lhs >= rhs if sense == gp.GRB.GREATER_EQUAL else
                                lhs <= rhs if sense == gp.GRB.LESS_EQUAL else
                                lhs == rhs)
                    return

    return callback

def make_lazy_callback(var_objs, lazy_constr_data):
    def callback(model, where):
        if where != gp.GRB.Callback.MIPSOL:
            return

        for var in var_objs:
            if var not in lazy_constr_data:
                continue

            for (lhs, sense, rhs, name) in lazy_constr_data[var]:
                lhs_val = sum(
                    model.cbGetSolution(lhs.getVar(i)) * lhs.getCoeff(i)
                    for i in range(lhs.size())
                )

                violated = (
                    sense == gp.GRB.EQUAL and abs(lhs_val - rhs) > 1e-6 or
                    sense == gp.GRB.LESS_EQUAL and lhs_val > rhs + 1e-6 or
                    sense == gp.GRB.GREATER_EQUAL and lhs_val < rhs - 1e-6
                )

                if violated:
                    print(f"Adding lazy constraint '{name}' due to variable {var.VarName}")
                    if sense == gp.GRB.EQUAL:
                        model.cbLazy(lhs == rhs)
                    elif sense == gp.GRB.LESS_EQUAL:
                        model.cbLazy(lhs <= rhs)
                    elif sense == gp.GRB.GREATER_EQUAL:
                        model.cbLazy(lhs >= rhs)
                    return  # Only add one violated constraint
    return callback

def lazy_callback(model, where):
    if where != gp.GRB.Callback.MIPSOL:
        return

    for var in var_objs:
        if var not in lazy_constr_data:
            continue

        for (lhs, sense, rhs, name) in lazy_constr_data[var]:
            lhs_val = sum(model.cbGetSolution(v) * c for v, c in lhs)

            violated = (
                sense == gp.GRB.EQUAL and abs(lhs_val - rhs) > 1e-5 or
                sense == gp.GRB.LESS_EQUAL and lhs_val > rhs + 1e-5 or
                sense == gp.GRB.GREATER_EQUAL and lhs_val < rhs - 1e-5
            )

            if violated:
                print(f"Adding lazy constraint '{name}' due to variable {var.VarName}")
                if sense == gp.GRB.EQUAL:
                    model.cbLazy(lhs == rhs)
                elif sense == gp.GRB.LESS_EQUAL:
                    model.cbLazy(lhs <= rhs)
                elif sense == gp.GRB.GREATER_EQUAL:
                    model.cbLazy(lhs >= rhs)
                return  # only add the *first* violated constraint


model_path = "model.mps"
var_list = ['v40417', 'v40418', 'v40419', 'v40420', 'v40421', 'v40422','v40423','v40424','v40425','v40426']

# 1. Read the model
model_path = "../ALL_JSON_BIG/model_name.mps"
branch_file = "../ALL_JSON_BIG/branch_priorities.txt"

options = {
    "WLSACCESSID": "b7836a23-3df1-40ac-be4d-310282e2178e",
    "WLSSECRET": "8dd2c11c-cb9b-46f3-b072-4887712ea0c9",
    "LICENSEID": 2690165
}
min_priority_keep_integer=0
with gp.Env(params=options) as env:
    with gp.read(model_path, env=env) as model:
        with open(branch_file, "r") as f:
            for line in f:
                name, bp = line.strip().split()
                var = model.getVarByName(name)
                if int(bp) < min_priority_keep_integer:
                    if var.VType == gp.GRB.BINARY:
                        var.UB = 1
                    var.VType = gp.GRB.CONTINUOUS
                else:
                    var.BranchPriority = int(bp)
        # 2. Map Gurobi Var objects from var_list names
        vars_by_name = {v.VarName: v for v in model.getVars()}
        var_objs = [vars_by_name[name] for name in var_list]

        # 3. Extract constraints involving each variable and store expressions
        lazy_constr_data = defaultdict(list)  # var -> list of (lhs, sense, rhs, name)
        constrs_to_remove = set()

        for var in var_objs:
            col = model.getCol(var)
            for i in range(col.size()):
                constr = col.getConstr(i)
                expr = model.getRow(constr)
                lhs = gp.LinExpr()
                for j in range(expr.size()):
                    lhs.addTerms(expr.getCoeff(j), expr.getVar(j))
                lazy_constr_data[var].append((lhs, constr.Sense, constr.RHS, constr.ConstrName))
                constrs_to_remove.add(constr)

        # 4. Remove the constraints from the model
        model.remove(list(constrs_to_remove))
        model.update()

        # 5. Enable lazy constraints
        model.Params.LazyConstraints = 1

        model.Params.LazyConstraints = 1
        #model.optimize(make_lazy_callback(var_objs, lazy_constr_data))

        model.Params.LazyConstraints = 1
        model.optimize(combined_callback)