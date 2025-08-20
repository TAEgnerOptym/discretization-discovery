import gurobipy as gp
import numpy as np

model_path = "../ALL_JSON_BIG/model_name.mps"
branch_file = "../ALL_JSON_BIG/branch_priorities.txt"

options = {
    "WLSACCESSID": "b7836a23-3df1-40ac-be4d-310282e2178e",
    "WLSSECRET": "8dd2c11c-cb9b-46f3-b072-4887712ea0c9",
    "LICENSEID": 2690165
}

var_list = ['v40417', 'v40418', 'v40419', 'v40420', 'v40421', 'v40422','v40423','v40424','v40425','v40426']
min_priority_keep_integer = 99

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
        model.update()
        # Step 1: Cache constraints for each variable
        var_to_constr_exprs = {}
        for varname in var_list:
            var = model.getVarByName(varname)
            col = model.getCol(var)
            var_to_constr_exprs[var] = []
            for i in range(col.size()):
                constr = col.getConstr(i)
                expr = model.getRow(constr)
                lhs = gp.LinExpr()
                for j in range(expr.size()):
                    lhs.addTerms(expr.getCoeff(j), expr.getVar(j))
                sense = constr.Sense
                rhs = constr.RHS
                name = constr.ConstrName
                var_to_constr_exprs[var].append((lhs, sense, rhs, name))

        # Step 2: Remove those constraints
        all_constrs = {constr for constr_exprs in var_to_constr_exprs.values() for (lhs, sense, rhs, name) in constr_exprs for constr in model.getConstrs() if constr.ConstrName == name}
        model.remove(list(all_constrs))
        model.update()
        model.optimize()

        # Step 3: Incrementally add back constraints per variable
        for k in range(len(var_list)):
            var = model.getVarByName(var_list[k])
            for (lhs, sense, rhs, name) in var_to_constr_exprs[var]:
                if sense == gp.GRB.EQUAL:
                    model.addConstr(lhs == rhs, name=name)
                elif sense == gp.GRB.LESS_EQUAL:
                    model.addConstr(lhs <= rhs, name=name)
                elif sense == gp.GRB.GREATER_EQUAL:
                    model.addConstr(lhs >= rhs, name=name)
            model.update()
            model.optimize()
