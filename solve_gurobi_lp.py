import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import time

import io
import sys

class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)
    def flush(self):
        for s in self.streams:
            s.flush()


def solve_gurobi_lp(dict_var_name_2_obj,
                    dict_var_con_2_lhs_exog,
                    dict_con_name_2_LB,
                    dict_var_con_2_lhs_eq,
                    dict_con_name_2_eq):

    time_pre = time.time()

    # Step 0: Create safe names for variables and constraints
    var_names = list(dict_var_name_2_obj.keys())
    con_names_exog = list(dict_con_name_2_LB.keys())
    con_names_eq = list(dict_con_name_2_eq.keys())
    all_con_names = list(set(con_names_exog) | set(con_names_eq))

    var_name_map = {v: f"v{i}" for i, v in enumerate(var_names)}
    con_name_map = {c: f"c{i}" for i, c in enumerate(all_con_names)}

    var_name_rev = {v_alias: v for v, v_alias in var_name_map.items()}
    con_name_rev = {c_alias: c for c, c_alias in con_name_map.items()}

    # Remap data structures using safe names
    safe_var_obj = {var_name_map[k]: v for k, v in dict_var_name_2_obj.items()}
    safe_exog = {(var_name_map[v], con_name_map[c]): coeff
                 for (v, c), coeff in dict_var_con_2_lhs_exog.items()}
    safe_eq_map = {(var_name_map[v], con_name_map[c]): coeff
                   for (v, c), coeff in dict_var_con_2_lhs_eq.items()}
    safe_LB = {con_name_map[k]: v for k, v in dict_con_name_2_LB.items()}
    safe_EQ = {con_name_map[k]: v for k, v in dict_con_name_2_eq.items()}

    options = {
        "WLSACCESSID": "8f7bb9d6-8fe5-4349-9dd3-6abbaa9199a0",
        "WLSSECRET": "cb02810a-e0e2-4a1f-8fc0-fd375f65fc65",
        "LICENSEID": 2660300
    }

    with gp.Env(params=options) as env:
        with gp.Model("converted_LP", env=env) as model:
            model.setParam("OutputFlag", 0)  # Suppress solver output

            # Step 1: Add variables
            var_dict = {
                name: model.addVar(lb=0, obj=obj_coeff, name=name)
                for name, obj_coeff in safe_var_obj.items()
            }

            model.update()

            # Step 2: Group and add constraints
            group_exog = defaultdict(list)
            for (var, con), coeff in safe_exog.items():
                group_exog[con].append((var_dict[var], coeff))

            group_eq = defaultdict(list)
            for (var, con), coeff in safe_eq_map.items():
                group_eq[con].append((var_dict[var], coeff))

            for con_name, terms in group_exog.items():
                expr = gp.LinExpr()
                for var, coeff in terms:
                    expr.addTerms(coeff, var)
                model.addConstr(expr >= safe_LB[con_name], name=con_name)

            for con_name, terms in group_eq.items():
                expr = gp.LinExpr()
                for var, coeff in terms:
                    expr.addTerms(coeff, var)
                model.addConstr(expr == safe_EQ[con_name], name=con_name)

            model.ModelSense = GRB.MINIMIZE

            time_pre = time.time() - time_pre
            print('Starting Gur LP')
            time_opt = time.time()
            model.optimize()
            time_opt = time.time() - time_opt
            print('DONE Gur LP')

            time_post = time.time()

            if model.status != GRB.OPTIMAL:

                print('model.status')
                print(model.status)
                raise RuntimeError("Gurobi did not find an optimal solution.")

            # Step 3: Recover solutions and remap names
            primal_solution = {
                var_name_rev[var.VarName]: var.X for var in model.getVars()
            }

            dual_solution = {
                con_name_rev[con.ConstrName]: con.Pi for con in model.getConstrs()
            }

            objective = model.ObjVal
            time_post = time.time() - time_post

            return {
                "primal_solution": primal_solution,
                "dual_solution": dual_solution,
                "objective": objective,
                "time_pre": time_pre,
                "time_opt": time_opt,
                "time_post": time_post
            }



def solve_gurobi_milp(dict_var_name_2_obj,
                      dict_var_con_2_lhs_exog,
                      dict_con_name_2_LB,
                      dict_var_con_2_lhs_eq,
                      dict_con_name_2_eq,
                      dict_binary_vars,max_ILP_time=1000):
    time_pre = time.time()

    # Step 0: Name remapping for Gurobi safety
    var_names = list(dict_var_name_2_obj.keys())
    con_names_exog = list(dict_con_name_2_LB.keys())
    con_names_eq = list(dict_con_name_2_eq.keys())
    all_con_names = list(set(con_names_exog) | set(con_names_eq))

    var_name_map = {v: f"v{i}" for i, v in enumerate(var_names)}
    con_name_map = {c: f"c{i}" for i, c in enumerate(all_con_names)}
    var_name_rev = {v_alias: v for v, v_alias in var_name_map.items()}
    con_name_rev = {c_alias: c for c, c_alias in con_name_map.items()}

    # Remap data structures using safe names
    safe_var_obj = {var_name_map[k]: v for k, v in dict_var_name_2_obj.items()}
    safe_exog = {(var_name_map[v], con_name_map[c]): coeff
                 for (v, c), coeff in dict_var_con_2_lhs_exog.items()}
    safe_eq_map = {(var_name_map[v], con_name_map[c]): coeff
                   for (v, c), coeff in dict_var_con_2_lhs_eq.items()}
    safe_LB = {con_name_map[k]: v for k, v in dict_con_name_2_LB.items()}
    safe_EQ = {con_name_map[k]: v for k, v in dict_con_name_2_eq.items()}
    safe_binary_set = {var_name_map[v] for v in dict_binary_vars}

    options = {
        "WLSACCESSID": "8f7bb9d6-8fe5-4349-9dd3-6abbaa9199a0",
        "WLSSECRET": "cb02810a-e0e2-4a1f-8fc0-fd375f65fc65",
        "LICENSEID": 2660300
    }

    with gp.Env(params=options) as env:
        with gp.Model("converted_MILP", env=env) as model:
            model.setParam("OutputFlag", 0)
            model.setParam("TimeLimit", max_ILP_time)

            # Add variables, using binary type where needed
            var_dict = {}
            for name, obj_coeff in safe_var_obj.items():
                vtype = GRB.BINARY if name in safe_binary_set else GRB.CONTINUOUS
                var_dict[name] = model.addVar(lb=0, obj=obj_coeff, vtype=vtype, name=name)

            model.update()

            # Group and add constraints
            group_exog = defaultdict(list)
            for (var, con), coeff in safe_exog.items():
                group_exog[con].append((var_dict[var], coeff))

            group_eq = defaultdict(list)
            for (var, con), coeff in safe_eq_map.items():
                group_eq[con].append((var_dict[var], coeff))

            for con_name, terms in group_exog.items():
                expr = gp.LinExpr()
                for var, coeff in terms:
                    expr.addTerms(coeff, var)
                model.addConstr(expr >= safe_LB[con_name], name=con_name)

            for con_name, terms in group_eq.items():
                expr = gp.LinExpr()
                for var, coeff in terms:
                    expr.addTerms(coeff, var)
                model.addConstr(expr == safe_EQ[con_name], name=con_name)

            model.ModelSense = GRB.MINIMIZE

            time_pre = time.time() - time_pre
            model.setParam("OutputFlag", 1)
            time_opt = time.time()
            model.optimize()
            time_opt = time.time() - time_opt
            time_post = time.time()
            MIP_lower_bound=model.ObjBound
            #if model.status != GRB.OPTIMAL:
            #    raise RuntimeError("Gurobi did not find an optimal MILP solution.")

            # Extract primal solution and objective only (no duals in MILP)
            primal_solution = {var_name_rev[var.VarName]: var.X for var in model.getVars()}
            objective = model.ObjVal
            time_post = time.time() - time_post

            return {
                "primal_solution": primal_solution,
                "objective": objective,
                "time_pre": time_pre,
                "time_opt": time_opt,
                "time_post": time_post,
                "MIP_lower_bound":MIP_lower_bound
            }



def solve_gurobi_milp_bounds(dict_var_name_2_obj,
                      dict_var_con_2_lhs_exog,
                      dict_con_name_2_LB,
                      dict_var_con_2_lhs_eq,
                      dict_con_name_2_eq,
                      dict_var_name_2_LB,dict_var_name_2_UB,
                      dict_binary_vars,max_ILP_time=1000):
    time_pre = time.time()

    # Step 0: Name remapping for Gurobi safety
    var_names = list(dict_var_name_2_obj.keys())
    con_names_exog = list(dict_con_name_2_LB.keys())
    con_names_eq = list(dict_con_name_2_eq.keys())
    all_con_names = list(set(con_names_exog) | set(con_names_eq))

    var_name_map = {v: f"v{i}" for i, v in enumerate(var_names)}
    con_name_map = {c: f"c{i}" for i, c in enumerate(all_con_names)}
    var_name_rev = {v_alias: v for v, v_alias in var_name_map.items()}
    con_name_rev = {c_alias: c for c, c_alias in con_name_map.items()}

    # Remap data structures using safe names
    safe_var_obj = {var_name_map[k]: v for k, v in dict_var_name_2_obj.items()}
    safe_exog = {(var_name_map[v], con_name_map[c]): coeff
                 for (v, c), coeff in dict_var_con_2_lhs_exog.items()}
    safe_eq_map = {(var_name_map[v], con_name_map[c]): coeff
                   for (v, c), coeff in dict_var_con_2_lhs_eq.items()}
    safe_LB = {con_name_map[k]: v for k, v in dict_con_name_2_LB.items()}
    safe_EQ = {con_name_map[k]: v for k, v in dict_con_name_2_eq.items()}
    safe_binary_set = {var_name_map[v] for v in dict_binary_vars}
    safe_var_LB = {var_name_map[k]: v for k, v in dict_var_name_2_LB.items()}
    safe_var_UB = {var_name_map[k]: v for k, v in dict_var_name_2_UB.items()}

    options = {
        "WLSACCESSID": "8f7bb9d6-8fe5-4349-9dd3-6abbaa9199a0",
        "WLSSECRET": "cb02810a-e0e2-4a1f-8fc0-fd375f65fc65",
        "LICENSEID": 2660300
    }

    with gp.Env(params=options) as env:
        with gp.Model("converted_MILP", env=env) as model:
            model.setParam("OutputFlag", 0)
            model.setParam("TimeLimit", max_ILP_time)
            model.setParam("LogFile", "../ALL_JSON_BIG/gurobi_log.txt")
            # Add variables, using binary type where needed
            var_dict = {}
            for name, obj_coeff in safe_var_obj.items():
                lb = safe_var_LB.get(name, 0.0)
                ub = safe_var_UB.get(name, GRB.INFINITY)
                vtype = GRB.BINARY if name in safe_binary_set else GRB.CONTINUOUS
                var_dict[name] = model.addVar(lb=lb, ub=ub, obj=obj_coeff, vtype=vtype, name=name)

            if  any(not var_name_rev[v].startswith("act") for v in safe_binary_set):
                #input('HERE')
                count_1=0
                count_2=0
                for v_name in safe_binary_set:
                    safe_name = v_name
                    v=var_dict[v_name]
                    orig_name = var_name_rev[safe_name]
                    
                    if orig_name.startswith("act"):
                        v.BranchPriority = 100
                        count_1=count_1+1
                    else:
                        v.BranchPriority = 1
                        count_2=count_2+1
                print('[count_1,count_2]')
                print([count_1,count_2])
            model.update()

            # Group and add constraints
            group_exog = defaultdict(list)
            for (var, con), coeff in safe_exog.items():
                group_exog[con].append((var_dict[var], coeff))

            group_eq = defaultdict(list)
            for (var, con), coeff in safe_eq_map.items():
                group_eq[con].append((var_dict[var], coeff))

            for con_name, terms in group_exog.items():
                expr = gp.LinExpr()
                for var, coeff in terms:
                    expr.addTerms(coeff, var)
                model.addConstr(expr >= safe_LB[con_name], name=con_name)

            for con_name, terms in group_eq.items():
                expr = gp.LinExpr()
                for var, coeff in terms:
                    expr.addTerms(coeff, var)
                model.addConstr(expr == safe_EQ[con_name], name=con_name)

            model.ModelSense = GRB.MINIMIZE

            time_pre = time.time() - time_pre
            model.setParam("OutputFlag", 1)
            log_buffer = io.StringIO()
            if 1<0:
                model.setParam("Cuts", 0)
                model.setParam("Heuristics", 0)
                model.setParam("CutPasses", 0)
                model.setParam("Presolve", 2)  # Leave presolve on, it's helpful
                #model.setParam("NodeMethod", 1)  # Use dual simplex in the tree
                model.update()

            # Set up Tee to write to both stdout and buffer
            tee = Tee(sys.__stdout__, log_buffer)
            sys.stdout = tee
            time_opt = time.time()
            model.optimize()
            time_opt = time.time() - time_opt
            sys.stdout = sys.__stdout__

            # Extract the log from memory
            gurobi_log_string = log_buffer.getvalue()
            log_buffer.close()
            time_post = time.time()
            MIP_lower_bound=model.ObjBound
            #if model.status != GRB.OPTIMAL:
            #    raise RuntimeError("Gurobi did not find an optimal MILP solution.")

            # Extract primal solution and objective only (no duals in MILP)
            primal_solution = {var_name_rev[var.VarName]: var.X for var in model.getVars()}
            objective = model.ObjVal
            time_post = time.time() - time_post

            return {
                "primal_solution": primal_solution,
                "objective": objective,
                "time_pre": time_pre,
                "time_opt": time_opt,
                "time_post": time_post,
                "MIP_lower_bound":MIP_lower_bound,
                "gurobi_log_string":gurobi_log_string
            }





def solve_gurobi_lp_bounds(dict_var_name_2_obj,
                    dict_var_con_2_lhs_exog,
                    dict_con_name_2_LB,
                    dict_var_con_2_lhs_eq,
                    dict_con_name_2_eq,dict_var_name_2_LB,dict_var_name_2_UB):

    time_pre = time.time()

    # Step 0: Create safe names for variables and constraints
    var_names = list(dict_var_name_2_obj.keys())
    con_names_exog = list(dict_con_name_2_LB.keys())
    con_names_eq = list(dict_con_name_2_eq.keys())
    all_con_names = list(set(con_names_exog) | set(con_names_eq))

    var_name_map = {v: f"v{i}" for i, v in enumerate(var_names)}
    con_name_map = {c: f"c{i}" for i, c in enumerate(all_con_names)}

    var_name_rev = {v_alias: v for v, v_alias in var_name_map.items()}
    con_name_rev = {c_alias: c for c, c_alias in con_name_map.items()}

    # Remap data structures using safe names
    safe_var_obj = {var_name_map[k]: v for k, v in dict_var_name_2_obj.items()}
    safe_exog = {(var_name_map[v], con_name_map[c]): coeff
                 for (v, c), coeff in dict_var_con_2_lhs_exog.items()}
    safe_eq_map = {(var_name_map[v], con_name_map[c]): coeff
                   for (v, c), coeff in dict_var_con_2_lhs_eq.items()}
    safe_LB = {con_name_map[k]: v for k, v in dict_con_name_2_LB.items()}
    safe_EQ = {con_name_map[k]: v for k, v in dict_con_name_2_eq.items()}

    safe_var_LB = {var_name_map[k]: v for k, v in dict_var_name_2_LB.items()}
    safe_var_UB = {var_name_map[k]: v for k, v in dict_var_name_2_UB.items()}

    options = {
        "WLSACCESSID": "8f7bb9d6-8fe5-4349-9dd3-6abbaa9199a0",
        "WLSSECRET": "cb02810a-e0e2-4a1f-8fc0-fd375f65fc65",
        "LICENSEID": 2660300
    }

    with gp.Env(params=options) as env:
        with gp.Model("converted_LP", env=env) as model:
            model.setParam("OutputFlag", 0)  # Suppress solver output

            # Step 1: Add variables
            var_dict = {}
            for name, obj_coeff in safe_var_obj.items():
                lb = safe_var_LB.get(name, 0.0)
                ub = safe_var_UB.get(name, GRB.INFINITY)
                #if lb>0 or ub<100000:
                #    print('[lb,ub]')
                 #   print([lb,ub])
                 #   input('here')
                var_dict[name] = model.addVar(lb=lb, ub=ub, obj=obj_coeff, name=name)


            model.update()

            # Step 2: Group and add constraints
            group_exog = defaultdict(list)
            for (var, con), coeff in safe_exog.items():
                group_exog[con].append((var_dict[var], coeff))

            group_eq = defaultdict(list)
            for (var, con), coeff in safe_eq_map.items():
                group_eq[con].append((var_dict[var], coeff))

            for con_name, terms in group_exog.items():
                expr = gp.LinExpr()
                for var, coeff in terms:
                    expr.addTerms(coeff, var)
                model.addConstr(expr >= safe_LB[con_name], name=con_name)

            for con_name, terms in group_eq.items():
                expr = gp.LinExpr()
                for var, coeff in terms:
                    expr.addTerms(coeff, var)
                model.addConstr(expr == safe_EQ[con_name], name=con_name)

            model.ModelSense = GRB.MINIMIZE

            time_pre = time.time() - time_pre
            print('Starting Gur LP')
            time_opt = time.time()
            model.optimize()
            time_opt = time.time() - time_opt
            print('DONE Gur LP')

            time_post = time.time()

            if model.status != GRB.OPTIMAL:

                print('model.status')
                print(model.status)
                raise RuntimeError("Gurobi did not find an optimal solution.")

            # Step 3: Recover solutions and remap names
            primal_solution = {
                var_name_rev[var.VarName]: var.X for var in model.getVars()
            }

            dual_solution = {
                con_name_rev[con.ConstrName]: con.Pi for con in model.getConstrs()
            }

            objective = model.ObjVal
            time_post = time.time() - time_post

            return {
                "primal_solution": primal_solution,
                "dual_solution": dual_solution,
                "objective": objective,
                "time_pre": time_pre,
                "time_opt": time_opt,
                "time_post": time_post
            }

