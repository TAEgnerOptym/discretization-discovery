import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import time
import pickle,gzip
from pathlib import Path
from datetime import datetime
import io
import sys
import os
import numpy as np
from itertools import groupby
from operator import itemgetter
USE_WORKING_I_KNOW_SLOW=False
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

    #var_name_map = {v: f"v{i}" for i, v in enumerate(var_names)}
    var_name_map = {
        v: v if len(v) < 20 else f"v{i}"
        for i, v in enumerate(var_names)}
    #con_name_map = {c: f"c{i}" for i, c in enumerate(all_con_names)}
    con_name_map = {
        c: c if len(c) < 20 else f"c{i}"
        for i, c in enumerate(all_con_names)}
    #print('len(all_con_names)')
    #print(len(all_con_names))
    #print('len(con_names_exog)')
    #print(len(con_names_exog))
    #print('len(con_names_eq)')
    #print(len(con_names_eq))
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
        "WLSACCESSID": "b7836a23-3df1-40ac-be4d-310282e2178e",
        "WLSSECRET": "8dd2c11c-cb9b-46f3-b072-4887712ea0c9",
        "LICENSEID": 2690165
    }
    old_old=sys.stdout
    sys.stdout = open(os.devnull, 'w')
    with gp.Env(params=options) as env:
        with gp.Model("converted_LP", env=env) as model:
            sys.stdout = old_old#self._original_stdout
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
                #if con_name.startswith("Valid_ineq_"):
                #    print("expr")
                #    print(expr)
                #    print("safe_LB[con_name]")
                #    print(safe_LB[con_name])
                #    print('con_name')
                #    print(con_name)
                #    input('---')
                
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
            #if time_opt>0:
            #print()
            #model.write('LONG_proj.mps')
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
            #print('len(con_name_rev)')
            #print(len(con_name_rev))
            #print('len(model.getConstrs())')
            #print(len(model.getConstrs()))
            objective = model.ObjVal
            time_post = time.time() - time_post
            reduced_costs   = {var_name_rev[var.VarName]: var.RC for var in model.getVars()}

            return {
                "primal_solution": primal_solution,
                "dual_solution": dual_solution,
                "objective": objective,
                "time_pre": time_pre,
                "time_opt": time_opt,
                "time_post": time_post,
                "reduced_costs":reduced_costs
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

    #var_name_map = {v: f"v{i}" for i, v in enumerate(var_names)}
    var_name_map = {
        v: v if len(v) < 30 else f"v{i}"
        for i, v in enumerate(var_names)}
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
            model.setParam("OutputFlag", 1)
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
            model.setParam("OutputFlag", 0)
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
                      dict_binary_vars,dict_var_name_2_is_integer,max_ILP_time=1000,use_interior=False,extra_var_name_priority=dict(),init_sol=dict()):
    #print('use_interior')
    #print(use_interior)
    #print('use_interior')
    #input('--')
    #print('HOLD')
    #lb = dict_con_name_2_LB

    # Map constraint name -> LB, filtered to those starting with "side_ineq_use"
    #side_ineq_use_constraints = {
    #    c: lb[c]
    #    for c in lb
    #    if isinstance(c, str) and c.startswith("side_ineq_use")
    #}

    # Get (var, con) pairs (works if function returns a dict or an iterable of tuples)
    #var_con_pairs = dict_var_con_2_lhs_exog
    #iter_pairs = var_con_pairs.keys() if isinstance(var_con_pairs, dict) else var_con_pairs

    # Map (var, con) -> 1 for constraints with names starting with "side_ineq_use"
    #side_ineq_use_constraints_2 = {
    #    (v, c): 1
    ##    for (v, c) in iter_pairs
    #    if isinstance(c, str) and c.startswith("side_ineq_use")
    #}

    #print('side_ineq_use_constraints_2')
    #print(side_ineq_use_constraints_2)
    #print('side_ineq_use_constraints')
    #print(side_ineq_use_constraints)
    #input('- looking here--')
    time_pre = time.time()

    # Step 0: Name remapping for Gurobi safety
    var_names = list(dict_var_name_2_obj.keys())
    con_names_exog = list(dict_con_name_2_LB.keys())
    con_names_eq = list(dict_con_name_2_eq.keys())
    all_con_names = list(set(con_names_exog) | set(con_names_eq))

    #var_name_map = {v: f"v{i}" for i, v in enumerate(var_names)}
    var_name_map = {
        v: v if len(v) < 40 else f"v{i}"
        for i, v in enumerate(var_names)}
    con_name_map = {
        c: c if len(c) < 60 else f"c{i}"
        for i, c in enumerate(all_con_names)
    }

    #side_ineq_use_constraints_3 = {
    #    c: con_name_map[c]
    #    for c in lb
    #    if isinstance(c, str) and c.startswith("side_ineq_use")
    #}
    #print('side_ineq_use_constraints_3')
    #print(side_ineq_use_constraints_3)
    #input('---')
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
    
    safe_integer_set = {var_name_map[v] for v in dict_var_name_2_is_integer}
    #safe_binary_integer_set=
    safe_var_LB = {var_name_map[k]: v for k, v in dict_var_name_2_LB.items()}
    safe_var_UB = {var_name_map[k]: v for k, v in dict_var_name_2_UB.items()}

    options = {
        "WLSACCESSID": "b7836a23-3df1-40ac-be4d-310282e2178e",
        "WLSSECRET": "8dd2c11c-cb9b-46f3-b072-4887712ea0c9",
        "LICENSEID": 2690165
    }


    fancy_fixed_keys_SIMP = [
        k for k in dict_var_name_2_LB
        if k.startswith("fancy")
    ]
    print('fancy_fixed_keys SIMP')
    print(fancy_fixed_keys_SIMP)
    fancy_fixed_keys = [
        k for k in dict_var_name_2_LB
        if k.startswith("fancy")
        and dict_var_name_2_LB[k] ==dict_var_name_2_UB.get(k)
    ]

    with gp.Env(params=options) as env:
        with gp.Model("converted_MILP", env=env) as model:
            model.setParam("OutputFlag", 1)
            if use_interior==True:
                #model.setParam("Method", 2)
                #model.setParam("OutputFlag", 0)
                #model.setParam("Presolve", 0)
                #model.setParam("Cuts", 0)
                print('using interior')

            model.setParam("TimeLimit", max_ILP_time)
            model.setParam("LogFile", "../ALL_JSON_BIG/gurobi_log.txt")
            #model.setParam("VarBranch", -1)
            #model.setParam("Presolve", 0)

            # Add variables, using binary type where needed
            var_dict = {}
            for name, obj_coeff in safe_var_obj.items():
                lb = safe_var_LB.get(name, 0.0)
                ub = safe_var_UB.get(name, GRB.INFINITY)
                #vtype = GRB.BINARY if name in safe_binary_set else GRB.CONTINUOUS
                vtype=[]
                if name in safe_binary_set:
                    vtype = GRB.BINARY

                   # my_orig_name=var_name_rev[name]
                   # if my_orig_name in init_sol:
                   #     use_start_val=True
                   #     var_dict[name] = model.addVar(lb=lb, ub=ub, obj=obj_coeff, vtype=vtype, name=name,start=init_sol[my_orig_name])

                elif name in safe_integer_set:
                    vtype = GRB.INTEGER
                else:
                    vtype = GRB.CONTINUOUS
                #if use_start_val==False:
                var_dict[name] = model.addVar(lb=lb, ub=ub, obj=obj_coeff, vtype=vtype, name=name)

                #if vtype == GRB.CONTINUOUS:
                    #if 
                    #print('var_name_rev[name]')
                    #print(var_name_rev[name])
                    #input('--')
            model.update()
            for name, var in var_dict.items():
                my_name_rev=var_name_rev[name]
                if my_name_rev in init_sol:         # incumbent_values is your dict of starts
                    var.Start = init_sol[my_name_rev]
            
            model.update()


            if  any(not var_name_rev[v].startswith("act") for v in safe_binary_set | safe_integer_set):
                #input('HERE')
                count_1=0
                count_2=0
                count_3=0
                for v_name in safe_binary_set:
                    safe_name = v_name
                    v=var_dict[v_name]
                    orig_name = var_name_rev[safe_name]
                    
                    if orig_name.startswith("act"):
                        v.BranchPriority = 50
                        count_1=count_1+1
                    else:
                        if orig_name.startswith("fancy_branching_var"):
                            v.BranchPriority = 1000#-count_3
                            if orig_name in extra_var_name_priority:
                                v.BranchPriority=extra_var_name_priority[orig_name]
                            #print('orig_name')
                            #print(orig_name)
                            #print('1000-count_3')
                            #print(1000-count_3)
                            #count_3=count_3+1
                            #input('GOOD NEWS HERE')
                        else:
                            v.BranchPriority = 1
                            count_2=count_2+1
                for v_name in safe_integer_set:
                    orig_name = var_name_rev[v_name]
                    v=var_dict[v_name]
                    v.BranchPriority = 1
                    #print('[count_1,count_2]')
                    #print([count_1,count_2])
                #count_3=0
                #count_4=0
                for v_name in safe_integer_set:
                    v=var_dict[v_name]
                    orig_name = var_name_rev[v_name]
                    #print('orig_name')
                    #print(orig_name)
                    #3#input('look here')
                    if orig_name.startswith("fancy_branching_var"):
                        v.BranchPriority = 1000
                        if orig_name in extra_var_name_priority:
                            v.BranchPriority=extra_var_name_priority[orig_name]
                        #input('----')
                        #print('v_name 3 ')
                        #print(v_name)
                        #if orig_name in pred_val_gain:
                        #    v.BranchPriority =int(np.ceil(pred_val_gain[orig_name]))+51
                        #    print('v.BranchPriority')
                        #    print(v.BranchPriority)
                        #else:
                        #    if pred_val_gain!=None:
                        #        input('error here dict should not be empty')
                        #count_3=count_3+1
                    #else:
                    #    v.BranchPriority = 1
                    #    count_4=count_4+1
            #print('len(safe_binary_set)')
            #print(len(safe_binary_set))
            #input('hihi')
            #model.setParam("VarBranch", 2)
            #print('count_3')
            #print(count_3)
            
            #model.update()
            #count_4=0
            #for v in model.getVars():
            #    if v.BranchPriority>600:
            #        count_4=count_4+1
                    #print('v_name 3 ')
            #input('---')
            #print('count_4')
            #print(count_4)
            #input('--')
            # Group and add constraints
            group_exog = defaultdict(list)
            for (var, con), coeff in safe_exog.items():
                group_exog[con].append((var_dict[var], coeff))

            group_eq = defaultdict(list)
            for (var, con), coeff in safe_eq_map.items():
                group_eq[con].append((var_dict[var], coeff))

            for con_name, terms in group_exog.items():
                #if not terms:   # skip empty lists
                #    continue
                vars_, coeffs = zip(*terms)                     # unpack once
                expr = gp.LinExpr(coeffs, vars_)                # build in C
                model.addConstr(expr >= safe_LB[con_name], name=con_name)

            # Equality constraints
            for con_name, terms in group_eq.items():
                #if not terms:
                #    continue
                vars_, coeffs = zip(*terms)
                expr = gp.LinExpr(coeffs, vars_)
                model.addConstr(expr == safe_EQ[con_name], name=con_name)


            model.ModelSense = GRB.MINIMIZE

            time_pre = time.time() - time_pre
            #model.setParam("OutputFlag", 1)
            log_buffer = io.StringIO()
            print('writing ')
            model.write("model_name.mps")
            with open("branch_priorities.txt", "w") as f:
                for v in model.getVars():
                    bp = v.getAttr("BranchPriority")
                    if bp != 0:
                        f.write(f"{v.VarName} {bp}\n")            
            print('done writing')
            if 1<0:
                model.setParam("Cuts", 0)                # Disable all cutting planes
                model.setParam("Heuristics", 0)          # Disable primal heuristics
                model.setParam("CutPasses", 0)           # No passes even beyond root
                model.setParam("Presolve", 2)            # Leave presolve on (it's cheap and useful)
                model.setParam("NodeMethod", 1)          # Use dual simplex in nodes
                model.setParam("Method", 1)              # Use dual simplex for LPs
                model.setParam("StartNodeLimit", 1)  # Leave presolve on, it's helpful
                model.setParam("VarBranch", 2)  #STRONG BRANCHING 
                #model.setParam("NodeMethod", 1)  # Use dual simplex in the tree
               
                model.update()

            # Set up Tee to write to both stdout and buffer

            side_ineq_use_constraints = {
                constr.ConstrName: constr
                for constr in model.getConstrs()
                if constr.ConstrName.startswith("side_ineq_use")
            }
            print('side_ineq_use_constraints')
            print(side_ineq_use_constraints)
            print('len(side_ineq_use_constraints)')
            print(len(side_ineq_use_constraints))

            #print('writingSCOND')
            #model.write("model_name2.mps")

            #input('look here')
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
            print('done solving')
            print('solve time. '+str(time_opt))

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
                "gurobi_log_string":gurobi_log_string,
            }





def solve_gurobi_lp_bounds(dict_var_name_2_obj,
                    dict_var_con_2_lhs_exog,
                    dict_con_name_2_LB,
                    dict_var_con_2_lhs_eq,
                    dict_con_name_2_eq,dict_var_name_2_LB,dict_var_name_2_UB,use_fast_interior=False):

    

    time_pre = time.time()
    time_pre_1 = time.time()

    # Step 0: Create safe names for variables and constraints
    var_names = list(dict_var_name_2_obj.keys())
    con_names_exog = list(dict_con_name_2_LB.keys())
    con_names_eq = list(dict_con_name_2_eq.keys())
    all_con_names = list(set(con_names_exog) | set(con_names_eq))

    var_name_map = {
        v: v if len(v) < 20 else f"v{i}"
        for i, v in enumerate(var_names)}
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
    time_pre_1 = time.time()-time_pre_1

    options = {
        "WLSACCESSID": "b7836a23-3df1-40ac-be4d-310282e2178e",
        "WLSSECRET": "8dd2c11c-cb9b-46f3-b072-4887712ea0c9",
        "LICENSEID": 2690165
    }
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    with gp.Env(params=options) as env:
        with gp.Model("converted_LP", env=env) as model:
            sys.stdout =original_stdout
            model.setParam("OutputFlag", 0)  
            if use_fast_interior==True:
                model.setParam("OutputFlag", 1)  
                #model.setParam("OutputFlag", 0)  # Suppress solver output
                #model.setParam("Method" , 2)
                #model.setParam("Crossover" , 0)        # 2 = Barrier (interior-point)
                model.setParam("BarConvTol", 1e-2)
            var_dict = {}
            time_pre_2=time.time()
            for name, obj_coeff in safe_var_obj.items():
                lb = safe_var_LB.get(name, 0.0)
                ub = safe_var_UB.get(name, GRB.INFINITY)
                var_dict[name] = model.addVar(lb=lb, ub=ub, obj=obj_coeff, name=name)
            time_pre_2=time.time()-time_pre_2
            time_pre_3=time.time()

            model.update()
            time_pre_3=time.time()-time_pre_3
            time_pre_4=time.time()

            if 1<0:
                #group_exog = defaultdict(list)
                #for (var, con), coeff in safe_exog.items():
                #    group_exog[con].append((var_dict[var], coeff))
#
#                group_eq = defaultdict(list)
#                for (var, con), coeff in safe_eq_map.items():
#                    group_eq[con].append((var_dict[var], coeff))
                vget = var_dict.__getitem__
                def group_by_con_pandas(mapping):
                    """mapping: dict with keys (var, con) and values coeff"""
                    if not mapping:
                        return {}

                    # Build a DataFrame: columns = var, con, coeff
                    rows = ((var, con, coeff) for (var, con), coeff in mapping.items())
                    df = pd.DataFrame.from_records(rows, columns=["var", "con", "coeff"])

                    # Map var via var_dict in vectorized style
                    df["var"] = df["var"].map(vget)

                    # Group by 'con' and rebuild: con -> [(mapped_var, coeff), ...]
                    grouped = {
                        con: list(zip(g["var"].to_numpy(), g["coeff"].to_numpy()))
                        for con, g in df.groupby("con", sort=False)
                    }
                    return grouped

                # Use it for both
                group_exog = group_by_con_pandas(safe_exog)
                group_eq   = group_by_con_pandas(safe_eq_map)

            else:
                vget = var_dict.__getitem__
                get_con = lambda kv: kv[0][1]          # (var, con) -> con
                get_pair = itemgetter(0, 1)            # ((var, con), coeff) -> ((var, con), coeff)

                # ---- exog ----
                ex_items = safe_exog.items()
                # sort once by con (kv[0][1])
                ex_sorted = sorted(ex_items, key=get_con)
                group_exog = {
                    con: [(vget(var), coeff) for (var, _), coeff in grp]
                    for con, grp in groupby(ex_sorted, key=get_con)
                }

                # ---- eq ----
                eq_items = safe_eq_map.items()
                eq_sorted = sorted(eq_items, key=get_con)
                group_eq = {
                    con: [(vget(var), coeff) for (var, _), coeff in grp]
                    for con, grp in groupby(eq_sorted, key=get_con)
                }
            time_pre_4=time.time()-time_pre_4
            time_pre_5=time.time()
            for con_name, terms in group_exog.items():
                vars_, coeffs = zip(*terms)                     # unpack once
                expr = gp.LinExpr(coeffs, vars_)                # build in C
                model.addConstr(expr >= safe_LB[con_name], name=con_name)
            time_pre_5=time.time()-time_pre_5
            time_pre_6=time.time()
            for con_name, terms in group_eq.items():

                vars_, coeffs = zip(*terms)
                expr = gp.LinExpr(coeffs, vars_)
                model.addConstr(expr == safe_EQ[con_name], name=con_name)
            model.ModelSense = GRB.MINIMIZE
            time_pre_6=time.time()-time_pre_6
            time_pre=time.time()-time_pre
            time_opt = time.time()
            model.optimize()
            time_opt = time.time() - time_opt
            time_post=time.time()
            if model.status != GRB.OPTIMAL:

                model.write('errHere.mps')
                print('model.status')
                print(model.status)
                raise RuntimeError("Gurobi did not find an optimal solution.")

            primal_solution = {
                var_name_rev[var.VarName]: var.X for var in model.getVars()
            }

            dual_solution = {
                con_name_rev[con.ConstrName]: con.Pi for con in model.getConstrs()
            }

            objective = model.ObjVal
            reduced_costs   = {var_name_rev[var.VarName]: var.RC for var in model.getVars()}
            time_post=time.time()-time_post
            print(['time_pre,time_opt,time_post'])
            print([time_pre,time_opt,time_post])
            print('time_pre_1,time_pre_2,time_pre_3,time_pre_4,time_pre_5,time_pre_6')
            print([time_pre_1,time_pre_2,time_pre_3,time_pre_4,time_pre_5,time_pre_6])
            return {
                "primal_solution": primal_solution,
                "dual_solution": dual_solution,
                "objective": objective,
                "time_pre": time_pre,
                "time_opt": time_opt,
                "time_post": time_post,
                "reduced_costs":reduced_costs
            }






def NEW_solve_gurobi_lp_bounds(
    dict_var_name_2_obj,
    dict_var_con_2_lhs_exog,
    dict_con_name_2_LB,
    dict_var_con_2_lhs_eq,
    dict_con_name_2_eq,
    dict_var_name_2_LB,
    dict_var_name_2_UB,use_pareto=False
):
    """
    Super-fast LP builder using Gurobi's matrix API.
    - No Python loops over terms or constraints.
    - Variables and constraints added in bulk.
    - Returns solutions keyed by your original names (we avoid naming inside Gurobi for speed).
    """
    import os, sys, time
    import numpy as np
    import scipy.sparse as sp
    import gurobipy as gp
    from gurobipy import GRB

    t0 = time.time()

    # ---------- Ordered labels ----------
    # Keep insertion order of your dicts; deterministic and cheap.
    var_names  = list(dict_var_name_2_obj.keys())
    exog_names = list(dict_con_name_2_LB.keys())
    eq_names   = list(dict_con_name_2_eq.keys())

    n = len(var_names)
    mexog = len(exog_names)
    meq   = len(eq_names)

    # ---------- Variable data (vectorized) ----------
    obj = np.fromiter((dict_var_name_2_obj[v]                 for v in var_names), float, count=n)
    lb  = np.fromiter((dict_var_name_2_LB.get(v, 0.0)         for v in var_names), float, count=n)
    ub  = np.fromiter((dict_var_name_2_UB.get(v, GRB.INFINITY) for v in var_names), float, count=n)

    # ---------- Helper: build CSR without Python loops ----------
    def _indexer(values, categories):
        """Map array of 'values' into indices of 'categories' (both object arrays), vectorized."""
        order = np.argsort(categories)
        sorted_cats = categories[order]
        pos = np.searchsorted(sorted_cats, values)
        ok = (pos < sorted_cats.size) & (sorted_cats[pos] == values)
        if not np.all(ok):
            missing = np.unique(values[~ok]).tolist()
            raise KeyError(f"Labels not found: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        return order[pos]

    def build_block(map_dict, row_labels, col_labels):
        """Convert {(var, con): coeff} -> CSR with rows=row_labels, cols=col_labels, no Python loops."""
        if not map_dict:
            return None
        m, n = len(row_labels), len(col_labels)
        N = len(map_dict)
        # keys -> object array of shape (N, 2); vals -> float array
        keys = np.fromiter(map_dict.keys(), dtype=object, count=N)
        vals = np.fromiter(map_dict.values(), dtype=float,  count=N)
        keys = np.stack(keys)       # shape (N, 2): [var, con]
        vars_arr = keys[:, 0]
        cons_arr = keys[:, 1]
        row_labels = np.asarray(row_labels, dtype=object)
        col_labels = np.asarray(col_labels, dtype=object)
        rows = _indexer(cons_arr, row_labels)
        cols = _indexer(vars_arr, col_labels)
        return sp.coo_matrix((vals, (rows, cols)), shape=(m, n)).tocsr()

    # ---------- Build constraint blocks (vectorized) ----------
    A_exog = build_block(dict_var_con_2_lhs_exog, exog_names, var_names)  # rows: exog constraints, cols: vars
    A_eq   = build_block(dict_var_con_2_lhs_eq,   eq_names,   var_names)  # rows: eq constraints,   cols: vars

    rhs_exog = (np.fromiter((dict_con_name_2_LB[c] for c in exog_names), float, count=mexog)
                if A_exog is not None else None)
    rhs_eq   = (np.fromiter((dict_con_name_2_eq[c] for c in eq_names),   float, count=meq)
                if A_eq   is not None else None)

    # Stack into one big matrix constraint to minimize calls into the API
    have_exog = A_exog is not None
    have_eq   = A_eq   is not None

    if have_exog and have_eq:
        A = sp.vstack([A_exog, A_eq], format='csr')
        rhs = np.concatenate([rhs_exog, rhs_eq])
        senses = np.concatenate([np.full(mexog, '>'), np.full(meq, '=')])
        # Keep slice info for later mapping of duals
        exog_slice = slice(0, mexog)
        eq_slice   = slice(mexog, mexog + meq)
    elif have_exog:
        A = A_exog
        rhs = rhs_exog
        senses = np.full(mexog, '>')
        exog_slice = slice(0, mexog)
        eq_slice   = slice(0, 0)   # empty
    elif have_eq:
        A = A_eq
        rhs = rhs_eq
        senses = np.full(meq, '=')
        exog_slice = slice(0, 0)   # empty
        eq_slice   = slice(0, meq)
    else:
        A = None

    # ---------- Build & solve model (silence license banner) ----------
    options = {
        "WLSACCESSID": "b7836a23-3df1-40ac-be4d-310282e2178e",
        "WLSSECRET":   "8dd2c11c-cb9b-46f3-b072-4887712ea0c9",
        "LICENSEID":   2690165
    }

    original_stdout = sys.stdout
    devnull = open(os.devnull, 'w')
    sys.stdout = devnull
    try:
        with gp.Env(params=options) as env:
            with gp.Model("lp_fast", env=env) as model:
                model.setParam("OutputFlag", 0)
                if use_pareto==True:
                    model.setParam("OutputFlag", 0)  # Suppress solver output
                    model.setParam("Method" , 2)
                    model.setParam("Crossover" , 0)        # 2 = Barrier (interior-point)
                    model.setParam("BarConvTol", 1e-4)
                # Add all variables in one shot (continuous by default)
                x = model.addMVar(shape=n, lb=lb, ub=ub, obj=obj)  # no per-var naming

                # Add all constraints (if any) in a single matrix call
                mcon = None
                if A is not None:
                    mcon = model.addMConstr(A, x, senses, rhs)     # accepts SciPy CSR & sense vector

                model.ModelSense = GRB.MINIMIZE

                time_pre = time.time() - t0
                t_opt = time.time()
                model.optimize()
                time_opt = time.time() - t_opt

                t_post = time.time()

                # Retrieve primal/dual/rc in bulk (NumPy arrays)
                x_val = x.X          # ndarray of variable values
                rc    = x.RC         # ndarray of reduced costs (LP) 
                objv  = float(model.ObjVal)

                # Duals only for continuous LPs; mcon.Pi returns ndarray
                dual_solution = {}
                if mcon is not None:
                    pi = mcon.Pi     # ndarray of size rhs.shape[0]
                    if mexog:
                        dual_solution.update(dict(zip(exog_names, map(float, pi[exog_slice]))))
                    if meq:
                        dual_solution.update(dict(zip(eq_names,   map(float, pi[eq_slice]))))

                # Map arrays back to your original names without naming inside Gurobi
                primal_solution = dict(zip(var_names, map(float, x_val)))
                reduced_costs   = dict(zip(var_names, map(float, rc)))

                time_post = time.time() - t_post
    finally:
        sys.stdout = original_stdout
        devnull.close()

    return {
        "primal_solution": primal_solution,
        "dual_solution": dual_solution,
        "objective": objv,
        "time_pre": time_pre,
        "time_opt": time_opt,
        "time_post": time_post,
        "reduced_costs": reduced_costs,
    }
