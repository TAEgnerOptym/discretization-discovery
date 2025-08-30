import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict

import pickle
with open("debug_snapshot_20250827_154207.pkl", "rb") as f:
    state = pickle.load(f)

    dict_var_name_2_obj=state['dict_var_name_2_obj']
    dict_con_name_2_LB=state['dict_con_name_2_LB']
    dict_con_name_2_eq=state['dict_con_name_2_eq']
    dict_var_con_2_lhs_exog=state['dict_var_con_2_lhs_exog']
    dict_var_con_2_lhs_eq=state['dict_var_con_2_lhs_eq']
    dict_var_name_2_LB=state['dict_var_name_2_LB']
    dict_var_name_2_UB=state['dict_var_name_2_UB']
    # Step 0: Create safe names for variables and constraints
    var_names = list(dict_var_name_2_obj.keys())
    con_names_exog = list(dict_con_name_2_LB.keys())
    #for zz in newlb_items:
    #    dict_con_name_2_LB[zz]=0

    newlb_safe_exog = {
        (v, c): coeff
        for (v, c), coeff in dict_var_con_2_lhs_exog.items()
        if isinstance(c, str) and c.startswith("NewLB_")
    }
    print('newlb_safe_exog')
    print(newlb_safe_exog)
    input('---')
    con_names_eq = list(dict_con_name_2_eq.keys())
    all_con_names = list(set(con_names_exog) | set(con_names_eq))

    #var_name_map = {v: f"v{i}" for i, v in enumerate(var_names)}
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

    options = {
        "WLSACCESSID": "b7836a23-3df1-40ac-be4d-310282e2178e",
        "WLSSECRET": "8dd2c11c-cb9b-46f3-b072-4887712ea0c9",
        "LICENSEID": 2690165
    }
    with gp.Env(params=options) as env:
        with gp.Model("converted_LP", env=env) as model:
            #model.setParam("OutputFlag", 0)  # Suppress solver output
            #model.setParam("Method", 1)
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


            newlb_items = {k: v for k, v in dict_con_name_2_LB.items() if isinstance(k, str) and k.startswith("NewLB_")}
            my_special_con_name=list(newlb_items)[0]

            for con_name, terms in group_exog.items():
                expr = gp.LinExpr()
                for var, coeff in terms:
                    expr.addTerms(coeff, var)
                model.addConstr(expr >= safe_LB[con_name], name=con_name)
                #print('con_name')
                #print(con_name)
                #print('con_name_rev[con_name]')
                #print(con_name_rev[con_name])
                #input('--')
                if con_name_rev[con_name].startswith("NewLB_")==True:
                    print('expr')
                    print(expr)
                    print('safe_LB[con_name]')
                    print(safe_LB[con_name])
                    print('con_name')
                    print(con_name)
                    input('---')
            for con_name, terms in group_eq.items():
                expr = gp.LinExpr()
                for var, coeff in terms:
                    expr.addTerms(coeff, var)
                model.addConstr(expr == safe_EQ[con_name], name=con_name)

            model.ModelSense = GRB.MINIMIZE


            newlb_safe_exog = {
                (var_name_map[v], con_name_map[c]): coeff
                for (v, c), coeff in dict_var_con_2_lhs_exog.items()
                if isinstance(c, str) and c.startswith("NewLB_")
            }
            print('newlb_safe_exog')
            print(newlb_safe_exog)
            

            model.optimize()
