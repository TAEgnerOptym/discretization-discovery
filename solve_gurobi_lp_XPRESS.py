import time, sys, os, io
from collections import defaultdict
from itertools import groupby
from operator import itemgetter

import xpress as xp


def _maybe_init_xpress(xpress_auth_path=None):
    try:
        if xpress_auth_path:
            xp.init(xpress_auth_path)
        else:
            xp.init()
    except Exception:
        # Already initialized or license auto-detected
        pass


def solve_gurobi_milp_bounds(
    dict_var_name_2_obj,
    dict_var_con_2_lhs_exog,
    dict_con_name_2_LB,
    dict_var_con_2_lhs_eq,
    dict_con_name_2_eq,
    dict_var_name_2_LB, dict_var_name_2_UB,
    dict_binary_vars, dict_var_name_2_is_integer,
    max_ILP_time=1000, use_interior=False,
    extra_var_name_priority=dict(),
    init_sol=dict(),
    xpress_auth_path=None,
    log_path="../ALL_JSON_BIG/xpress_milp_log.txt"
):
    _maybe_init_xpress(xpress_auth_path)

    time_pre = time.time()

    # --- Safe name remapping (unchanged) ---
    var_names = list(dict_var_name_2_obj.keys())
    con_names_exog = list(dict_con_name_2_LB.keys())
    con_names_eq = list(dict_con_name_2_eq.keys())
    all_con_names = list(set(con_names_exog) | set(con_names_eq))

    var_name_map = {v: (v if len(v) < 40 else f"v{i}") for i, v in enumerate(var_names)}
    con_name_map = {c: (c if len(c) < 60 else f"c{i}") for i, c in enumerate(all_con_names)}
    var_name_rev = {v_alias: v for v, v_alias in var_name_map.items()}
    con_name_rev = {c_alias: c for c, c_alias in con_name_map.items()}

    safe_var_obj = {var_name_map[k]: v for k, v in dict_var_name_2_obj.items()}
    safe_exog = {(var_name_map[v], con_name_map[c]): coeff
                 for (v, c), coeff in dict_var_con_2_lhs_exog.items()}
    safe_eq_map = {(var_name_map[v], con_name_map[c]): coeff
                   for (v, c), coeff in dict_var_con_2_lhs_eq.items()}
    safe_LB = {con_name_map[k]: v for k, v in dict_con_name_2_LB.items()}
    safe_EQ = {con_name_map[k]: v for k, v in dict_con_name_2_eq.items()}
    safe_binary_set = {var_name_map[v] for v in dict_binary_vars}
    safe_integer_set = {var_name_map[v] for v in dict_var_name_2_is_integer}
    safe_var_LB = {var_name_map[k]: v for k, v in dict_var_name_2_LB.items()}
    safe_var_UB = {var_name_map[k]: v for k, v in dict_var_name_2_UB.items()}

    # --- Build model in Xpress ---
    p = xp.problem(name="converted_MILP")
    if log_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            p.setlogfile(log_path)
        except Exception:
            pass

    # Time limit
    try:
        p.controls.maxtime = max_ILP_time
        #p.controls.timelimit   = 600      # seconds (set what you need) :contentReference[oaicite:0]{index=0}
       # p.controls.miprelstop  = 0.05     # stop at 5% MIP gap (tweak as desired) :contentReference[oaicite:1]{index=1}

        # Heuristic emphasis (primal focus)
        p.controls.heuremphasis     = 1   # 1 = focus on reducing gap early; 2 = very aggressive :contentReference[oaicite:2]{index=2}
       # p.controls.feasibilitypump  = 1   # run Feasibility Pump at root to get a feasible solution :contentReference[oaicite:3]{index=3}
        p.controls.heurfreq         = 1   # use heuristics frequently in the tree :contentReference[oaicite:4]{index=4}

        # Optional: enable pre-root parallel heuristics if you want even more primal push
        #p.controls.prerooteffort = 1      # >0 enables and dials effort for pre-root heuristics :contentReference[oaicite:5]{index=5}

        # Optional: de-emphasize proving bounds to spend more time on solutions
        #p.controls.cutstrategy = xp.CutStrategy.CONSERVATIVE   # fewer cuts → less time proving bounds :contentReference[oaicite:6]{index=6}
        #p.controls.cut
    except Exception:
        pass

    # Variables (use xp.var + addVariable to avoid keyword-arg issues)
    var_dict = {}
    for name, obj_coeff in safe_var_obj.items():
        lb = safe_var_LB.get(name, 0.0)
        ub = safe_var_UB.get(name, xp.infinity)
        if name in safe_binary_set:
            vtype = xp.binary
        elif name in safe_integer_set:
            vtype = xp.integer
        else:
            vtype = xp.continuous
        v = xp.var(lb=lb, ub=ub, vartype=vtype, name=name)
        p.addVariable(v)
        var_dict[name] = v

    # Group terms by constraint
    group_exog = defaultdict(list)
    for (v, c), coeff in safe_exog.items():
        group_exog[c].append((var_dict[v], coeff))

    group_eq = defaultdict(list)
    for (v, c), coeff in safe_eq_map.items():
        group_eq[c].append((var_dict[v], coeff))

    con_dict = {}

    # >= constraints (no keyword args; set name afterwards)
    for con_name, terms in group_exog.items():
        expr = xp.Sum(ci * vi for vi, ci in terms)
        con = p.addConstraint(expr >= safe_LB[con_name])
        try:
            con.name = con_name
        except Exception:
            pass
        con_dict[con_name] = con

    # == constraints
    for con_name, terms in group_eq.items():
        expr = xp.Sum(ci * vi for vi, ci in terms)
        con = p.addConstraint(expr == safe_EQ[con_name])
        try:
            con.name = con_name
        except Exception:
            pass
        con_dict[con_name] = con

    # Objective (minimize)
    obj_expr = xp.Sum(safe_var_obj[n] * var_dict[n] for n in safe_var_obj)
    p.setObjective(obj_expr, sense=xp.minimize)

    # MIP start (optional)
    if init_sol:
        vals, cols = [], []
        for v_alias, var in var_dict.items():
            orig = var_name_rev[v_alias]
            if orig in init_sol:
                vals.append(init_sol[orig])
                cols.append(var)
        if vals:
            try:
                p.addmipsol(vals, cols, "init")
            except Exception:
                pass

    time_pre = time.time() - time_pre

    # Solve
    p.write("my_ILP.mps")

    time_opt = time.time()
    p.optimize()
    time_opt = time.time() - time_opt
    # Extract solution
    vars_in_order = list(var_dict.values())
    vals = p.getSolution(vars_in_order)
    primal_solution = {var_name_rev[v.name]: val for v, val in zip(vars_in_order, vals)}

    # Objective & bound
    objective = None
    MIP_lower_bound = None
    try:
        objective = p.attributes.objval
    except Exception:
        pass
    try:
        MIP_lower_bound = p.attributes.bestbound
    except Exception:
        pass

    # Log string
    xpress_log_string = ""
    if log_path and os.path.isfile(log_path):
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as fh:
                xpress_log_string = fh.read()
        except Exception:
            pass

    time_post = 0.0
    return {
        "primal_solution": primal_solution,
        "objective": objective,
        "time_pre": time_pre,
        "time_opt": time_opt,
        "time_post": time_post,
        "MIP_lower_bound": MIP_lower_bound,
        "gurobi_log_string": xpress_log_string,  # kept same key for compatibility
    }


def solve_gurobi_lp_bounds(
    dict_var_name_2_obj,
    dict_var_con_2_lhs_exog,
    dict_con_name_2_LB,
    dict_var_con_2_lhs_eq,
    dict_con_name_2_eq,
    dict_var_name_2_LB,
    dict_var_name_2_UB,
    use_fast_interior=False,
    optTol=-1,
    xpress_auth_path=None,
    log_path="../ALL_JSON_BIG/xpress_lp_log.txt"
):
    _maybe_init_xpress(xpress_auth_path)

    t0_all = time.time()
    t_pre1 = time.time()
    #if use_fast_interior:
    #    input('r u sure')
    # Safe names
    var_names = list(dict_var_name_2_obj.keys())
    con_names_exog = list(dict_con_name_2_LB.keys())
    con_names_eq = list(dict_con_name_2_eq.keys())
    all_con_names = list(set(con_names_exog) | set(con_names_eq))

    var_name_map = {v: (v if len(v) < 20 else f"v{i}") for i, v in enumerate(var_names)}
    con_name_map = {c: f"c{i}" for i, c in enumerate(all_con_names)}
    var_name_rev = {v_alias: v for v, v_alias in var_name_map.items()}
    con_name_rev = {c_alias: c for c, c_alias in con_name_map.items()}

    safe_var_obj = {var_name_map[k]: v for k, v in dict_var_name_2_obj.items()}
    safe_exog = {(var_name_map[v], con_name_map[c]): coeff
                 for (v, c), coeff in dict_var_con_2_lhs_exog.items()}
    safe_eq_map = {(var_name_map[v], con_name_map[c]): coeff
                   for (v, c), coeff in dict_var_con_2_lhs_eq.items()}
    safe_LB = {con_name_map[k]: v for k, v in dict_con_name_2_LB.items()}
    safe_EQ = {con_name_map[k]: v for k, v in dict_con_name_2_eq.items()}
    safe_var_LB = {var_name_map[k]: v for k, v in dict_var_name_2_LB.items()}
    safe_var_UB = {var_name_map[k]: v for k, v in dict_var_name_2_UB.items()}
    t_pre1 = time.time() - t_pre1

    # Build LP
    p = xp.problem(name="converted_LP")
    #p.controls.crossover = 0   # 0 = no crossover (pure barrier)
    
    if log_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            p.setlogfile(log_path)
        except Exception:
            pass

    # Variables (robust creation)
    var_dict = {}
    t_pre2 = time.time()
    for name, obj_coeff in safe_var_obj.items():
        lb = safe_var_LB.get(name, 0.0)
        ub = safe_var_UB.get(name, xp.infinity)
        v = xp.var(lb=lb, ub=ub, vartype=xp.continuous, name=name)
        p.addVariable(v)
        var_dict[name] = v
    t_pre2 = time.time() - t_pre2

    # Group coefficients
    vget = var_dict.__getitem__
    get_con = lambda kv: kv[0][1]
    t_pre4 = time.time()
    ex_sorted = sorted(safe_exog.items(), key=get_con)
    group_exog = {}
    for con, grp in groupby(ex_sorted, key=get_con):
        group_exog[con] = [(vget(var), coeff) for (var, _), coeff in grp]

    eq_sorted = sorted(safe_eq_map.items(), key=get_con)
    group_eq = {}
    for con, grp in groupby(eq_sorted, key=get_con):
        group_eq[con] = [(vget(var), coeff) for (var, _), coeff in grp]
    t_pre4 = time.time() - t_pre4

    # Add constraints (no keyword args; set names after)
    t_pre5 = time.time()
    con_list = []
    for con_name, terms in group_exog.items():
        expr = xp.Sum(ci * vi for vi, ci in terms)
        c = xp.constraint(expr >= safe_LB[con_name], name=con_name)
        p.addConstraint(c)
        con_list.append(c)
    t_pre5 = time.time() - t_pre5

    t_pre6 = time.time()
    for con_name, terms in group_eq.items():
        expr = xp.Sum(ci * vi for vi, ci in terms)
        c = xp.constraint(expr == safe_EQ[con_name], name=con_name)
        p.addConstraint(c)
        con_list.append(c)
    t_pre6 = time.time() - t_pre6
    # Objective
    obj_expr = xp.Sum(safe_var_obj[n] * var_dict[n] for n in safe_var_obj)
    p.setObjective(obj_expr, sense=xp.minimize)

    time_pre = time.time() - t0_all
    #p.setOutputEnabled(False)
 
    # Solve (optionally Barrier w/out crossover)
    time_opt = time.time()
    if use_fast_interior:
        #input('here')
        p.controls.crossover = 0
        
        p.lpoptimize('b')
    else:
        p.optimize()
    if optTol>-0.001:
        p.setControl("optimalitytol",optTol)
    time_opt = time.time() - time_opt

    # Basic optimality check via objval availability
    try:
        obj = p.attributes.objval
    except Exception:
        obj = None
    if obj is None:
        try:
            p.write("errHere.lp", "lp")
        except Exception:
            pass
        raise RuntimeError("Xpress did not find an optimal LP solution.")

    # Primal solution
    vars_in_order = list(var_dict.values())
    vals = p.getSolution(vars_in_order)
    primal_solution = {var_name_rev[v.name]: val for v, val in zip(vars_in_order, vals)}
    dual_vals = p.getDual(con_list)
    rc_vals = p.getRCost(vars_in_order)
    dual_solution = { (con_name_rev.get(c.name, c.name)): dv for c, dv in zip(con_list, dual_vals) }
    reduced_costs = { var_name_rev[v.name]: rc for v, rc in zip(vars_in_order, rc_vals) }

    reduced_costs = {var_name_rev[v.name]: rc for v, rc in zip(vars_in_order, rc_vals)}

    # Log (optional)
    xpress_log_string = ""
    if log_path and os.path.isfile(log_path):
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as fh:
                xpress_log_string = fh.read()
        except Exception:
            pass

    time_post = 0.0
    return {
        "primal_solution": primal_solution,
        "dual_solution": dual_solution,
        "objective": obj,
        "time_pre": time_pre,
        "time_opt": time_opt,
        "time_post": time_post,
        "reduced_costs": reduced_costs,
        "xpress_log_string": xpress_log_string
    }
