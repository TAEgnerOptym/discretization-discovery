import xpress as xp
import time
import numpy as np
from  jy_fast_lp import jy_fast_lp
from  jy_fast_lp_gurobi import jy_fast_lp_gurobi
def warm_start_lp(lp_prob, var_dict, zero_names, flags='d'):
    """
    Warm‐start an LP by first fixing a subset of vars to zero,
    then solving the restricted and full LPs, returning the warmed problem.

    Args:
      lp_prob    : an xpress.problem
      var_dict   : dict mapping var_name -> xp.var object
      zero_names : iterable of var_name you believe should be 0
      flags      : solver flags for both solves (e.g. 'd' for dual simplex)

    Returns:
      lp_prob : the same xpress.problem, already solved with the warm start
    """
    lp_prob.controls.defaultalg = 1
    # 1) Gather the xp.var objects to fix
    zero_vars = [var_dict[name] for name in zero_names]

    # 2) Save original bounds from the var objects
    orig_bounds = {v: (v.lb, v.ub) for v in zero_vars}
    epsilon=.001
    lp_prob.chgbounds(zero_vars, ['U'] * len(zero_vars), [epsilon] * len(zero_vars))
    ##print('n')
    #pr#int(n)
    num_non_zero_entries=0
    all_vars=lp_prob.getVariable()
    
    
    # 4) Solve the restricted LP
    #lp_prob.setControl('xslp_deletioncontrol', 1)
    
    #lp_prob.controls.presolve = 1            # apply LP presolve and remove fixed‐column bounds :contentReference[oaicite:0]{index=0}

    # or equivalently:
    #lp_prob.setControl('presolve', 1)
    start_time1=time.time()
    lp_prob.solve(flags=flags)
    end_time1 = time.time()
    var_list=list(var_dict.values())
    #print(var_list)
    #print('var_list[0].type()')
    #print(type(var_list[0]))
    my_status_1=lp_prob.getProbStatus()
    my_obj_1=lp_prob.getObjVal()
    if my_status_1!=1:
        print('my_status')
        print(my_status_1)
        input('error here')
    vals = lp_prob.getSolution()
    

    #print(lp_prob.getProbStatus()=='feasible')
    new_lp_primal_solution = {
            var.name: vals[i]
            for i, var in enumerate(var_list)
        }
    tot_sum=0
    num_non_zero_entries=0
    for v in all_vars:
        if v.ub!=0:#and v.lb!=0:
            num_non_zero_entries=num_non_zero_entries+1 
        if v.ub<.001 and new_lp_primal_solution[v.name]>.001:
            print(' v.ub')
            print( v.ub)
            print('new_lp_primal_solution[v.name]')
            print(new_lp_primal_solution[v.name])
            input('error ')
        tot_sum=tot_sum+new_lp_primal_solution[v.name]
    #print('tot_sum')
    #print(tot_sum)
    #print('num_non_zero_entries')
    #print(num_non_zero_entries)
    time_lp_1=end_time1-start_time1
    # 5) Restore original bounds by resetting var attributes
    lp_prob.chgbounds(zero_vars, ['U'] * len(zero_vars), [np.inf] * len(zero_vars))

    # 6) Enable basis reuse
    #lp_prob.controls.defaultalg = 1#self.full_prob.jy_opt['lplb_solver']

    lp_prob.controls.keepbasis = 1

    # 7) Resolve the full LP from warm basis
    start_time2=time.time()
    lp_prob.solve(flags=flags)
    end_time2 = time.time()
    time_lp_2 = end_time2 - start_time2
    my_obj_2=lp_prob.getObjVal()
    my_status_2=lp_prob.getProbStatus()
    my_obj_2=lp_prob.getObjVal()
    if my_status_1!=1:
        print('my_status')
        print(my_status_1)
        input('error here')
    if my_obj_2>my_obj_1+.001:
        print('my_obj_2')
        print(my_obj_2)
        print('my_obj_1')
        print(my_obj_1)
        input('error here')
    print('my_obj_2')
    print(my_obj_2)
    print('my_obj_1')
    print(my_obj_1)
    print('time_lp_1')
    print(time_lp_1)
    print('time_lp_2')
    print(time_lp_2)
    # 8) Return the warmed LP problem
    return lp_prob,time_lp_1,time_lp_2



import xpress as xp
import numpy as np
import time

def forbidden_variables_loop(lp_prob, var_dict, forbidden_var_names, epsilon=1e-6, K=2, verbose=True):
    """
    Solve LP while progressively relaxing forbidden variable bounds based on primal values.

    Parameters:
    lp_prob: xpress.problem
        The LP model.
    forbidden_var_names: list or set
        Names of variables to restrict initially.
    epsilon: float
        Small positive value to set upper bounds.
    K: int
        Maximum number of variables to relax per iteration (default 10).
    verbose: bool
        Whether to print progress.

    Returns:
    lp_prob: xpress.problem
        The solved LP after adjustments.
    """
    forbidden_var_names = set(forbidden_var_names)

    # Mapping from name to variable
    all_vars = {v.name: v for v in lp_prob.getVariable()}

    # Only keep existing variables
    forbidden_var_names &= set(all_vars.keys())
    vars_list = lp_prob.getVariable()
    
    lp_prob.controls.defaultalg = 1  # primal simplex

    if not forbidden_var_names:
        if verbose:
            print("No forbidden variables found in model.")
        return lp_prob, 0.0

    # Initial bound change for forbidden variables
    vars_to_restrict = [all_vars[name] for name in forbidden_var_names]
    lp_prob.chgbounds(vars_to_restrict, ['U'] * len(vars_to_restrict), [epsilon] * len(vars_to_restrict))

    iteration = 0
    total_solve_time = 0.0

    while True:
        iteration += 1
        if verbose:
            print(f"\n--- Iteration {iteration} ---")
            print(f"Forbidden variables to check: {len(forbidden_var_names)}")

        # Solve and measure time
        t_start = time.time()
        lp_prob.solve()
        t_end = time.time()

        solve_time = t_end - t_start
        total_solve_time += solve_time

        if verbose:
            print(f"Solve time: {solve_time:.4f} seconds")
            print(f"Objective value: {lp_prob.getObjVal()}")

        # Fetch primal solution
        vals = lp_prob.getSolution()
        lp_primal_solution = {
            var.name: vals[i]
            for i, var in enumerate(vars_list)
        }

        # Identify forbidden vars with nonzero primal values
        active_vars_with_vals = [
            (v, lp_primal_solution.get(v.name, 0.0))
            for v in vars_to_restrict
            if v.name in forbidden_var_names and abs(lp_primal_solution.get(v.name, 0.0)) > 1e-8
        ]

        if verbose:
            print(f"Active forbidden variables with nonzero values: {len(active_vars_with_vals)}")

        if not active_vars_with_vals:
            if verbose:
                print("No more forbidden variables with active primal values. Done!")
                print(f"Total LP solve time: {total_solve_time:.4f} seconds")
            break

        # Sort by primal value descending, pick top-K
        active_vars_with_vals.sort(key=lambda x: abs(x[1]), reverse=True)
        selected_active_vars = [v for v, _ in active_vars_with_vals[:K]]

        if verbose:
            print(f"Relaxing {len(selected_active_vars)} variables this iteration.")

        # Remove selected variables from forbidden set
        forbidden_var_names -= {v.name for v in selected_active_vars}

        # Reset bounds (make them free again)
        lp_prob.chgbounds(selected_active_vars, ['U'] * len(selected_active_vars), [np.inf] * len(selected_active_vars))

        # Update vars_to_restrict for next iteration
        vars_to_restrict = [all_vars[name] for name in forbidden_var_names]

    if verbose:
        print("\nFinished forbidden_variables_loop.")
    return lp_prob, total_solve_time



def forbidden_variables_loop_dual(lp_prob, var_dict, forbidden_var_names, K=20, verbose=False):
    """
    Solve LP while progressively relaxing forbidden variable bounds based on primal values.

    Parameters:
    lp_prob: xpress.problem
        The LP model.
    forbidden_var_names: list or set
        Names of variables to restrict initially.
    epsilon: float
        Small positive value to set upper bounds.
    K: int
        Maximum number of variables to relax per iteration (default 10).
    verbose: bool
        Whether to print progress.

    Returns:
    lp_prob: xpress.problem
        The solved LP after adjustments.
    """
    forbidden_var_names = set(forbidden_var_names)

    # Mapping from name to variable
    all_vars = {v.name: v for v in lp_prob.getVariable()}

    # Only keep existing variables
    forbidden_var_names &= set(all_vars.keys())
    vars_list = lp_prob.getVariable()
    
    lp_prob.controls.defaultalg = 1  # primal simplex

    if not forbidden_var_names:
        if verbose:
            print("No forbidden variables found in model.")
            input('---')
        return lp_prob, 0.0

    # Initial bound change for forbidden variables
    vars_to_restrict = [all_vars[name] for name in forbidden_var_names]
    lp_prob.chgbounds(vars_to_restrict, ['U'] * len(vars_to_restrict), [0] * len(vars_to_restrict))

    iteration = 0
    total_solve_time = 0.0

    while True:
        iteration += 1
        if verbose:
            print(f"\n--- Iteration {iteration} ---")
            print(f"Forbidden variables to check: {len(forbidden_var_names)}")

        # Solve and measure time
        t_start = time.time()
        lp_prob.solve()
        t_end = time.time()

        solve_time = t_end - t_start
        total_solve_time += solve_time

        if verbose:
            print(f"Solve time: {solve_time:.4f} seconds")
            print(f"Objective value: {lp_prob.getObjVal()}")

        # Fetch primal solution
        vals = lp_prob.getSolution()
        lp_primal_solution = {
            var.name: vals[i]
            for i, var in enumerate(vars_list)
        }

        # Identify forbidden vars with nonzero primal values
        reduced_costs = lp_prob.getRCost()

        # Build mapping
        reduced_costs_dict = {var.name: reduced_costs[i] for i, var in enumerate(vars_list)}

        # Find active forbidden variables by reduced cost
        active_vars_with_vals = [
            (v, reduced_costs_dict.get(v.name, 0.0))
            for v in vars_to_restrict
            if v.name in forbidden_var_names and reduced_costs_dict.get(v.name, 0.0) < -1e-8
        ]

        if verbose:
            print(f"Active forbidden variables with nonzero values: {len(active_vars_with_vals)}")

        if not active_vars_with_vals:
            if verbose:
                print("No more forbidden variables with active primal values. Done!")
                print(f"Total LP solve time: {total_solve_time:.4f} seconds")
            break

        # Sort by primal value descending, pick top-K
        active_vars_with_vals.sort(key=lambda x: -x[1], reverse=True)
        selected_active_vars = [v for v, _ in active_vars_with_vals[:K]]

        if verbose:
            print(f"Relaxing {len(selected_active_vars)} variables this iteration.")

        # Remove selected variables from forbidden set
        forbidden_var_names -= {v.name for v in selected_active_vars}

        # Reset bounds (make them free again)
        lp_prob.chgbounds(selected_active_vars, ['U'] * len(selected_active_vars), [np.inf] * len(selected_active_vars))

        # Update vars_to_restrict for next iteration
        vars_to_restrict = [all_vars[name] for name in forbidden_var_names]

    if verbose:
        print("\nFinished forbidden_variables_loop.")

    debugOn=False
    if debugOn==True:
        input('debug on ')
        old_val=lp_prob.getObjVal()
        lp_prob.controls.defaultalg = 4  # primal simplex

        lp_prob.chgbounds(vars_list, ['U'] * len(vars_list), [1000000000] * len(vars_list))
        t_bakcup=time.time()
        lp_prob.solve()
        t_bakcup=time.time()-t_bakcup
        new_val=lp_prob.getObjVal()
        if abs(new_val-old_val)>.001:
            input('error')
        print('new_val')
        print(new_val)
        print('new_val')
        print(new_val)
        print('t_bakcup')
        print(t_bakcup)
        print('total_solve_time')
        print(total_solve_time)
        input('done debug')
    return lp_prob, total_solve_time


def warm_start_lp_using_class(lp_prob, var_dict,all_possible_forbidden_names,cur_forbidden_name):

    my_jy_lp_fast=jy_fast_lp(lp_prob, var_dict,all_possible_forbidden_names,cur_forbidden_name)
    return [my_jy_lp_fast.lp_prob,my_jy_lp_fast.tot_lp_time]

def warm_start_lp_using_class_gurobi(dict_var_name_2_obj,
                    dict_var_con_2_lhs_exog,
                    dict_con_name_2_LB,
                    dict_var_con_2_lhs_eq,
                    dict_con_name_2_eq, all_possible_forbidden_names,cur_forbidden_name):

    my_jy_lp_fast=jy_fast_lp_gurobi(dict_var_name_2_obj,
                    dict_var_con_2_lhs_exog,
                    dict_con_name_2_LB,
                    dict_var_con_2_lhs_eq,
                    dict_con_name_2_eq,all_possible_forbidden_names,cur_forbidden_name)
    return [my_jy_lp_fast,my_jy_lp_fast.tot_lp_time]