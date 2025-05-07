from collections import defaultdict
import pulp
import numpy as np
import time

def jy_active_set_lp(dict_var_name_2_obj,
                     dict_var_con_2_lhs_exog,
                     dict_con_name_2_LB,
                     dict_var_con_2_lhs_eq,
                     dict_con_name_2_eq,
                     init_constraint_ineq_names=None,
                     init_constraint_eq_names=None,
                     tol=1e-6,           # tolerance for constraint violation
                     tol_bound=1e-4,     # tolerance for bound change to trigger cleanup (if cleanup_final_only is False)
                     tol_dual=1e-6,      # tolerance for inactive dual (near zero)
                     max_iter=100,       # maximum iterations for the active set procedure
                     cleanup_final_only=False,
                     cleanup_time_threshold=1.0  # threshold in seconds; cleanup only performed if total LP time exceeds this
                     ):
    """
    Build and solve an LP with an active-set approach.

    This function uses all variables (from dict_var_name_2_obj) and starts
    with an initial (possibly empty) set of active constraints.
    It iteratively solves a restricted LP and:
       - Adds any candidate inequality or equality constraint that is violated.
       - Optionally removes inactive (dual near zero) inequality constraints either
         continuously (if cleanup_final_only is False) or only on the final iteration 
         (if cleanup_final_only is True), but only if total LP-solving time exceeds cleanup_time_threshold.

    Inputs:
      - dict_var_name_2_obj: dict mapping variable names to objective coefficients.
      - dict_var_con_2_lhs_exog: dict mapping (var_name, ineq_name) to coefficient (for inequalities).
      - dict_con_name_2_LB: dict mapping inequality constraint names to lower bounds.
      - dict_var_con_2_lhs_eq: dict mapping (var_name, eq_name) to coefficient (for equalities).
      - dict_con_name_2_eq: dict mapping equality constraint names to right-hand side values.
      - init_constraint_ineq_names: (optional) iterable of inequality constraint names to start with.
      - init_constraint_eq_names: (optional) iterable of equality constraint names to start with.
      - tol: tolerance for constraint violation.
      - tol_bound: required LP bound change for triggering cleanup (if cleanup_final_only is False).
      - tol_dual: dual value below which a constraint is considered inactive.
      - max_iter: maximum active-set iterations.
      - cleanup_final_only: if True, defer removal of inactive inequality constraints until the final iteration.
      - cleanup_time_threshold: cleanup is performed only if total LP solving time exceeds this threshold (in seconds).
    
    Returns:
      (lp_prob, var_dict, active_ineq, active_eq, total_lp_time)
        - lp_prob: the final PuLP problem.
        - var_dict: dict mapping variable names to PuLP variables (.varValue holds the solution).
        - active_ineq: set of inequality constraint names active in the final LP.
        - active_eq: set of equality constraint names active in the final LP.
        - total_lp_time: total time spent in LP solves (in seconds).
    """

    # Precompute the full candidate constraint names.
    all_ineq_names = list(dict_con_name_2_LB.keys())
    all_eq_names   = list(dict_con_name_2_eq.keys())
    
    # Precompute mapping for inequality constraints: cons_name -> list of (var, coeff)
    ineq_terms_map = {}
    for (var, cons_name), coeff in dict_var_con_2_lhs_exog.items():
        ineq_terms_map.setdefault(cons_name, []).append((var, coeff))
        
    # Precompute mapping for equality constraints: cons_name -> list of (var, coeff)
    eq_terms_map = {}
    for (var, cons_name), coeff in dict_var_con_2_lhs_eq.items():
        eq_terms_map.setdefault(cons_name, []).append((var, coeff))
    
    # Initialize active constraint sets.
    active_ineq = set(init_constraint_ineq_names) if init_constraint_ineq_names is not None else set()
    active_eq   = set(init_constraint_eq_names) if init_constraint_eq_names is not None else set()
    
    total_lp_time = 0.0
    iter_count = 0
    prev_bound = None   # store previous LP objective bound

    while True:
        iter_count += 1
        if iter_count > max_iter:
            print("Max iterations reached; stopping active set iterations.")
            break
        
        # Build the Restricted Master Problem (RMP) with the current active constraints.
        lp_prob = pulp.LpProblem("ActiveSetLP", pulp.LpMinimize)
        
        # Create all variables (assume continuous nonnegative).
        var_dict = {var_name: pulp.LpVariable(var_name, lowBound=0)
                    for var_name in dict_var_name_2_obj.keys()}
        
        # Set the objective function.
        lp_prob += pulp.lpSum(dict_var_name_2_obj[var_name] * var_dict[var_name] 
                              for var_name in var_dict), "Objective"
        
        # Add active inequality constraints.
        for ineq_name in active_ineq:
            terms = ineq_terms_map.get(ineq_name, [])
            expr = pulp.lpSum(coeff * var_dict[var] for var, coeff in terms)
            lp_prob += expr >= dict_con_name_2_LB[ineq_name], ineq_name
        
        # Add active equality constraints.
        for eq_name in active_eq:
            terms = eq_terms_map.get(eq_name, [])
            expr = pulp.lpSum(coeff * var_dict[var] for var, coeff in terms)
            lp_prob += expr == dict_con_name_2_eq[eq_name], eq_name
        
        # Solve the current LP and measure the time taken.
        t_start = time.time()
        lp_prob.solve(pulp.PULP_CBC_CMD(msg=False))
        t_end = time.time()
        total_lp_time += (t_end - t_start)
        
        # Get current LP objective value.
        curr_bound = pulp.value(lp_prob.objective)
        
        # Cleanup of inactive inequality constraints
        if (not cleanup_final_only and total_lp_time >= cleanup_time_threshold 
            and prev_bound is not None and abs(prev_bound - curr_bound) >= tol_bound):
            removed = []
            for ineq_name in list(active_ineq):
                if ineq_name in lp_prob.constraints:
                    dual_val = lp_prob.constraints[ineq_name].pi
                    if dual_val is not None and abs(dual_val) < tol_dual:
                        active_ineq.remove(ineq_name)
                        removed.append(ineq_name)
            if removed:
                print(f"Iteration {iter_count}: Removed inactive inequalities {removed}")
        prev_bound = curr_bound
        
        new_constraint_added = False
        
        # Check inequality constraints that are not yet active.
        for ineq_name in all_ineq_names:
            if ineq_name in active_ineq:
                continue
            terms = ineq_terms_map.get(ineq_name, [])
            # Compute LHS value as dot product.
            lhs_val = sum(coeff * (var_dict[var].varValue if var_dict[var].varValue is not None else 0)
                          for var, coeff in terms)
            required_val = dict_con_name_2_LB[ineq_name]
            if lhs_val < required_val - tol:
                active_ineq.add(ineq_name)
                new_constraint_added = True
                print(f"Iteration {iter_count}: Added inequality '{ineq_name}' (LHS = {lhs_val}, required >= {required_val})")
        
        # Check equality constraints that are not yet active.
        for eq_name in all_eq_names:
            if eq_name in active_eq:
                continue
            terms = eq_terms_map.get(eq_name, [])
            lhs_val = sum(coeff * (var_dict[var].varValue if var_dict[var].varValue is not None else 0)
                          for var, coeff in terms)
            required_val = dict_con_name_2_eq[eq_name]
            if abs(lhs_val - required_val) > tol:
                active_eq.add(eq_name)
                new_constraint_added = True
                print(f"Iteration {iter_count}: Added equality '{eq_name}' (LHS = {lhs_val}, required == {required_val})")
        
        if not new_constraint_added:
            # On the final iteration, if cleanup_final_only is True and if LP time exceeds threshold, do cleanup.
            if cleanup_final_only and total_lp_time >= cleanup_time_threshold:
                removed = []
                for ineq_name in list(active_ineq):
                    if ineq_name in lp_prob.constraints:
                        dual_val = lp_prob.constraints[ineq_name].pi
                        if dual_val is not None and abs(dual_val) < tol_dual:
                            active_ineq.remove(ineq_name)
                            removed.append(ineq_name)
                if removed:
                    print(f"Final cleanup: Removed inactive inequalities {removed}")
            print("No new constraints added. Active set optimization complete.")
            break
        else:
            print(f"Iteration {iter_count}: New constraints added. Resolving...")
    
    final_solution = {var: var_dict[var].varValue for var in var_dict}
    print(f"Total LP solving time: {total_lp_time:.6f} seconds")
    
    return lp_prob, var_dict, active_ineq, active_eq, total_lp_time





def jy_active_set_lp_primal_dual(dict_var_name_2_obj,
                                dict_var_con_2_lhs_exog,
                                dict_con_name_2_LB,
                                dict_var_con_2_lhs_eq,
                                dict_con_name_2_eq,
                                init_active_primal=None,
                                init_constraint_ineq_names=None,
                                init_constraint_eq_names=None,
                                tol=1e-6,           # tolerance for constraint violation
                                tol_bound=1e-4,     # tolerance for bound change to trigger cleanup
                                tol_dual=1e-6,      # tolerance for inactive dual (near zero)
                                max_iter=100,       # maximum iterations in the outer loop
                                cleanup_final_only=False,
                                cleanup_time_threshold=1.0,  # seconds threshold for performing cleanup
                                M=1e6              # high penalty cost for slack variables
                                ):
    """
    Active-set primal–dual LP algorithm.
    
    This function alternates between (dual) row generation and (primal) column generation.
    
    - The full set of variables (columns) is defined in dict_var_name_2_obj.
    - We maintain an active set (restricted master) of primal variables (active_primal).
    Initially, if init_active_primal is provided, we use that; otherwise, we start empty.
    - We also maintain active inequality and equality constraint sets (active_ineq, active_eq),
    which are added via dual pricing (as in the previous function).
    - When a candidate variable (column) not in active_primal has negative reduced cost,
    we add it to active_primal. To avoid infeasibility, we also introduce an artificial slack
    variable for that column and add the equality constraint x_j - s_j = 0 with a high penalty M.
    
    Returns:
    (lp_prob, var_dict, active_ineq, active_eq, active_primal, total_lp_time)
    
    where var_dict includes both the originally added primal variables and any slack variables.
    """
    
    # Full candidate sets for constraints.
    all_ineq_names = list(dict_con_name_2_LB.keys())
    all_eq_names   = list(dict_con_name_2_eq.keys())
    
    # Precompute mapping for inequality constraints:
    ineq_terms_map = defaultdict(list)
    for (var, cons), coeff in dict_var_con_2_lhs_exog.items():
        ineq_terms_map[cons].append((var, coeff))
    
    # Precompute mapping for equality constraints:
    eq_terms_map = defaultdict(list)
    for (var, cons), coeff in dict_var_con_2_lhs_eq.items():
        eq_terms_map[cons].append((var, coeff))
    
    # Precompute candidate mapping for each variable.
    # candidate_map: var -> {'ineq': list of (cons, coeff), 'eq': list of (cons, coeff)}
    candidate_map = {var: {'ineq': [], 'eq': []} for var in dict_var_name_2_obj}
    for (var, cons), coeff in dict_var_con_2_lhs_exog.items():
        candidate_map[var]['ineq'].append((cons, coeff))
    for (var, cons), coeff in dict_var_con_2_lhs_eq.items():
        candidate_map[var]['eq'].append((cons, coeff))
    
    # Initialize active sets.
    active_primal = set(init_active_primal) if init_active_primal is not None else set()
    active_ineq   = set(init_constraint_ineq_names) if init_constraint_ineq_names is not None else set()
    active_eq     = set(init_constraint_eq_names) if init_constraint_eq_names is not None else set()
    
    # Dictionary to store slack variables for added columns.
    slack_dict = {}  # key: variable name, value: slack variable
    
    total_lp_time = 0.0
    iter_count = 0
    prev_bound = None  # previous LP objective
    
    while True:
        iter_count += 1
        print('iter_count')
        print(iter_count)
        if iter_count > max_iter:
            print("Max iterations reached; terminating.")
            break
        
        # Build restricted master LP using current active_primal, active constraints.
        lp_prob = pulp.LpProblem("ActiveSetLP_PrimalDual", pulp.LpMinimize)
        
        # Create decision variables for active primal variables.
        var_dict = {var: pulp.LpVariable(var, lowBound=0)
                    for var in active_primal}
        # Add slack variables (already added for some columns).
        for var in slack_dict:
            var_dict["slack_" + var] = slack_dict[var]
        
        # Set objective: sum_{j in active_primal} c_j*x_j + M * (sum of slack variables)
        lp_prob += (pulp.lpSum(dict_var_name_2_obj[var] * var_dict[var] for var in active_primal) +
                    pulp.lpSum(M * var_dict["slack_" + var] for var in slack_dict)), "Objective"
        
        # Add active inequality constraints.
        for con in active_ineq:
            # For each constraint, sum over all variables that appear.
            expr = pulp.lpSum(coeff * var_dict[var] 
                            for var, coeff in ineq_terms_map.get(con, [])
                            if var in active_primal)
            lp_prob += expr >= dict_con_name_2_LB[con], con
        
        # Add active equality constraints.
        for con in active_eq:
            expr = pulp.lpSum(coeff * var_dict[var]
                            for var, coeff in eq_terms_map.get(con, [])
                            if var in active_primal)
            lp_prob += expr == dict_con_name_2_eq[con], con
        
        # For every new column that was added with slack, add the constraint: x_j - s_j = 0.
        for var in slack_dict:
            lp_prob += var_dict[var] - var_dict["slack_" + var] == 0, "col_fix_" + var
        
        # Solve the restricted master LP.
        t_start = time.time()
        lp_prob.solve(pulp.PULP_CBC_CMD(msg=False))
        t_end = time.time()
        total_lp_time += (t_end - t_start)
        
        curr_bound = pulp.value(lp_prob.objective)
        
        # (Optional) Cleanup of inactive dual rows could be done here as in previous functions.
        # [Omitted for brevity; see jy_active_set_lp for details.]
        
        new_element_added = False  # flag if any new row/column is added this iteration.
        
        # ----- Dual (Row) Generation: Check candidate constraints -----
        # Check inactive inequality constraints.
        for con in all_ineq_names:
            if con in active_ineq:
                continue
            lhs_val = sum(coeff * (var_dict[var].varValue if var in var_dict and var_dict[var].varValue is not None else 0)
                        for var, coeff in ineq_terms_map.get(con, [])
                        if var in active_primal)
            if lhs_val < dict_con_name_2_LB[con] - tol:
                active_ineq.add(con)
                new_element_added = True
                print(f"Iteration {iter_count}: Added inequality constraint {con}")
        
        # Check inactive equality constraints.
        for con in all_eq_names:
            if con in active_eq:
                continue
            lhs_val = sum(coeff * (var_dict[var].varValue if var in var_dict and var_dict[var].varValue is not None else 0)
                        for var, coeff in eq_terms_map.get(con, [])
                        if var in active_primal)
            if abs(lhs_val - dict_con_name_2_eq[con]) > tol:
                active_eq.add(con)
                new_element_added = True
                print(f"Iteration {iter_count}: Added equality constraint {con}")
        
        # ----- Primal (Column) Generation: Check candidate variables -----
        # For each variable not yet in active_primal, compute its reduced cost.
        for var in dict_var_name_2_obj:
            if var in active_primal:
                continue
            rc = dict_var_name_2_obj[var]
            # Subtract contributions from active inequality constraints.
            for con, coeff in candidate_map[var]['ineq']:
                if con in lp_prob.constraints:
                    dual = lp_prob.constraints[con].pi
                    rc -= (dual if dual is not None else 0) * coeff
            for con, coeff in candidate_map[var]['eq']:
                if con in lp_prob.constraints:
                    dual = lp_prob.constraints[con].pi
                    rc -= (dual if dual is not None else 0) * coeff
            if rc < -tol:
                # Candidate variable is promising; add it.
                active_primal.add(var)
                # Create its slack variable with high penalty.
                slack_var = pulp.LpVariable("slack_" + var, lowBound=0)
                slack_dict[var] = slack_var
                new_element_added = True
                print(f"Iteration {iter_count}: Added primal variable {var} with reduced cost {rc}")
        
        if not new_element_added:
            if cleanup_final_only:
                # If deferred cleanup is enabled, perform it now (not shown here for brevity).
                pass
            print("No new constraints or columns added. Algorithm terminated.")
            break
        else:
            print(f"Iteration {iter_count}: New rows/columns added. Resolving...")
        
    # Save the final solution.
    final_solution = {var: var_dict[var].varValue for var in var_dict}
    print(f"Total LP solve time: {total_lp_time:.6f} seconds")
    return lp_prob, var_dict, active_ineq, active_eq, active_primal, total_lp_time
