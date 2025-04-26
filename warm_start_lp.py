import xpress as xp
import time
import numpy as np
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
    lp_prob.controls.defaultalg = 2
    # 1) Gather the xp.var objects to fix
    zero_vars = [var_dict[name] for name in zero_names]

    # 2) Save original bounds from the var objects
    orig_bounds = {v: (v.lb, v.ub) for v in zero_vars}

    lp_prob.chgbounds(zero_vars, ['U'] * len(zero_vars), [0] * len(zero_vars))
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
