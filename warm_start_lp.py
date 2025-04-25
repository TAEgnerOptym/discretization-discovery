import xpress as xp

def restricted_warm_start(lp_prob, var_dict, zero_names, flags='d'):
    """
    Warm-start an LP by first fixing a subset of vars to zero.

    Args:
      lp_prob    : an xpress.problem
      var_dict   : dict mapping var_name -> xp.var object
      zero_names : iterable of var_name you believe should be 0
      flags      : solver flags for both solves (e.g. 'd' for dual simplex)

    Returns:
      (restricted_sol, full_sol), each a dict var_name -> value
    """
    # 1) Gather the xp.var objects
    zero_vars = [var_dict[n] for n in zero_names]

    # 2) Save original bounds
    orig_bounds = {
        v: (lp_prob.getlb(v), lp_prob.getub(v))
        for v in zero_vars
    }  # :contentReference[oaicite:0]{index=0}

    # 3) Fix them to zero
    for v in zero_vars:
        lp_prob.chgbounds(v, 0.0, 0.0)    # :contentReference[oaicite:1]{index=1}

    # 4) Solve the restricted LP
    lp_prob.solve(flags=flags)           # :contentReference[oaicite:2]{index=2}
    restricted_sol = {
        name: lp_prob.getSolution(var)   # :contentReference[oaicite:3]{index=3}
        for name, var in var_dict.items()
    }

    # 5) Restore all bounds
    for v, (lb, ub) in orig_bounds.items():
        lp_prob.chgbounds(v, lb, ub)    # :contentReference[oaicite:4]{index=4}

    # 6) Tell Xpress to keep the existing basis
    lp_prob.controls.keepbasis = 1     # :contentReference[oaicite:5]{index=5}

    # 7) Resolve full model from that warm basis
    lp_prob.solve(flags=flags)
    full_sol = {
        name: lp_prob.getSolution(var)
        for name, var in var_dict.items()
    }

    return restricted_sol, full_sol
