import gurobipy as gp
import time
import numpy as np
from collections import defaultdict

class jy_fast_lp_gurobi:

    def __init__(self, dict_var_name_2_obj,
                 dict_var_con_2_lhs_exog,
                 dict_con_name_2_LB,
                 dict_var_con_2_lhs_eq,
                 dict_con_name_2_eq,
                 all_possible_forbidden_names,
                 init_forbidden_names,
                 K=20, verbose=False, remove_choice=3, alg_use=1, debug_on=False,
                 min_improvement_dump=0.1, epsilon=1e-4):

        self.verbose = verbose
        self.remove_choice = remove_choice
        self.alg_use = alg_use
        self.min_improvement_dump = min_improvement_dump
        self.epsilon = epsilon
        self.debug_on = debug_on
        self.hist = {'lp': [], 'numCurMid': [], 'numCurEnd': [], 'time_iter': []}
        self.max_terms_add_per_round = K
        self.all_possible_forbidden_names = all_possible_forbidden_names
        self.init_forbidden_names = init_forbidden_names

        (self.model, self.var_dict,
         self.var_name_rev_map, self.con_name_rev_map) = self.build_model_safe(
            dict_var_name_2_obj,
            dict_var_con_2_lhs_exog,
            dict_con_name_2_LB,
            dict_var_con_2_lhs_eq,
            dict_con_name_2_eq)

        self.setup_alg()
        self.call_core_alg()
        if debug_on:
            self.backup_lp()

        if verbose:
            input('Done call')

    def build_model_safe(self, dict_var_name_2_obj,
                         dict_var_con_2_lhs_exog,
                         dict_con_name_2_LB,
                         dict_var_con_2_lhs_eq,
                         dict_con_name_2_eq):

        model = gp.Model("safe_lp")
        model.setParam("OutputFlag", 0)

        # Create safe names
        original_vars = list(dict_var_name_2_obj.keys())
        original_cons = list(set(dict_con_name_2_LB.keys()) | set(dict_con_name_2_eq.keys()))
        var_name_map = {name: f"v{i}" for i, name in enumerate(original_vars)}
        con_name_map = {name: f"c{i}" for i, name in enumerate(original_cons)}
        var_name_rev_map = {v: k for k, v in var_name_map.items()}
        con_name_rev_map = {v: k for k, v in con_name_map.items()}

        # Apply mapping to all data
        safe_var_obj = {var_name_map[k]: v for k, v in dict_var_name_2_obj.items()}
        safe_exog = {(var_name_map[v], con_name_map[c]): coeff for (v, c), coeff in dict_var_con_2_lhs_exog.items()}
        safe_eq_map = {(var_name_map[v], con_name_map[c]): coeff for (v, c), coeff in dict_var_con_2_lhs_eq.items()}
        safe_LB = {con_name_map[k]: v for k, v in dict_con_name_2_LB.items()}
        safe_EQ = {con_name_map[k]: v for k, v in dict_con_name_2_eq.items()}

        # Create variables
        var_dict = {name: model.addVar(lb=0, obj=obj, name=name) for name, obj in safe_var_obj.items()}
        model.update()

        # Group constraints
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

        model.ModelSense = gp.GRB.MINIMIZE
        return model, var_dict, var_name_rev_map, con_name_rev_map

    def setup_alg(self):
        self.forbidden_var_names = set()
        self.tot_lp_time = 0
        self.all_removable_vars = [self.var_dict[f"v{self.var_name_rev_map[name]}"] if name in self.var_name_rev_map else self.var_dict[name] for name in self.all_possible_forbidden_names]
        self.vars_list = list(self.model.getVars())

    def call_core_alg(self):
        self.incumbent_lp_val = np.inf
        self.current_forbidden_vars = set()
        self.forbidden_var_names = set()

        vars_to_forbid = [self.var_dict[var] for var in self.var_dict if self.var_name_rev_map.get(var, var) in self.init_forbidden_names]
        self.add_to_forbidden(vars_to_forbid)

        if len(self.forbidden_var_names) == 0:
            print("No forbidden variables found in model. This may be a typo.")
            input('---')

        iter = 0
        while True:
            iter += 1
            self.compute_bound()
            self.hist['lp'].append(self.lp_obj_val)

            if self.lp_obj_val > self.incumbent_lp_val + 0.01:
                input('Bound increased unexpectedly')

            if len(self.forbidden_vars_with_neg_red_cost) < 0.5:
                break

            if self.lp_obj_val < self.incumbent_lp_val - self.min_improvement_dump:
                self.incumbent_lp_val = self.lp_obj_val
                if self.remove_choice == 2:
                    self.remove_all_non_pos_after_improvement()
                elif self.remove_choice == 3:
                    self.remove_all_pos_red_cost_after_improvement()

            self.hist['numCurMid'].append(len(self.forbidden_var_names))
            self.add_neg_red_cost_vars()
            self.hist['numCurEnd'].append(len(self.forbidden_var_names))

            if self.verbose:
                print(f"Iter {iter}: LP={self.lp_obj_val}, Incumbent={self.incumbent_lp_val}, "
                      f"Neg RC={len(self.forbidden_vars_with_neg_red_cost)}, Forbidden={len(self.forbidden_var_names)}")
