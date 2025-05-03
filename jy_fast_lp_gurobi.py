import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import time
import numpy as np
class jy_fast_lp_gurobi:


    def add_to_forbidden(self, vars_to_remove):
        #input('this may not work here')
        vars_to_remove = list(set(vars_to_remove) - self.current_forbidden_vars)
        for var in vars_to_remove:
            var.ub = 0
            self.forbidden_var_names.add(self.var_name_rev_map[var.VarName])
            self.current_forbidden_vars.add(var)

    def remove_from_forbidden(self, vars_to_add):
        #input('this may not work here')

        for var in vars_to_add:
            var.ub = gp.GRB.INFINITY
            self.forbidden_var_names.remove(self.var_name_rev_map[var.VarName])
            self.current_forbidden_vars.remove(var)

    def remove_all_pos_red_cost_after_improvement(self):
        #input('this may not work here')

        self.add_to_forbidden(self.pos_red_cost_removable)

    def remove_all_non_pos_after_improvement(self):
        #input('this may not work here')

        self.add_to_forbidden(self.inactive_removable_vars)

    def add_neg_red_cost_vars(self):
        #input('this may not work here')

        self.forbidden_vars_with_neg_red_cost.sort(key=lambda x: -x[1])
        selected = [v for v, _ in self.forbidden_vars_with_neg_red_cost[:self.max_terms_add_per_round]]
        self.remove_from_forbidden(selected)



    def __init__(self, dict_var_name_2_obj,
                 dict_var_con_2_lhs_exog,
                 dict_con_name_2_LB,
                 dict_var_con_2_lhs_eq,
                 dict_con_name_2_eq,
                 all_possible_forbidden_names,
                 init_forbidden_names,
                 K=100, verbose=True, remove_choice=3, alg_use=1, debug_on=False,
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

        options = {
                "WLSACCESSID": "8f7bb9d6-8fe5-4349-9dd3-6abbaa9199a0",
                "WLSSECRET": "cb02810a-e0e2-4a1f-8fc0-fd375f65fc65",
                "LICENSEID": 2660300
            }


        with gp.Env(params=options) as env:
            with gp.Model("converted_LP", env=env) as model:
                model.setParam("OutputFlag", 0)

                original_vars = list(dict_var_name_2_obj.keys())
                original_cons = list(set(dict_con_name_2_LB.keys()) | set(dict_con_name_2_eq.keys()))
                var_name_map = {name: f"v{i}" for i, name in enumerate(original_vars)}
                con_name_map = {name: f"c{i}" for i, name in enumerate(original_cons)}
                self.var_name_rev_map = {v: k for k, v in var_name_map.items()}
                self.con_name_rev_map = {v: k for k, v in con_name_map.items()}

                safe_var_obj = {var_name_map[k]: v for k, v in dict_var_name_2_obj.items()}
                safe_exog = {(var_name_map[v], con_name_map[c]): coeff for (v, c), coeff in dict_var_con_2_lhs_exog.items()}
                safe_eq_map = {(var_name_map[v], con_name_map[c]): coeff for (v, c), coeff in dict_var_con_2_lhs_eq.items()}
                safe_LB = {con_name_map[k]: v for k, v in dict_con_name_2_LB.items()}
                safe_EQ = {con_name_map[k]: v for k, v in dict_con_name_2_eq.items()}

                self.var_dict = {name: model.addVar(lb=0, obj=obj, name=name) for name, obj in safe_var_obj.items()}
                model.update()

                group_exog = defaultdict(list)
                for (var, con), coeff in safe_exog.items():
                    group_exog[con].append((self.var_dict[var], coeff))

                group_eq = defaultdict(list)
                for (var, con), coeff in safe_eq_map.items():
                    group_eq[con].append((self.var_dict[var], coeff))

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
                model.update()

                model.ModelSense = gp.GRB.MINIMIZE
            ###    return model, var_dict, var_name_rev_map, con_name_rev_map

            ####def setup_alg(self):
                self.forbidden_var_names = set()
                self.tot_lp_time = 0
                self.all_removable_vars = [self.var_dict[name] for name in self.var_dict if self.var_name_rev_map[name] in self.all_possible_forbidden_names]
                self.vars_list = list(model.getVars())

                
            ####def call_core_alg(self):
                self.incumbent_lp_val = np.inf
                self.current_forbidden_vars = set()
                self.forbidden_var_names = set()

                vars_to_forbid = [self.var_dict[name] for name in self.var_dict if self.var_name_rev_map[name] in self.init_forbidden_names]
                self.add_to_forbidden(vars_to_forbid)

                
                if len(self.forbidden_var_names) == 0:
                    print("No forbidden variables found in model. This may be a typo.")
                    input('---')

                iter = 0
                while True:
                    t_before_itt=time.time()
                    iter += 1
                    
                    #print('starting iter')
                    #print(iter)
                    #num_vars = model.NumVars
                    #num_constraints = model.NumConstrs
                    #num_entries = model.NumNZs 
                    #print('[num_vars,num_constraints,num_entries]')
                    ##print([num_vars,num_constraints,num_entries])
                    #input('---')
                    model.update()
                    t_before_itt=time.time()-t_before_itt
                    t_start = time.time()
                    model.optimize()
                    t_end = time.time()
                    t_this_rount_opt=(t_end - t_start)
                    t_after_itt=time.time()
                    t_after_itt_1=time.time()
                    self.tot_lp_time += (t_end - t_start)
                    self.hist['time_iter'].append(t_end - t_start)

                    if model.Status != gp.GRB.OPTIMAL:
                        input('status failed')
                    t_after_itt_1=time.time()-t_after_itt_1
                    t_after_itt_1p5=time.time()
                    self.lp_obj_val = model.ObjVal

                    # Primal solution with original variable names
                    lp_primal_solution = {
                        self.var_name_rev_map[v.VarName]: v.X for v in self.vars_list
                    }
                    self.lp_primal_solution=lp_primal_solution
                    #get_constrs = model.getConstrs
                    #get_attr = model.getAttr
                    #pi_values = get_attr("Pi", get_constrs())
                    constrs = model.getConstrs()
                    pi_values = model.getAttr("Pi", constrs)
                    rev_map = self.con_name_rev_map
                    t_after_itt_1p5=time.time()-t_after_itt_1p5

                    t_after_itt_2=time.time()
                    
                   
                    #self.lp_dual_solution = {
                    #    rev_map[con.ConstrName]: pi for con, pi in zip(get_constrs(), pi_values)
                    #}
                    self.lp_dual_solution = dict(zip((rev_map[c.ConstrName] for c in constrs), pi_values))

                    t_after_itt_2=time.time()-t_after_itt_2
                    t_after_itt_3=time.time()
                    self.lp_objective=self.lp_obj_val
                    # Identify forbidden vars with nonzero primal values
                    self.active_removable_vars = [
                        v for v in self.all_removable_vars
                        if lp_primal_solution.get(self.var_name_rev_map[v.VarName], 0.0) > self.epsilon
                    ]

                    t_after_itt_3=time.time()-t_after_itt_3
                    t_after_itt_4=time.time()
                    self.inactive_removable_vars = [
                        v for v in self.current_forbidden_vars
                        if self.var_name_rev_map[v.VarName] in self.all_possible_forbidden_names and
                        abs(lp_primal_solution.get(self.var_name_rev_map[v.VarName], 0.0)) < self.epsilon
                    ]

                    t_after_itt_4=time.time()-t_after_itt_4
                    t_after_itt_5=time.time()
                    reduced_costs = model.getAttr("RC", self.vars_list)
                    self.reduced_costs_dict = {
                        self.var_name_rev_map[v.VarName]: rc for v, rc in zip(self.vars_list, reduced_costs)
                    }
                    t_after_itt_5=time.time()-t_after_itt_5
                    t_after_itt_6=time.time()
                    self.forbidden_vars_with_neg_red_cost = [
                        (v, self.reduced_costs_dict.get(self.var_name_rev_map[v.VarName], 0.0))
                        for v in self.all_removable_vars
                        if self.var_name_rev_map[v.VarName] in self.forbidden_var_names and
                        self.reduced_costs_dict.get(self.var_name_rev_map[v.VarName], 0.0) < -self.epsilon
                    ]
                    t_after_itt_6=time.time()-t_after_itt_6
                    t_after_itt_7=time.time()
                    self.pos_red_cost_removable = [
                        v for v in self.all_removable_vars
                        if self.reduced_costs_dict.get(self.var_name_rev_map[v.VarName], 0.0) > self.epsilon
                    ]
                    t_after_itt_7=time.time()-t_after_itt_7
                    t_after_itt_8=time.time()
                    self.non_pos_red_cost_removable = [
                        v for v in self.all_removable_vars
                        if self.reduced_costs_dict.get(self.var_name_rev_map[v.VarName], 0.0) < self.epsilon
                    ]

                    t_after_itt_8=time.time()-t_after_itt_8
                    t_after_itt_9=time.time()
                    
                    
                    
                    
                    #self.compute_bound()
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

                    t_after_itt_9=time.time()-t_after_itt_9
                    if self.verbose:
                        print(f"Iter {iter}: LP={self.lp_obj_val}, Incumbent={self.incumbent_lp_val}, "
                            f"Neg RC={len(self.forbidden_vars_with_neg_red_cost)}, Forbidden={len(self.forbidden_var_names)}")
                    t_after_itt=time.time()-t_after_itt
                    print('t_after_itt,t_before_itt,t_this_rount_opt')
                    print([t_after_itt,t_before_itt,t_this_rount_opt])
                    print('t_after_itt_1:  '+str(t_after_itt_1))
                    print('t_after_itt_1p5:  '+str(t_after_itt_1p5))
                    print('t_after_itt_2:  '+str(t_after_itt_2))
                    print('t_after_itt_3:  '+str(t_after_itt_3))
                    print('t_after_itt_4:  '+str(t_after_itt_4))
                    print('t_after_itt_5:  '+str(t_after_itt_5))
                    print('t_after_itt_6:  '+str(t_after_itt_6))
                    print('t_after_itt_7:  '+str(t_after_itt_7))
                    print('t_after_itt_8:  '+str(t_after_itt_8))
                    print('t_after_itt_9:  '+str(t_after_itt_9))
                    print('---')
                    #print('done iter')
                    #print(iter)
                    
                    #input('---')
        #input('Done CALL')