import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import time
import numpy as np
import re
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
        print('REMOVING IN len(self.pos_red_cost_removable)')
        print(len(self.pos_red_cost_removable))
        self.add_to_forbidden(self.pos_red_cost_removable)

    def remove_all_non_pos_after_improvement(self):
        #input('this may not work here')
        print('REMOVING IN len(self.inactive_removable_vars)')
        print(len(self.inactive_removable_vars))
        self.add_to_forbidden(self.inactive_removable_vars)

    def add_neg_red_cost_vars(self):
        #input('this may not work here')

        self.forbidden_vars_with_neg_red_cost.sort(key=lambda x: -x[1])
        selected = [v for v, _ in self.forbidden_vars_with_neg_red_cost[:self.max_terms_add_per_round]]
        self.remove_from_forbidden(selected)
    

    def add_all_vars(self):
        #input('this may not work here')

        #self.forbidden_vars_with_neg_red_cost.sort(key=lambda x: -x[1])
        #selected = [v for v, _ in self.forbidden_vars_with_neg_red_cost[:self.max_terms_add_per_round]]
        self.remove_from_forbidden(self.current_forbidden_vars.copy())




    def __init__(self, dict_var_name_2_obj,
                 dict_var_con_2_lhs_exog,
                 dict_con_name_2_LB,
                 dict_var_con_2_lhs_eq,
                 dict_con_name_2_eq,
                 all_possible_forbidden_names,
                 init_forbidden_names,
                 K=20, verbose=True, remove_choice=3, alg_use=1, debug_on=False,
                 min_improvement_dump=0.1, epsilon=1e-4):

        print('verbose')
        print(verbose)
        input('--')
        self.options = {
                "WLSACCESSID": "8f7bb9d6-8fe5-4349-9dd3-6abbaa9199a0",
                "WLSSECRET": "cb02810a-e0e2-4a1f-8fc0-fd375f65fc65",
                "LICENSEID": 2660300
        }
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

        self.dict_var_name_2_obj=dict_var_name_2_obj
        self.dict_var_con_2_lhs_exog=dict_var_con_2_lhs_exog
        self.dict_con_name_2_LB=dict_con_name_2_LB
        self.dict_var_con_2_lhs_eq=dict_var_con_2_lhs_eq
        self.dict_con_name_2_eq=dict_con_name_2_eq
        self.running_removal=False
        #self.call_current_solver()
        #self.call_solver_warm_start()
        self.call_solver_one_big_extra()
        #self.call_solver_warm_start_alternative()
        #self.call_solver_warm_epsilon()
        input('all done')
    def formulate_mapping(self,model):
        dict_var_name_2_obj=self.dict_var_name_2_obj
        dict_var_con_2_lhs_exog=self.dict_var_con_2_lhs_exog
        dict_con_name_2_LB=self.dict_con_name_2_LB
        dict_var_con_2_lhs_eq=self.dict_var_con_2_lhs_eq
        dict_con_name_2_eq=self.dict_con_name_2_eq
    
        original_vars = list(dict_var_name_2_obj.keys())
        original_cons = list(set(dict_con_name_2_LB.keys()) | set(dict_con_name_2_eq.keys()))
        var_name_map = {name: f"v{i}" for i, name in enumerate(original_vars)}
        con_name_map = {name: f"c{i}" for i, name in enumerate(original_cons)}
        self.var_name_rev_map = {v: k for k, v in var_name_map.items()}
        self.con_name_rev_map = {v: k for k, v in con_name_map.items()}

        safe_var_obj = {var_name_map[k]: v for k, v in dict_var_name_2_obj.items()}
        self.safe_exog = {(var_name_map[v], con_name_map[c]): coeff for (v, c), coeff in dict_var_con_2_lhs_exog.items()}
        self.safe_eq_map = {(var_name_map[v], con_name_map[c]): coeff for (v, c), coeff in dict_var_con_2_lhs_eq.items()}
        self.safe_LB = {con_name_map[k]: v for k, v in dict_con_name_2_LB.items()}
        self.safe_EQ = {con_name_map[k]: v for k, v in dict_con_name_2_eq.items()}

        self.var_dict = {name: model.addVar(lb=0, obj=obj, name=name) for name, obj in safe_var_obj.items()}
                

    def add_expressions(self,model):
        safe_exog=self.safe_exog
        safe_eq_map=self.safe_eq_map
        safe_LB=self.safe_LB
        safe_EQ=self.safe_EQ
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


    def call_solver_warm_epsilon(self):
        options=self.options
        with gp.Env(params=options) as env:
            with gp.Model("converted_LP", env=env) as model:
                model.setParam("OutputFlag", 1)
                self.formulate_mapping(model)
                model.update()
                self.add_expressions(model)
                self.init_key_info(model)
               

                model.update()
                epsilon=.001
                tiny_tiny=min([0.001,epsilon/100])
                iter=0
                for v in self.vars_list:
                    if v.ub<epsilon:
                        v.ub=epsilon
                while(True):
                    print('iteration')
                    num_inf_init=0 
                    for v in self.vars_list:
                        if v.ub>100:
                            num_inf_init=num_inf_init+1
                    t1=time.time()
                    model.optimize()
                    t1=time.time()-t1

                    self.tot_lp_time += t1

                    num_add=0
                    obj_1=model.ObjVal
                    cur_val=dict()
                        
                
                    for v in self.vars_list:
                        cur_val[v]=v.X
                        if v.ub<gp.GRB.INFINITY and tiny_tiny>abs(cur_val[v]-v.ub):
                            print('adding')
                            print([cur_val[v],v.ub,tiny_tiny])
                            v.ub = gp.GRB.INFINITY
                            num_add=num_add+1
                                                    
                    iter=iter+1
                    print('iter')
                    print(iter)
                    print('num_add,num_inf_init')
                    print([num_add,num_inf_init])
                    print('t1')
                    print([t1,])
                    print('[obj_1]')
                    print([obj_1,])
                    input('---')
                    if num_add==0:
                        self.grab_key_info_from_solution(model)

                        break

    #def basis_callback(model, where):
    #    if where == GRB.Callback.SIMPLEX:
    #        # Access iteration count
    #        iteration = model.cbGet(GRB.Callback.SPX_ITRCNT)
            
    #        # For basic variables at current iteration
            if iteration % 10 == 0:  # Every 10th iteration
                print(f"Iteration {iteration}")
                # You can retrieve basis status for each variable
                # Note: This requires Gurobi version that supports this specific callback
        input('--')
    def call_solver_warm_start(self):
        options=self.options
        with gp.Env(params=options) as env:
            with gp.Model("converted_LP", env=env) as model:
                model.setParam("OutputFlag", 1)
                model.setParam("Method", 0)
                self.formulate_mapping(model)
                model.update()
                self.add_expressions(model)
                self.init_key_info(model)
               
                model.update()
                print("FIRST")
                print("FIRST")
                print("FIRST")
                print("FIRST")
                print("FIRST")
                print("FIRST")

                t1=time.time()
                model.optimize()
                t1=time.time()-t1
                obj_1=model.ObjVal
                #model.optimize(basis_callback)
                v_basis = model.getAttr("VBasis", model.getVars())
                c_basis = model.getAttr("CBasis", model.getConstrs())
                # Optional: map statuses to readable strings
                basis_status_map = {
                    -1: "Nonbasic at lower bound",
                    0: "Basic",
                    1: "Nonbasic at upper bound",
                    2: "Superbasic (for QP problems only)"
                }

                # Example: print readable basis for variables
                for var, status in zip(model.getVars(), v_basis):
                    safe_name = var.VarName
                    original_name = self.var_name_rev_map.get(safe_name, safe_name)  # fallback to safe_name
                    print(f"{original_name}: {basis_status_map.get(status, 'Unknown')}")#self.add_all_vars()
                cur_val=dict()
                for v in self.vars_list:
                    cur_val[v]=v.X
                
                #model.reset()
                #model.setParam("Method", 1)  # dual simplex
                #model.setParam("Crossover", 0) 
                
                    #else:
                    #    v.Start=cur_val[v]
                for v in self.vars_list:
                    if v.ub<gp.GRB.INFINITY:
                        v.ub = gp.GRB.INFINITY#gp.GRB.INFINITY
                model.update()

                print("SECOND")
                print("SECOND")
                print("SECOND")
                print("SECOND")
                print("SECOND")
                print("SECOND")
                print("SECOND")
                
                t2=time.time()
                model.optimize()
                t2=time.time()-t2
                self.tot_lp_time =t2+t1
                obj_2=model.ObjVal
                self.grab_key_info_from_solution(model)
                print('t1,t2')
                print([t1,t2])
                print('[obj_1,obj_2]')
                print([obj_1,obj_2])
                input('---')
    def call_current_solver(self):
        options=self.options

       

        with gp.Env(params=options) as env:
            with gp.Model("converted_LP", env=env) as model:
                model.setParam("OutputFlag", 0)
                self.formulate_mapping(model)
                model.update()
                self.add_expressions(model)
                self.init_key_info(model)
                iter = 0
                #self.add_all_vars()
                print("HELLO")
                #model.setParam("Method", 0)
                model.setParam("Method", 2)  # Barrier
                model.setParam("Crossover", 1)
                while True:
                    iter += 1
                    model.update()
                    t_start = time.time()
                    model.optimize()
                    t_end = time.time()
                    t_this_rount_opt=(t_end - t_start)
                    
                   
                    self.tot_lp_time += t_this_rount_opt
                    self.hist['time_iter'].append(t_this_rount_opt)

                    if model.Status != gp.GRB.OPTIMAL:
                        input('status failed')
                
                    self.grab_key_info_from_solution(model)
                    #self.compute_bound()
                    self.hist['lp'].append(self.lp_obj_val)

                    if 1>0:

                        filtered_duals = { k: v for k, v in self.lp_dual_solution.items() if k.startswith("action_match_h")}
                        pattern = r'action_match_h=(\w+)_p=act_(\d+)_(\d+)'

                        tot_uv_map=defaultdict(float)
                        time_uv_dual_map=dict()
                        tot_uv_map_abs=defaultdict(float)
                        cap_uv_dual_map=dict()
                        ng_uv_dual_map=dict()
                        for key, value in filtered_duals.items():
                            match = re.search(pattern, key)
                            if match:
                                graph_type, u, v = match.groups()
                                u, v = int(u), int(v)
                                tot_uv_map[(u, v)] += value
                                tot_uv_map_abs[(u, v)] += abs(value)

                                if graph_type == 'timeGraph':
                                    time_uv_dual_map[(u, v)] = value
                                elif graph_type == 'capGraph':
                                    cap_uv_dual_map[(u, v)] = value
                                elif graph_type == 'ngGraph':
                                    ng_uv_dual_map[(u, v)] = value

                    
                    if self.lp_obj_val > self.incumbent_lp_val + 0.01:
                        input('Bound increased unexpectedly')
                    

                    self.apply_compression(model)

                    self.hist['numCurMid'].append(len(self.forbidden_var_names))
                    self.add_neg_red_cost_vars()
                    self.hist['numCurEnd'].append(len(self.forbidden_var_names))
                    if self.verbose:
                        print(f"Iter {iter}: LP={self.lp_obj_val},TimeLP={self.hist['time_iter'][-1]}, Incumbent={self.incumbent_lp_val}, "
                            f"Neg RC={len(self.forbidden_vars_with_neg_red_cost)}, Forbidden={len(self.forbidden_var_names)}")
                    
                    if len(self.forbidden_vars_with_neg_red_cost) < 0.5:
                        break
                  
        input('Done CALL')
    def grab_key_info_from_solution(self,model):

        self.lp_obj_val = model.ObjVal
        
        # Primal solution with original variable names
        lp_primal_solution = {
            self.var_name_rev_map[v.VarName]: v.X for v in self.vars_list
        }
        self.lp_primal_solution=lp_primal_solution
        constrs = model.getConstrs()
        pi_values = model.getAttr("Pi", constrs)
        if self.running_removal:
            pi_values = {
                constr.getAttr("ConstrName"): pi
                for constr, pi in zip(constrs, pi_values)
                if constr.getAttr("ConstrName") not in self.cons_2_remove
            }
            constrs = [my_constr for my_constr in constrs if my_constr.getAttr("ConstrName") not in self.cons_2_remove]
        rev_map = self.con_name_rev_map
        
        self.lp_dual_solution = dict(zip((rev_map[c.ConstrName] for c in constrs), pi_values))

        self.lp_objective=self.lp_obj_val
        # Identify forbidden vars with nonzero primal values
        self.active_removable_vars = [
            v for v in self.all_removable_vars
            if lp_primal_solution.get(self.var_name_rev_map[v.VarName], 0.0) > self.epsilon
        ]
        self.inactive_removable_vars = [
            v for v in self.current_forbidden_vars
            if self.var_name_rev_map[v.VarName] in self.all_possible_forbidden_names and
            abs(lp_primal_solution.get(self.var_name_rev_map[v.VarName], 0.0)) < self.epsilon
        ]
        reduced_costs = model.getAttr("RC", self.vars_list)
        self.reduced_costs_dict = {
            self.var_name_rev_map[v.VarName]: rc for v, rc in zip(self.vars_list, reduced_costs)
        }
        self.forbidden_vars_with_neg_red_cost = [
            (v, self.reduced_costs_dict.get(self.var_name_rev_map[v.VarName], 0.0))
            for v in self.all_removable_vars
            if self.var_name_rev_map[v.VarName] in self.forbidden_var_names and
            self.reduced_costs_dict.get(self.var_name_rev_map[v.VarName], 0.0) < -self.epsilon
        ]
        
        self.pos_red_cost_removable = [
            v for v in self.all_removable_vars
            if self.reduced_costs_dict.get(self.var_name_rev_map[v.VarName], 0.0) > self.epsilon
        ]
        self.non_pos_red_cost_removable = [
            v for v in self.all_removable_vars
            if self.reduced_costs_dict.get(self.var_name_rev_map[v.VarName], 0.0) < self.epsilon
        ]
        #for v in self.all_removable_vars:
        #    print('v.name')
        #    print(v.VarName)
        #    myRed=self.reduced_costs_dict.get(self.var_name_rev_map[v.VarName], 0.0)
        #    print('myRed')
        #    print(myRed)
        #    print('v in self.current_forbidden_vars')
         #   print(v in self.current_forbidden_vars)
        #input('--')
    def apply_compression(self,model):
        if self.lp_obj_val < self.incumbent_lp_val - self.min_improvement_dump:
            self.incumbent_lp_val = self.lp_obj_val
            if self.remove_choice == 2:
                self.remove_all_non_pos_after_improvement()
            elif self.remove_choice == 3:
                self.remove_all_pos_red_cost_after_improvement()
    def init_key_info(self,model):
    ###    return model, var_dict, var_name_rev_map, con_name_rev_map
        model.ModelSense = gp.GRB.MINIMIZE

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


#                    uv_2_red=dict()
#                    for uv in tot_uv_map_abs:
#                        u=uv[0]
#                        v=uv[1]
#                        my_name='act_'+str(u)+'_'+str(v)
#                        uv_2_red[uv]=self.reduced_costs_dict[my_name]
#                        if tot_uv_map_abs[uv]>.0001:
#                            print('---******--')
#
  #                          print('ng_uv_dual_map[uv]')
  #                          print(ng_uv_dual_map[uv])
  #                          print('cap_uv_dual_map[uv]')
  #                          print(cap_uv_dual_map[uv])
  #                          print('time_uv_dual_map[uv]')
  #                          print(time_uv_dual_map[uv])
  #                          print('tot_uv_map[uv]')
  #                          print(tot_uv_map[uv])
  #                          print('uv_2_red')
  #                          print(uv_2_red[uv])
  #                          input('--')
  #                          print('-----')
  #                  print('hihi')
    def call_solver_warm_start_alternative(self):
        options=self.options
        self.running_removal=True
        with gp.Env(params=options) as env:
            with gp.Model("converted_LP", env=env) as model:
                model.setParam("OutputFlag", 1)
                model.setParam("Method", 0)
                self.formulate_mapping(model)
                model.update()
                self.add_expressions(model)
                self.init_key_info(model)
                model.update()

                cons_2_remove=[]
                con_name_2_var=dict()
                my_count_2_rem=0
                #input('hello')
                for v in self.vars_list:
                    if v.ub<gp.GRB.INFINITY:
                        expr = gp.LinExpr()
                        expr.addTerms(1, v)
                        con_name="con_to_remove"+str(my_count_2_rem)
                        cons_2_remove.append(con_name)
                        model.addConstr(expr <= 0, name=con_name)
                        #print('con_name')
                        #print(con_name)
                        con_name_2_var[con_name]=v
                        v.ub = gp.GRB.INFINITY
                        my_count_2_rem=my_count_2_rem+1
                self.cons_2_remove=cons_2_remove
                model.update()
               
                print("FIRST")
                print("FIRST")
                print("FIRST")
                print("FIRST")
                print("FIRST")
                print("FIRST")

                t1=time.time()
                model.optimize()
                t1=time.time()-t1
                obj_1=model.ObjVal
                #self.add_all_vars()
                cur_val=dict()
                for v in self.vars_list:
                    cur_val[v]=v.X
                all_constrs = model.getConstrs()

                # Step 2: Create a mapping from constraint names to constraint objects
                name_to_constr = {c.ConstrName: c for c in all_constrs}
               
                model.update()

                for name in cons_2_remove:
                    if name in name_to_constr:
                        
                        #model.remove(name_to_constr[name])
                        
                        constr=name_to_constr[name]
                        #constr.RHS = 1e20
                        #if 10:
                        #    constr.RHS = 1e20
                        #else:
                        var=con_name_2_var[name]
                        model.chgCoeff(constr, var, -1)
                # Step 4: Update the model structure
                model.update()
                #model.reset()
                #model.setParam("Method", 1)  # dual simplex
                #model.setParam("Crossover", 0) 
                    #else:
                    #    v.Start=cur_val[v]
                #for v in self.vars_list:
               #     if v.ub<gp.GRB.INFINITY:
               #         v.ub = gp.GRB.INFINITY#gp.GRB.INFINITY
               # model.update()

                print("SECOND")
                print("SECOND")
                print("SECOND")
                print("SECOND")
                print("SECOND")
                print("SECOND")
                print("SECOND")
                
                t2=time.time()
                model.optimize()
                t2=time.time()-t2
                self.tot_lp_time =t2+t1
                obj_2=model.ObjVal
                self.grab_key_info_from_solution(model)
                print('t1,t2')
                print([t1,t2])
                print('[obj_1,obj_2]')
                print([obj_1,obj_2])
                input('---')
    
    def call_solver_one_big_extra(self):
        options=self.options
        with gp.Env(params=options) as env:
            with gp.Model("converted_LP", env=env) as model:
                model.setParam("OutputFlag", 1)
                #model.setParam("Method", )
                self.formulate_mapping(model)
                model.update()
                self.add_expressions(model)
                self.init_key_info(model)
               
                model.update()
                print("FIRST")
                print("FIRST")
                print("FIRST")
                print("FIRST")
                print("FIRST")
                print("FIRST")

                t1=time.time()
                model.optimize()
                t1=time.time()-t1
                obj_1=model.ObjVal
                model.optimize()
# Solve the model

                # Step 1: Get the objective value of the current solution
                # Step 1: Solve the model

                # Step 2: Extract the solution's objective value
                # Step 1: Solve the model
                model.optimize()

                # Step 2: Extract the solution's objective value
                solution_obj = model.ObjVal
                obj_1=solution_obj
                # Step 3: Get current variable values
                vars = model.getVars()
                x_vals = model.getAttr("X", vars)

                # Step 4: Build a fast lookup dictionary: variable index → value
                x_val_dict = {var.index: val for var, val in zip(vars, x_vals)}

                # Step 5: Compute LHS values for each constraint
                constrs = model.getConstrs()
                lhs_vals = []

                for constr in constrs:
                    row = model.getRow(constr)         # LinExpr
                    row_lhs = 0.0
                    for i in range(row.size()):
                        var = row.getVar(i)
                        coeff = row.getCoeff(i)
                        row_lhs += x_val_dict[var.index] * coeff
                    lhs_vals.append(row_lhs)

                if 1>0:
                    # Step 6: Create a new variable representing this solution
                    new_var = model.addVar(obj=solution_obj+.01, name="column_from_solution")

                    # Step 7: Set this variable’s coefficients in each constraint
                    for constr, lhs_val in zip(constrs, lhs_vals):
                        model.chgCoeff(constr, new_var, lhs_val)

                # Step 8: Finalize model update
                for v in self.vars_list:
                    if v.ub<gp.GRB.INFINITY:
                        v.ub = gp.GRB.INFINITY#gp.GRB.INFINITY
                model.update()

                # Finalize model update
                model.update()
                model.reset()
                model.update()
                t2=time.time()
                model.optimize()
                t2=time.time()-t2
                obj_2=model.ObjVal
                self.tot_lp_time =t2+t1
                self.grab_key_info_from_solution(model)
                print('t1,t2')
                print([t1,t2])
                print('[obj_1,obj_2]')
                print([obj_1,obj_2])
                input('---')