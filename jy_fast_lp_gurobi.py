import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import time
import numpy as np
import re
import random
import heapq
import networkx as nx
class jy_fast_lp_gurobi:


    def add_to_forbidden(self, vars_to_remove):
        #input('this may not work here')
       # print('len(vars_to_remove)')
        #print(len(vars_to_remove))
        #input('hihi')
        set_remove=set(vars_to_remove)
        vars_to_remove = list(set_remove - self.current_forbidden_vars)
        print('adding to forbidden '+str(len(vars_to_remove)))
        #input('---')
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
        print('len(self.inactive_removable_vars)')
        print(len(self.inactive_removable_vars))
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
                 init_forbidden_names,my_lower_bound_object,
                 K=100, verbose=True, remove_choice=3, alg_use=1, debug_on=False,
                 min_improvement_dump=0.1, epsilon=1e-4,pos_red_cut=0.01,min_dual_slack_add_poss=1,):
        self.pos_red_cut=pos_red_cut
        self.min_dual_slack_add_poss=min_dual_slack_add_poss
        
        self.options = {
            "WLSACCESSID": "b7836a23-3df1-40ac-be4d-310282e2178e",
            "WLSSECRET": "8dd2c11c-cb9b-46f3-b072-4887712ea0c9",
            "LICENSEID": 2690165
        }
        self.my_lower_bound_object=my_lower_bound_object
        self.verbose = verbose
        self.remove_choice = remove_choice
        self.alg_use = alg_use
        self.min_improvement_dump = min_improvement_dump
        self.epsilon = epsilon
        self.debug_on = debug_on
        self.hist = {'sum_red_cost':[],'lp': [],'numCurStart': [], 'numCurMid': [], 'numCurEnd': [], 'time_iter': []}
        self.max_terms_add_per_round = K
        self.all_possible_forbidden_names = all_possible_forbidden_names
        self.min_forbidden_apply=len(self.my_lower_bound_object.action_2_cost)*0.095#len(self.my_lower_bound_object.action_2_cost)-9*np.sqrt(len(self.my_lower_bound_object.action_2_cost))
        print('min_forbidden_apply')
        print(self.min_forbidden_apply)
        #init_forbidden_names=[]
        self.BIG_M=20000
        self.init_forbidden_names = init_forbidden_names
        num_adds_done=0
        max_value=0
        init_forbidden_names=set(init_forbidden_names)
        for act in self.all_possible_forbidden_names:#.my_lower_bound_object.action_2_cost:
            max_value=max([max_value,self.my_lower_bound_object.action_2_cost[act]])
            if self.my_lower_bound_object.action_2_cost[act]>self.BIG_M:#and act in all_possible_forbidden_names:
                init_forbidden_names.add(act)
                num_adds_done=num_adds_done+1
                #print('num_adds_done')
                #print(num_adds_done)
                #input('---')

        #print('self.BIG_M)')
        #print(self.BIG_M)
        #print('max_value')
        #print(max_value)
        print('num_adds_done')
        print(num_adds_done)
        print('---')
        #input('---')
        init_forbidden_names=list(init_forbidden_names)
        self.init_forbidden_names=init_forbidden_names
        self.dict_var_name_2_obj=dict_var_name_2_obj
        self.dict_var_con_2_lhs_exog=dict_var_con_2_lhs_exog
        self.dict_con_name_2_LB=dict_con_name_2_LB
        self.dict_var_con_2_lhs_eq=dict_var_con_2_lhs_eq
        self.dict_con_name_2_eq=dict_con_name_2_eq
        self.running_removal=False
        self.K=K
        
        self.prepare_for_additions_novel()
        self.prepare_for_clean_novel()
        baseline_check_debug=False
        baseline_valu=np.inf
        baseline_time_lp=np.inf
        if baseline_check_debug==True:
            self.init_forbidden_names = []
            print('running Baseline')
            #self.call_solver_warm_no_path()
            self.call_solver_baseline()#_clean_each
            baseline_valu=self.hist['lp'][-1]
            baseline_time_lp=self.hist['time_iter'][0]
            self.hist = {'sum_red_cost':[],'lp': [],'numCurStart': [], 'numCurMid': [], 'numCurEnd': [], 'time_iter': []}
            
            
            #input('done running baseline')
            self.init_forbidden_names=init_forbidden_names
        self.call_solver_warm_no_remove()
        #self.call_solver_warm_just_use_bounds()
        if baseline_check_debug==True and abs(baseline_valu-self.hist['lp'][-1])>0.001:
            print('disagreement')
            print((baseline_valu-self.hist['lp'][-1]))
            input('ERROR')
        #
        if baseline_check_debug==True:
            print('self.hist')
            print(self.hist)
            print('self.hist[lp]')
            print('time iter base')
            print(self.hist['time_iter'])
            print('sum times')
            print(sum(self.hist['time_iter']))
            print('baseline_time_lp')
            print(baseline_time_lp)
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
        self.var_name_map=var_name_map
        self.con_name_map=con_name_map
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

    def prepare_for_clean_novel(self):

        
        self.mapping_remove_var=dict()
        self.mapping_remove_con=dict()
        self.mapping_equiv=dict()
        #add in the original varaible name and all constraints corresponding to
        
        for act in self.all_possible_forbidden_names:
            self.mapping_remove_con[act]=[]
            self.mapping_remove_var[act]=[act]
            _, u, v = act.split("_")
            name_uv_cap='cap_uv_'+str(u)+'_'+str(v)
            name_uv_time='time_uv_'+str(u)+'_'+str(v)
            self.mapping_remove_con[act].append(name_uv_cap)
            self.mapping_remove_con[act].append(name_uv_time)
            for h in self.nov_h_uv_2_fg:
                for fg in self.nov_h_uv_2_fg[h][act]:
                    f=fg[0]
                    g=fg[1]
                    var_name='EDGE_h='+h+'_f='+f+'_g='+g
                    
                    self.mapping_remove_var[act].append(var_name)
                q=tuple([act])
                con_name_match='action_match_h='+h+"_p="+str(act)
                con_name_equiv='equiv_class='+h+"_q="+str(q)
                var_equiv_remove="fill_PQ_h="+str(h)+"_q="+str(q)+"_p="+str(act)
                self.mapping_remove_var[act].append(var_equiv_remove)
                self.mapping_remove_con[act].append(con_name_match)
                self.mapping_remove_con[act].append(con_name_equiv)
                
            #self.mapping_remove_con[act]=[]
    def clean_model(self,M):
        #input('do i want to be here')
        #remove variables forced to zero.  
        vars_names_to_remove=[]
        cons_names_to_remove=[]
        TMP_var_dict = {var.VarName: var for var in M.getVars()}
        TMP_con_dict = {con.ConstrName: con for con in M.getConstrs()}

        for var in M.getVars():
            if var.ub<self.epsilon:
                var.ub=np.inf
                
        for var in self.current_forbidden_vars:
            var_name_orig=self.var_name_rev_map[var.VarName]
            
            for v in self.mapping_remove_var[var_name_orig]:
                vars_names_to_remove.append(v)
            for con in self.mapping_remove_con[var_name_orig]:
                cons_names_to_remove.append(con)
        
        vars_to_remove=[]
        for var_name_orig in vars_names_to_remove:
            var_name_compress=self.var_name_map[var_name_orig]
            var=TMP_var_dict[var_name_compress]
            vars_to_remove.append(var)
        
        cons_to_remove=[]
        for var_name_orig in cons_names_to_remove:
            con_name_compress=self.con_name_map[var_name_orig]
            con=TMP_con_dict[con_name_compress]
            cons_to_remove.append(con)
        
        
        for var in vars_to_remove:
            M.remove(var)
        
        for con in cons_to_remove:
            M.remove(con)
        return M
    
    def call_solver_baseline(self):
        options=self.options
        with gp.Env(params=options) as env:
            with gp.Model("converted_LP", env=env) as model:
                model.setParam("OutputFlag", 1)
                self.formulate_mapping(model)
                model.update()
                self.add_expressions(model)
                self.init_key_info(model)
               
                t1=time.time()
                model.optimize()
                t1=time.time()-t1
                self.tot_lp_time += t1
                self.hist['time_iter'].append(t1)
                self.hist['lp'].append(model.ObjVal)
                self.hist['numCurStart'].append(self.countCurP())
                

    def call_solver_warm_just_use_bounds(self):
        options=self.options
        self.time_accel=[]
        with gp.Env(params=options) as env:
            with gp.Model("converted_LP", env=env) as model:
                self.formulate_mapping(model)
                model.update()
                self.add_expressions(model)
                self.init_key_info(model)
               

                model.update()
                incubent_bound=np.inf
                self.tot_lp_time=0
                counter=0
                while(True):
                    vars_current_zero=[]
                    for var in model.getVars():
                        if var.ub<self.epsilon:
                            vars_current_zero.append(var)
                    vars_current_zero=set(vars_current_zero)  
                    vars_names_need_zero=[]
                    for var in self.current_forbidden_vars:
                        var_name_orig=self.var_name_rev_map[var.VarName]
                        
                        for v in self.mapping_remove_var[var_name_orig]:
                            vars_names_need_zero.append(v)

                    vars_names_need_zero=set(vars_names_need_zero)
                    vars_need_zero=[]
                    for v_name in vars_names_need_zero:
                        var=self.var_dict[self.var_name_map[v_name]]
                        vars_need_zero.append(var)
                    vars_need_zero=set(vars_need_zero)
                    vars_to_set_inf=vars_current_zero-vars_need_zero
                    vars_to_set_zero=vars_need_zero-vars_current_zero
                    print('vars_to_set_inf')
                    print(len(vars_to_set_inf))
                    print('vars_to_set_zero')
                    print(len(vars_to_set_zero))
                    if len(vars_to_set_inf)<1 and counter>0:
                        break
                    model.update()

                    for var in vars_to_set_inf:
                        #print('var_name')
                        #print(var_name)
                        #print('self.var_name_map[var_name]')
                        #print(self.var_name_map[var_name])
                        #var=self.var_dict[self.var_name_map[var_name]]
                        var.ub=np.inf
                    model.update()

                    for var in vars_to_set_zero:
                        #var=self.var_dict[self.var_name_map[var_name]]

                        var.ub=0
                    model.update()
                    model.reset()
                    t1=time.time()
                    model.optimize()
                    t1=time.time()-t1
                    self.hist['time_iter'].append(t1)
                    self.hist['lp'].append(model.ObjVal)
                    
                    self.hist['numCurStart'].append(self.countCurP())
                    cur_bound=model.ObjVal
                    self.lp_obj_val=cur_bound
                    self.grab_key_info_from_solution(model)

                    if cur_bound>incubent_bound+0.001:
                        print('cur_bound')
                        print(cur_bound)
                        print('incubent_bound')
                        print(incubent_bound)
                        input('error here')
                    if cur_bound<incubent_bound-0.1:
                        self.apply_compression()
                        incubent_bound=cur_bound
                    self.hist['numCurMid'].append(self.countCurP())

                    self.remove_from_forbidden(self.var_add_novel)
                    self.hist['numCurEnd'].append(self.countCurP())

                    #print('cur_bound')
                    #print(cur_bound)
                    #print('vars_to_set_inf')
                    #print(len(vars_to_set_inf))
                    #print('vars_to_set_zero')
                    #print(len(vars_to_set_zero))
                    #input('done inter')
                    counter=counter+1
                self.apply_compression()
                #input('done call')

    def call_solver_warm_no_remove(self):
        options=self.options
        self.time_accel=[]
        with gp.Env(params=options) as env:
            with gp.Model("converted_LP", env=env) as model:
                model.setParam("OutputFlag", 1)
                self.formulate_mapping(model)
                model.update()
                self.add_expressions(model)
                self.init_key_info(model)
               

                model.update()
                incubent_bound=np.inf
                self.tot_lp_time=0
                counter=0
                while(True):
                    M=model.copy()
                    M=self.clean_model(M)
                    M.update()
                    M.reset()
                    t1=time.time()
                    M.optimize()
                    t1=time.time()-t1
                    self.tot_lp_time += t1
                    
                    self.hist['time_iter'].append(t1)
                    self.hist['lp'].append(M.ObjVal)
                    
                    self.hist['numCurStart'].append(self.countCurP())
                    cur_bound=M.ObjVal
                    print('cur_bound')
                    print(cur_bound)
                    print('cur_bound')
                    #input('cur_bound')
                    self.grab_key_info_from_solution(M)
                    self.hist['sum_red_cost'].append(self.sum_negative_rc)
                    use_PGM=False
                    if use_PGM==True:
                        self.compute_pricing_PGM()
                    if cur_bound>incubent_bound+0.001:
                        print('cur_bound')
                        print(cur_bound)
                        print('incubent_bound')
                        print(incubent_bound)
                        input('error here')
                    #if cur_bound<incubent_bound-0.1 and len(self.forbidden_var_names)<self.min_forbidden_apply:
                    #    self.apply_compression()
                    #    incubent_bound=cur_bound
                        #continue
                    self.hist['numCurMid'].append(self.countCurP())
                    print('counter')
                    print(counter)
                    print('self.sum_negative_rc')
                    print(self.sum_negative_rc)
                    if use_PGM==True:
                        print('self.sum_negative_rc_PGM')
                        print(self.sum_negative_rc_PGM)
                        print('len(self.var_add_novel_pgm)')
                        print(len(self.var_add_novel_pgm))
                    print('len(self.var_add_novel)')
                    print(len(self.var_add_novel))
                    print('len(self.dual_slack_amounts_rc_no_pgm)')
                    print( sum(1 for v in self.dual_slack_amounts_rc_no_pgm.values() if v < -0.000001)
)
                    if len(self.var_add_novel)==0 and self.sum_negative_rc<-0.01:
                        break
                        input('nothing added with neg')
                    if use_PGM==True:
                        if self.sum_negative_rc_PGM>-0.0001:
                            break
                    else:
                        if self.sum_negative_rc>-0.01:
                            break
                    if use_PGM==True:
                        self.remove_from_forbidden(self.var_add_novel_pgm)
                    else:
                        self.remove_from_forbidden(self.var_add_novel)
                    self.hist['numCurEnd'].append(self.countCurP())
                    counter=counter+1
                    model.update()
                    if use_PGM==True:
                        if len(self.var_add_novel_pgm)>0 and len(self.var_add_novel)==0:
                            print('self.var_add_novel_pgm')
                            print(self.var_add_novel_pgm)
                            print('self.var_add_novel')
                            print(self.var_add_novel)
                            input('bgi error here')
                    #input('----')
                if len(self.forbidden_var_names)<self.min_forbidden_apply:

                    self.apply_compression()


   
    def countCurP(self):
        count=0
        for var in self.all_removable_vars:
            if var.ub <gp.GRB.INFINITY:
                count=count+1
        return count
   
    def grab_key_info_from_solution(self,M):

        self.lp_obj_val = M.ObjVal
        
        # Primal solution with original variable names
        lp_primal_solution = defaultdict(
            lambda: 0.0,  # default value for missing keys
            {self.var_name_rev_map[v.VarName]: v.X for v in M.getVars()}
        )
        self.lp_primal_solution=lp_primal_solution
        constrs = M.getConstrs()
        pi_values = M.getAttr("Pi", constrs)
        rev_map = self.con_name_rev_map
        self.lp_dual_solution=defaultdict(float)
        #self.lp_dual_solution = dict(zip((rev_map[c.ConstrName] for c in constrs), pi_values))
        #self.lp_dual_solution = dict(
        #    zip(
        #        (rev_map[c.ConstrName] for c in constrs if c.ConstrName in rev_map),
        #        (pi for c, pi in zip(constrs, pi_values) if c.ConstrName in rev_map)
        #    )
        #)

        #rev_map = self.con_name_rev_map
        
        #self.lp_dual_solution = dict(zip((rev_map[c.ConstrName] for c in constrs), pi_values))
        pi_values = M.getAttr("Pi", constrs)

# Map from constraint object to dual value
        constr_to_pi = dict(zip(constrs, pi_values))

        self.lp_dual_solution = defaultdict(float)
        for con in M.getConstrs():
            con_name_simp=con.ConstrName
            con_name=self.con_name_rev_map[con_name_simp]
            self.lp_dual_solution[con_name] = constr_to_pi[con]
        #for con_name, con in self.con_name_map.items():
        #    dual_val = constr_to_pi[con]
         #   self.lp_dual_solution[con_name] = dual_val
        #for 
        
        #self.lp_dual_solution = defaultdict(
        #    lambda: 0,
        #    dict(
        #        zip(
        #            (rev_map[c.ConstrName] for c in constrs if c.ConstrName in rev_map),
        #            (pi for c, pi in zip(constrs, pi_values) if c.ConstrName in rev_map)
        #        )
        #    )
        #)

        self.lp_objective=self.lp_obj_val
        # Identify forbidden vars with nonzero primal values
        self.active_removable_vars = [
            v for v in self.all_removable_vars
            if lp_primal_solution.get(self.var_name_rev_map[v.VarName], 0.0) > self.epsilon
        ]
        self.inactive_removable_vars = [
            v for v in self.all_removable_vars
            if lp_primal_solution.get(self.var_name_rev_map[v.VarName], 0.0) < self.epsilon
        ]
        reduced_costs = M.getAttr("RC", M.getVars())
        self.reduced_costs_dict = {
            self.var_name_rev_map[v.VarName]: rc for v, rc in zip(M.getVars(), reduced_costs)
        }
        
        
        self.pos_red_cost_removable = [
            v for v in self.all_removable_vars
            if self.reduced_costs_dict.get(self.var_name_rev_map[v.VarName], 0.0) > self.pos_red_cut
        ]
        #print('set(self.pos_red_cost_removable)')
        #print(set(self.pos_red_cost_removable))
        #print('set(self.pos_red_cost_removable)')
        
        ### NEW MATERIAL
        L=self.my_lower_bound_object
        dual_slack_amounts=dict()
        dual_con_contrib=dict()
        dual_big_edge_contrib=dict()
        for act_uv in L.action_2_cost:
            dual_slack_amounts[act_uv]=0
            dual_con_contrib[act_uv]=0
            dual_big_edge_contrib[act_uv]=0
 
        for v_con in L.action_con_2_contrib:
            var_name=v_con[0]
            con_name=v_con[1]
            weight_use=L.action_con_2_contrib[v_con]
            dual_val=self.lp_dual_solution[con_name]
            sign_use=-1
            tmp=sign_use*weight_use*dual_val
            dual_con_contrib[var_name]+=tmp
        self.dual_con_contrib=dual_con_contrib
        dual_big_edge_contrib_by_h=dict()
        dual_slack_amounts_components=dict()
        dual_con_from_lp=dict()

        for my_act in L.action_2_cost:
            dual_con_from_lp[my_act]=dict()
            dual_big_edge_contrib_by_h[my_act]=dict()
            trm1=self.dict_var_name_2_obj[my_act]
            trm2=dual_con_contrib[my_act]
            trm_3=0
            if my_act not in self.forbidden_var_names and my_act !=L.null_action:
                for h in self.nov_h_uv_2_fg:
                    con_name='action_match_h='+h+"_p="+my_act
                    dual_val=self.lp_dual_solution[con_name]
                    dual_con_from_lp[my_act][h]=dual_val
            for h in self.nov_h_uv_2_fg:
                trm3_by_part=np.inf
                
                for fg in self.nov_h_uv_2_fg[h][my_act]:
                    f=fg[0]
                    g=fg[1]
                    dual_f=0
                    dual_g=0
                    if f!=self.nov_source_node[h]:
                        flow_con_name_f='flow_in_out_h='+h+"_n="+f#self.nov_node_2_flow_con_name[node_f]
                        dual_f=self.lp_dual_solution[flow_con_name_f]
                    if g!=self.nov_sink_node[h]:
                        flow_con_name_g='flow_in_out_h='+h+"_n="+g#self.nov_node_2_flow_con_name[node_g]
                        dual_g=self.lp_dual_solution[flow_con_name_g]
                    
                    

                    dual_fg_gap=-dual_f+dual_g
                    
                    trm3_by_part=min([trm3_by_part,dual_fg_gap])
                    
                dual_big_edge_contrib_by_h[my_act][h]=trm3_by_part
            trm3=sum(dual_big_edge_contrib_by_h[my_act].values())
            dual_big_edge_contrib[my_act]=trm3
            
            dual_slack_amounts_components[my_act]=[trm1,trm2,trm3]
            dual_slack_amounts[my_act]=trm1+trm2+trm3
            
        self.dual_big_edge_contrib_by_h=dual_big_edge_contrib_by_h
        self.sum_negative_rc = 0
        self.dual_con_from_lp=dual_con_from_lp
        self.dual_slack_amounts=dual_slack_amounts.copy()
        for v in dual_slack_amounts:
            if dual_slack_amounts[v]<-0.000001:
                self.sum_negative_rc=self.sum_negative_rc+dual_slack_amounts[v]
            #    print('dual_slack_amounts[v]')
            #    print(dual_slack_amounts[v])
            #    print('self.dual_big_edge_contrib_by_h[v]')
            #    print(self.dual_big_edge_contrib_by_h[v])
            #    input('---')
            if v not in self.forbidden_var_names:
                dual_slack_amounts[v]=np.inf
                #print('my_act')
                #print(my_act)
                #input('got one')
        self.dual_slack_amounts_rc_no_pgm=dual_slack_amounts
        selected = heapq.nsmallest(self.K, (v_uv for v_uv in dual_slack_amounts.items() if v_uv[1] < self.min_dual_slack_add_poss), key=lambda t: t[1])
        
        if 1>0:
            lowest_val_by_u=dict()
            lowest_val_by_v=dict()
            lowest_act_by_u=dict()
            lowest_act_by_v=dict()

            for my_act in dual_slack_amounts:
                this_red_cost=dual_slack_amounts[my_act]
                if this_red_cost<-0.01:#self.min_dual_slack_add_poss:
                    _, u, v = my_act.split("_")
                    if u not in lowest_val_by_u or lowest_val_by_u[u]>this_red_cost:
                        lowest_val_by_u[u]=this_red_cost
                        lowest_act_by_u[u]=my_act
                    if v not in lowest_val_by_v or lowest_val_by_v[v]>this_red_cost:
                        lowest_val_by_v[v]=this_red_cost
                        lowest_act_by_v[v]=my_act
            all_terms_add=[]
            s1=set(lowest_act_by_u.values())
            s2=set(lowest_act_by_v.values())
            s3 = [
                v for v, _ in heapq.nsmallest(
                    self.K,
                    ((v, slack) for v, slack in dual_slack_amounts.items() if slack < -0.001),
                    key=lambda t: t[1]
                )
            ]           
            #if len(s1)<len(s2):#len(set(lowest_act_by_u.values()))<len(set(lowest_act_by_v.values())):
            #    selected=s1#s
            #else:
            #    selected=s2
            selected=s1| s2 #|set(s3)
        self.var_add_novel=[]
        #print('dual_slack_amounts')
        #print(dual_slack_amounts)
        #print('selected')
        #print(selected)
        #print('selected')
        #print('self.sum_negative_rc')
        #print(self.sum_negative_rc)
        #print('---')
        for act_u_v_term in selected:
            act_u_v=act_u_v_term#[0]
            var_name_compress=self.var_name_map[act_u_v]
            var=self.var_dict[var_name_compress]
            self.var_add_novel.append(var)
    
    def prepare_for_additions_novel(self):
        L=self.my_lower_bound_object
        #nov_source_node
        self.nov_source_node=dict()
        self.nov_sink_node=dict()
        for h in L.all_graph_names:
            source_i=L.h_2_source_id[h]
            sink_i=L.h_2_sink_id[h]
            source_f=L.graph_node_2_agg_node[h][source_i]
            sink_f=L.graph_node_2_agg_node[h][sink_i]
            self.nov_source_node[h]=source_f
            self.nov_sink_node[h]=sink_f




        #nov_h_uv_2_fg
        self.nov_h_uv_2_fg=dict()
        for h in L.graph_names:
            self.nov_h_uv_2_fg[h]=dict()
            self.nov_h_uv_2_fg[h][L.null_action]=[]
            for my_act in L.all_actions:
                self.nov_h_uv_2_fg[h][my_act]=[]
        
        for h in L.graph_names:
            for fg in L.h_fg_2_q[h]:
                if len(L.h_fg_2_q[h][fg])!=1:
                    print('fg')
                    print(fg)
                    print('L.h_fg_2_q[h][fg]')
                    print(L.h_fg_2_q[h][fg])
                    input('this is not incorrect but I am assuming for now that p is single element')
                
                for my_act in L.h_fg_2_q[h][fg]:
                    
                    #print('h')
                    #print(h)
                    #print('my_act')
                    #print(my_act)
                    #print('L.null_action')
                    #print(L.null_action)
                    self.nov_h_uv_2_fg[h][my_act].append(fg)                    

     


    def apply_compression(self,):
        #if self.lp_obj_val < self.incumbent_lp_val - self.min_improvement_dump:
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
        self.all_removable_vars=[]
        for my_act_name in self.all_possible_forbidden_names:
            self.all_removable_vars.append(self.var_dict[self.var_name_map[my_act_name]])
        #self.all_removable_vars = [self.var_dict[name] for name in self.var_dict if self.var_name_rev_map[name] in self.all_possible_forbidden_names]
        self.vars_list = list(model.getVars())


        ####def call_core_alg(self):
        self.incumbent_lp_val = np.inf
        self.current_forbidden_vars = set()
        self.forbidden_var_names = set()


        vars_to_forbid=[]
        for my_act_name in self.init_forbidden_names:
            vars_to_forbid.append(self.var_dict[self.var_name_map[my_act_name]])
        
        #vars_to_forbid = [self.var_dict[name] for name in self.var_dict if self.var_name_rev_map[name] in self.init_forbidden_names]
        self.add_to_forbidden(vars_to_forbid)


       # if len(self.forbidden_var_names) == 0:
       #     print("No forbidden variables found in model. This may be a typo.")
       #     input('---')

    def compute_pricing_PGM(self):
        self.all_actions=self.my_lower_bound_object.action_2_cost.keys()
        L=self.my_lower_bound_object
        
        self.edge_weight_pricing_graph=dict()
        H=self.nov_h_uv_2_fg.keys()

        phi_by_h_p=dict()
        E_by_h=dict()
        for h in H:
            phi_by_h_p[h]=dict()
            E_by_h[h]=dict()
        for my_act in self.all_actions:
            if my_act=='null_action':
                continue
            trm1=self.dict_var_name_2_obj[my_act]
            trm2=self.dual_con_contrib[my_act]
            trm3=sum(self.dual_big_edge_contrib_by_h[my_act].values())
           # print('[trm1,trm2,trm3,my_act in self.forbidden_var_names]')
           # print([trm1,trm2,trm3,my_act in self.forbidden_var_names])
            do_add_epsilon=False
            if abs(trm3)<self.epsilon:
                do_add_epsilon=True
                if trm3>=0:
                    trm3=(len(H)*self.epsilon)
                else:
                    trm3=-(len(H)*self.epsilon)
            #print('my_act')
            #print(my_act)
            trm4=sum(self.dual_big_edge_contrib_by_h[my_act].values())
            slack=trm1+trm2+trm4
            for h in H:
                #print('in H')
                #print(h)
                #this_weight=self.dual_big_edge_contrib_by_h[my_act][h]#+self.epsilon
                #if do_add_epsilon:
                #    this_weight=trm3/len(H)
                
                dual_val=self.dual_big_edge_contrib_by_h[my_act][h]
                phi_by_h_p[h][my_act]=dual_val#-(slack/len(H))
                if h=='timeGraph':
                    phi_by_h_p[h][my_act]=phi_by_h_p[h][my_act]-slack
                    #phi_by_h_p[h][my_act]=(-trm1-trm2)*this_weight/trm3
                    #phi_by_h_p[h][my_act]=-slack/len(H)

                #print('phi_by_h_p[h][my_act]')
                #print(phi_by_h_p[h][my_act])
                #print('h')
                #print(h)
                #print('my_act')
                #print(my_act)
                #print('this_weight')
                #print(this_weight)
                #print('trm3')
                #print(trm3)
                #print('do_add_epsilon')
                #print(do_add_epsilon)
                if my_act not in self.forbidden_var_names and my_act!='null_action':
                    
                    if (phi_by_h_p[h][my_act]-dual_val)>0.01:
                        print('-------')
                        print('-------')
                        print('-------')
                        print('-------')
                        print('my_act')
                        print(my_act)
                        print('dual_val')
                        print(dual_val)
                        print('phi_by_h_p[h][my_act]')
                        print(phi_by_h_p[h][my_act])
                        print('self.dual_big_edge_contrib_by_h[my_act]')
                        print(self.dual_big_edge_contrib_by_h[my_act])
                        print('this_weight')
                        print(this_weight)
                        print('trm3')
                        print(trm3)
                        print('[trm1,trm2]')
                        print([trm1,trm2])
                        print()

                        input('wrong lilely could be numerical ')
                #print('here')
                #print('self.dual_big_edge_contrib_by_h[my_act]')
                #print(self.dual_big_edge_contrib_by_h[my_act])
                #print('self.dual_slack_amounts[my_act]')
                #print(self.dual_slack_amounts[my_act])
                #print('phi_by_h_p[h][my_act]')
                #print(phi_by_h_p[h][my_act])
                #input('hihii')

                #if phi_by_h_p[h][my_act]>0.1:
                #    print('phi_by_h_p[h][my_act]')
                #    print(phi_by_h_p[h][my_act])
                #    
                #    input('found one ')
            #print('self.dual_big_edge_contrib_by_h[my_act]')
            #print(self.dual_big_edge_contrib_by_h[my_act])
            #print('trm3')
            #print(trm3)
            #print('[trm1,trm2]')
            #print([trm1,trm2])
            #print('[phi_by_h_p][TIME][my_act]')
            #print(phi_by_h_p['timeGraph'][my_act]) 
            #print('ng')
            #print(phi_by_h_p['ngGraph'][my_act]) 
            #input('---')
        L=self.my_lower_bound_object
        all_p_found=set([])
        self.sum_negative_rc_PGM = 0
        for h in H:
            source=L.graph_node_2_agg_node[h][L.h_2_source_id[h]]
            sink=L.graph_node_2_agg_node[h][L.h_2_sink_id[h]]
            phi_by_h_p[h]['null_action']=0

            E_list=[]
            for fg in L.h_fg_2_q[h]:
                f=fg[0]
                g=fg[1]
                q=L.h_fg_2_q[h][fg]
                p=q[0]
                weight=-phi_by_h_p[h][p]+0.01
                #if weight<0:
                #    print('weight')
                #    print(weight)
                #    input('check')
                E_list.append([f,g,weight,p])
            #print(E_list)
            #input('----')
            out_dict_h=self.analyze_path_or_cycle(E_list, source, sink)
            if out_dict_h['cost']<-0.1 or out_dict_h['type']=='negative_cycle':
                #for p in out_dict_h['p_terms']:
                #    if p_to_cost[p]=min(p_to_cost[p])
                tmp=set(out_dict_h['p_terms']).intersection(set(self.forbidden_var_names))
                if len(tmp)<1:
                    print('out_dict_h')
                    print(out_dict_h)
                    print('out_dict_h[cost]')
                    print(out_dict_h['cost'])
                    print('out_dict_h[type]')
                    input('error here')
                all_p_found=all_p_found.union(tmp)
                
                self.sum_negative_rc_PGM=self.sum_negative_rc_PGM+out_dict_h['cost']
            #print('h')
            #print(h)
            #print('out_dict_h[cost')
            #print(out_dict_h['cost'])
        self.var_add_novel_pgm=[]

        #print('all_p_found')
        #print(all_p_found)
        #print('all_p_found')
        #print('self.var_name_map')
        #print(self.var_name_map)
        did_find=False
        #all_p_found=set(all_p_found)-set(self.forbidden_var_names)
        for act_u_v in all_p_found:
            #print('act_u_v')
            #print(act_u_v)
            #print('act_u_v')
            var_name_compress=self.var_name_map[act_u_v]
            var=self.var_dict[var_name_compress]
            self.var_add_novel_pgm.append(var)

    def analyze_path_or_cycle(self,E_list, source, sink):
        G = nx.DiGraph()
        for f, g, weight, p in E_list:
            G.add_edge(f, g, weight=weight, p=p)

        try:
            # Attempt shortest path using Bellman-Ford (handles negative weights)
            path = nx.bellman_ford_path(G, source, sink, weight='weight')
            cost = nx.path_weight(G, path, weight='weight')
            p_terms = {G[u][v]['p'] for u, v in zip(path, path[1:])}
            #print('path')
            #print(path)
            #print('p_terms')
            #print((u, v, G[u][v]['weight'], G[u][v]['p']) for u, v in zip(path, path[1:]))
            #print('cost')
            #print(cost)
            #print('---')

            return {
                "type": "shortest_path",
                "cost": cost,
                "path": [(u, v, G[u][v]['weight'], G[u][v]['p']) for u, v in zip(path, path[1:])],
                "p_terms": p_terms
            }
        except nx.NetworkXUnbounded:
            # Negative cycle detected
            for cycle in nx.simple_cycles(G):
                # Check if it's truly negative
                cycle_edges = list(zip(cycle, cycle[1:] + [cycle[0]]))
                weight_sum = sum(G[u][v]['weight'] for u, v in cycle_edges)
                #print('cycle_edges')
                #print(cycle_edges)
                #print('G[u][v][p] for u, v in cycle_edges')
                #p#rint(G[u][v]['p'] for u, v in cycle_edges)
                #p#rint('weight_sum')
                #print(weight_sum)
                #print('---')
                if weight_sum < 0:
                    return {
                        "type": "negative_cycle",
                        "cost": weight_sum,
                        "cycle": [(u, v, G[u][v]['weight'], G[u][v]['p']) for u, v in cycle_edges],
                        "p_terms": {G[u][v]['p'] for u, v in cycle_edges}
                    }
            return {
                "type": "negative_cycle_detected_but_not_extracted",
                "cost": None,
                "cycle": [],
                "p_terms": set()
            }
        except nx.NetworkXNoPath:
            return {
                "type": "no_path",
                "cost": None,
                "path": [],
                "p_terms": set()
            }
