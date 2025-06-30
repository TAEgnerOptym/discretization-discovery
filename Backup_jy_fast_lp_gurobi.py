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
                 init_forbidden_names,my_lower_bound_object,
                 K=100, verbose=True, remove_choice=2, alg_use=1, debug_on=False,
                 min_improvement_dump=0.1, epsilon=1e-4):

        #print('verbose')
        #print(verbose)
        #input('--')
        self.options = {
                "WLSACCESSID": "8f7bb9d6-8fe5-4349-9dd3-6abbaa9199a0",
                "WLSSECRET": "cb02810a-e0e2-4a1f-8fc0-fd375f65fc65",
                "LICENSEID": 2660300
        }
        self.my_lower_bound_object=my_lower_bound_object
        self.verbose = verbose
        self.remove_choice = remove_choice
        self.alg_use = alg_use
        self.min_improvement_dump = min_improvement_dump
        self.epsilon = epsilon
        self.debug_on = debug_on
        self.hist = {'lp': [],'numCurStart': [], 'numCurMid': [], 'numCurEnd': [], 'time_iter': []}
        self.max_terms_add_per_round = K
        self.all_possible_forbidden_names = all_possible_forbidden_names


        #init_forbidden_names=[]
        self.init_forbidden_names = init_forbidden_names
        
        #print('init_forbidden_names')
        #print(init_forbidden_names)
        #print('init_forbidden_names')
        #input('---')
        self.dict_var_name_2_obj=dict_var_name_2_obj
        self.dict_var_con_2_lhs_exog=dict_var_con_2_lhs_exog
        self.dict_con_name_2_LB=dict_con_name_2_LB
        self.dict_var_con_2_lhs_eq=dict_var_con_2_lhs_eq
        self.dict_con_name_2_eq=dict_con_name_2_eq
        self.running_removal=False
        self.K=K
        
        self.prepare_for_additions_novel()
        self.prepare_for_clean_novel()
        baseline_check_debug=True
        baseline_valu=np.inf
        baseline_time_lp=np.inf
        if baseline_check_debug==True:
            self.init_forbidden_names = []
            print('running Baseline')
            #self.call_solver_warm_no_path()
            self.call_solver_warm_no_path()#_clean_each
            baseline_valu=self.hist['lp'][-1]
            baseline_time_lp=self.hist['time_iter'][0]
            self.hist = {'lp': [],'numCurStart': [], 'numCurMid': [], 'numCurEnd': [], 'time_iter': []}
            
            
            input('done running baseline')
            self.init_forbidden_names=init_forbidden_names
        self.call_solver_warm_no_path_clean_each()
        if baseline_check_debug==True and abs(baseline_valu-self.hist['lp'][-1])>0.001:
            print('disagreement')
            print((baseline_valu-self.hist['lp'][-1]))
            input('ERROR')
        print('self.hist')
        print(self.hist)
        if baseline_check_debug==True:
            
            print('self.hist[lp]')
            print('time iter base')
            print(self.hist['time_iter'])
            print('sum times')
            print(sum(self.hist['time_iter']))
            print('baseline_time_lp')
            print(baseline_time_lp)
        print('self.times_accel')
        print(self.time_accel)
        print('sum(self.time_accel)')
        print(sum(self.time_accel))
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
        #remove variables forced to zero.  
        print('hi')
        vars_names_to_remove=[]
        cons_names_to_remove=[]
        for var in self.all_removable_vars:
            var_name_orig=self.var_name_rev_map[var.VarName]

            if (var.ub <0.5) !=(var in self.current_forbidden_vars) :
                print('var_name_orig')
                print(var_name_orig)
                print('var.ub')
                print(var.ub)
                print('var in self.current_forbidden_vars')
                print(var in self.current_forbidden_vars)
                input('wrong here ')
            #if var_name_orig=="act_0_9":
            #    print('var.ub')
            #    print(var.ub)
            #    input('look here')
            if var.ub>0.5:
                continue
            
            for v in self.mapping_remove_var[var_name_orig]:
                vars_names_to_remove.append(v)
            for con in self.mapping_remove_con[var_name_orig]:
                cons_names_to_remove.append(con)
        

        

        TMP_var_dict = {var.VarName: var for var in M.getVars()}
        TMP_con_dict = {con.ConstrName: con for con in M.getConstrs()}

        vars_to_remove=[]
        for var_name_orig in vars_names_to_remove:
            var_name_compress=self.var_name_map[var_name_orig]
            var=TMP_var_dict[var_name_compress]
            vars_to_remove.append(var)
        
        cons_to_remove=[]
        for con_name_orig in cons_names_to_remove:
            con_name_compress=self.con_name_map[con_name_orig]
            con=TMP_con_dict[con_name_compress]
            cons_to_remove.append(con)
        
        for var in vars_to_remove:
            M.remove(var)


        M.update()

        for con in cons_to_remove:
            M.remove(con)
        self.vars_names_to_remove=vars_names_to_remove
        self.cons_names_to_remove=cons_names_to_remove
        M.update()

        return M
    

    def call_solver_warm_no_path_clean_each(self):
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
                    model_compact_clean=model.copy()
                    model_compact_clean=self.clean_model(model_compact_clean)
                    model_compact_clean.update()
                    print('he 1')
                    t0=time.time()
                    model_compact_clean.optimize()
                    t0=time.time()-t0
                    print('he 2')

                    #lp_primal_solution_compact = {
                    #    self.var_name_rev_map[v.VarName]: v.X for v in self.vars_list
                    #}
                    self.time_accel.append(t0)
                    print('he 3')

                    t1=time.time()
                    model.optimize()
                    t1=time.time()-t1
                    print('he 4')

                    self.tot_lp_time += t1
                    
                    self.hist['time_iter'].append(t1)
                    self.hist['lp'].append(model.ObjVal)
                    if abs(model_compact_clean.ObjVal-model.ObjVal)>0.001:
                        #print('vars_list')
                        #print(self.vars_list)
                        lp_primal_solution = {
                            self.var_name_rev_map[v.VarName]: v.X for v in self.vars_list
                        }
                        for v in lp_primal_solution:
                            if lp_primal_solution[v]>0.0001 and v in self.vars_names_to_remove:
                                print('v')
                                print(v)
                                print('lp_primal_solution[v]')
                                print(lp_primal_solution[v])
                                var_orig=self.var_dict[self.var_name_map[v]]
                                print('var_orig.ub')
                                print(var_orig.ub)
                                v
                                input('wrong')
                        print('model_compact_clean.ObjVal')
                        print(model_compact_clean.ObjVal)
                        print('model.ObjVal')
                        print(model.ObjVal)
                        input('errror here')
                    self.hist['numCurStart'].append(self.countCurP())
                    cur_bound=model.ObjVal
                    print('cur_bound')
                    print(cur_bound)
                    print('cur_bound')
                    #input('cur_bound')
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
                        #continue
                    self.hist['numCurMid'].append(self.countCurP())
                    print('counter')
                    print(counter)

                    if len(self.var_add_novel)<1:
                        break
                    self.remove_from_forbidden(self.var_add_novel)
                    self.hist['numCurEnd'].append(self.countCurP())
                    counter=counter+1
                    model.update()
        print('self.time_accel')
        print(self.time_accel)
        print('sum(self.time_accel)')
        print(sum(self.time_accel))
        print('self.time_accel')


   
    def call_solver_warm_no_path(self):
        options=self.options
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
                    model.reset()
                    t1=time.time()
                    model.optimize()
                    t1=time.time()-t1
                    self.tot_lp_time += t1
                    self.hist['time_iter'].append(t1)
                    self.hist['lp'].append(model.ObjVal)
                    self.hist['numCurStart'].append(self.countCurP())
                    cur_bound=model.ObjVal
                    if len(self.init_forbidden_names)==0:
                        break
                    print('cur_bound')
                    print(cur_bound)
                    print('cur_bound')
                    #input('cur_bound')
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
                        #continue
                    self.hist['numCurMid'].append(self.countCurP())
                    print('counter')
                    print(counter)

                    if len(self.var_add_novel)<1:
                        break
                    self.remove_from_forbidden(self.var_add_novel)
                    self.hist['numCurEnd'].append(self.countCurP())
                    counter=counter+1

        
    def countCurP(self):
        count=0
        for var in self.all_removable_vars:
            if var.ub <gp.GRB.INFINITY:
                count=count+1
        return count
   
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
        
        #self.lp_dual_solution = dict(zip((rev_map[c.ConstrName] for c in constrs), pi_values))
        self.lp_dual_solution = dict(
            zip(
                (rev_map[c.ConstrName] for c in constrs if c.ConstrName in rev_map),
                (pi for c, pi in zip(constrs, pi_values) if c.ConstrName in rev_map)
            )
        )

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

        ### NEW MATERIAL
        L=self.my_lower_bound_object
        dual_slack_amounts=dict()
        dual_con_contrib=dict()
        dual_big_edge_contrib=dict()
        for act_uv in L.action_2_cost:
            dual_slack_amounts[act_uv]=0
            dual_con_contrib[act_uv]=0
            dual_big_edge_contrib[act_uv]=0
            
        ##print('dual_slack_amounts')
        #print(dual_slack_amounts)
        for v_con in L.action_con_2_contrib:
            var_name=v_con[0]
            con_name=v_con[1]
            weight_use=L.action_con_2_contrib[v_con]
            dual_val=self.lp_dual_solution[con_name]
            sign_use=-1
            tmp=sign_use*weight_use*dual_val
            dual_con_contrib[var_name]+=tmp
            
            #self.dict_var_con_2_lhs_exog[tuple([var_name,con_name])]=self.action_con_2_contrib[v_con]
        #print('dual_con_contrib')
        #print(dual_con_contrib)
        #print('self.all_possible_forbidden_names')
        #print(self.all_possible_forbidden_names)
        dual_big_edge_contrib_by_h=dict()
        dual_slack_amounts_components=dict()
        for my_act in L.action_2_cost:
            dual_big_edge_contrib_by_h[my_act]=dict()
            trm1=self.dict_var_name_2_obj[my_act]
            trm2=dual_con_contrib[my_act]
            trm_3=0
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
                    if my_act==L.null_action and dual_f>dual_g+0.0001:
                        print('dual_f')
                        print(dual_f)
                        print('dual_g')
                        print(dual_g)
                        print('f')
                        print(f)
                        print('g')
                        print(g)
                        input('error here')
                    

                    dual_fg_gap=-dual_f+dual_g
                    #if h=='ngGraph':
                    #    dual_fg_gap=-dual_fg_gap
                        #input('--flippiing-')
                   # if  my_act=='act_1_21':
                   #     print('dual_fg_gap')
                   #     print(dual_fg_gap)
                   #     print('dual_f,dual_g')
                   #     print([dual_f,dual_g])
                   #     print('h')
                   #     print(h)
                  #      flow_con_name_f='flow_in_out_h='+h+"_n="+f
                  #      flow_con_name_g='flow_in_out_h='+h+"_n="+g
                  #      dual_f_2=self.lp_dual_solution[flow_con_name_f]
                  #      dual_g_2=self.lp_dual_solution[flow_con_name_g]
                  #      print('[dual_f_2,dual_g_2]')
                  #      print([dual_f_2,dual_g_2])
                    trm3_by_part=min([trm3_by_part,dual_fg_gap])
                    #if act_u_v not in self.forbidden_var_names:

                       # dual_f-dual_g#
                if trm3_by_part==np.inf or np.isnan(trm3_by_part):
                    print('trm3_by_part')
                    print(trm3_by_part)
                    input('errr')
                #if  my_act=='act_1_21':
                #    print('trm3_by_part at end')
                #    print(trm3_by_part)
                #    print('h')
                #    print(h)
                #    input('---')
                dual_big_edge_contrib_by_h[my_act][h]=trm3_by_part
            trm3=sum(dual_big_edge_contrib_by_h[my_act].values())
            dual_big_edge_contrib[my_act]=trm3
            
            dual_slack_amounts_components[my_act]=[trm1,trm2,trm3]
            dual_slack_amounts[my_act]=trm1+trm2+trm3
            if np.isnan(dual_slack_amounts[my_act])==True:
                print('trm1')
                print(trm1)
                print('trm2')
                print(trm2)
                print('trm3')
                print(trm3)
                print('dual_big_edge_contrib_by_h[act_u_v]')
                print(dual_big_edge_contrib_by_h[my_act])
                input('error here')
        selected = heapq.nsmallest(self.K, (v_uv for v_uv in dual_slack_amounts.items() if v_uv[1] < -1e-4), key=lambda t: t[1])
        self.var_add_novel=[]
        #print('dual_slack_amounts')
        ##print(dual_slack_amounts)
        #print('selected')
        #print(selected)
        if 1<0:
            for my_act in L.action_2_cost:

                if my_act not in self.forbidden_var_names:
                    red_cost=self.reduced_costs_dict[my_act]
                    if np.isnan(dual_slack_amounts[my_act]) or abs(red_cost-dual_slack_amounts[my_act])>0.001:
                        print('-------')
                        print('-------')
                        print('-------')
                        print('my_act')
                        print(my_act)
                        
                        print('dual_slack_amounts[my_act]')
                        print(dual_slack_amounts[my_act])
                        print('self.dict_var_name_2_obj[my_act]')
                        print(self.dict_var_name_2_obj[my_act])
                        print('dual_con_contrib[my_act]')
                        print(dual_con_contrib[my_act])
                        print('gap=')
                        print(red_cost-dual_slack_amounts[my_act])
                        print('red_cost')
                        print(red_cost)
                        print('self.lp_primal_solution[my_act]')
                        print(self.lp_primal_solution[my_act])
                        pred_red_here=self.dict_var_name_2_obj[my_act]+dual_con_contrib[my_act]
                        tot_baseline_interation=0
                        dual_contrib_baseline_by_h=dict()
                        for h in L.graph_names:
                            eq_match_name='action_match_h='+h+"_p="+my_act
                            dual_contrib_baseline_by_h[h]=self.lp_dual_solution[eq_match_name]
                            pred_red_here+=dual_contrib_baseline_by_h[h]
                            tot_baseline_interation+=dual_contrib_baseline_by_h[h]
                            #print('h')
                            #print(h)
                            #print('eq_match_name')
                            #print(eq_match_name)
                        print('dual_contrib_baseline_by_h')
                        print(dual_contrib_baseline_by_h)
                        print('dual_big_edge_contrib_by_h[my_act]')
                        print(dual_big_edge_contrib_by_h[my_act])
                        print('sum(dual_big_edge_contrib_by_h[my_act].values())')
                        print(sum(dual_big_edge_contrib_by_h[my_act].values()))
                        print('dual_slack_amounts_components[act_u_v]')
                        print(dual_slack_amounts_components[my_act])
                        print('tot_baseline_interation')
                        print(tot_baseline_interation)
                        if (pred_red_here-red_cost)>0.001:
                            print('pred_red_here')
                            print(pred_red_here)
                            print('DOUBLE BAD ERROR')
                        print(' self.nov_h_uv_2_fg[ngGraph][my_act]')
                        print( self.nov_h_uv_2_fg['ngGraph'][my_act])
                        input('error no MATCH')
                    #else:
                    #    print('all good ')
                    #    print('my_act')
                    #    print(my_act)
                    #    print('dual_slack_amounts_components[my_act]')
                    #    print(dual_slack_amounts_components[my_act])
                    #    print('sum(dual_big_edge_contrib_by_h[my_act].values())')
                    #    print(dual_big_edge_contrib_by_h[my_act].values())
        for act_u_v_term in selected:
            act_u_v=act_u_v_term[0]
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
                    
                    print('h')
                    print(h)
                    print('my_act')
                    print(my_act)
                    print('L.null_action')
                    print(L.null_action)
                    self.nov_h_uv_2_fg[h][my_act].append(fg)                    

        self.nov_node_2_flow_con_name=dict()
        for h in L.graph_names:
            self.nov_node_2_flow_con_name[h]=dict()
            for f in L.agg_node_2_nodes[h]:
                my_con_name='flow_in_out_h='+h+"_n="+f
                self.nov_node_2_flow_con_name



    def apply_compression(self,):
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


       # if len(self.forbidden_var_names) == 0:
       #     print("No forbidden variables found in model. This may be a typo.")
       #     input('---')
