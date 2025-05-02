
import xpress as xp
import time
import numpy as np

class jy_fast_lp:
    
    def __init__(self,lp_prob, var_dict, all_possible_forbidden_names,init_forbiden_names, K=20, verbose=False,remove_choice=3,alg_use=1,debug_on=False,min_improvement_dump=.1,epsilon=.0001):
        #Solve LP while progressively relaxing forbidden variable bounds based on primal values.
        #""
        #Parameters:
        #lp_prob: xpress.problem
        #    The LP model.
        #var_dict:  dictionary mapping variable names to varaibles 
        #all_possible_forbidden_names:  all variables that can ever be removed
        #init_forbiden_names:  subset of all_possible_forbidden_names excluding positive terms
        #epsilon: float
        #    Small positive value to set upper bounds.
        #K: int
        #    Maximum number of variables to relax per iteration (default 10).
        #verbose: bool
        #    Whether to print progress.
        # remove_choice:
        # 1.  Never remove
        # 2.  remove terms after decrease  of LP bound  and are positive 
        # 3.  remove terms after improvment of LP bound and are non-postive 
        # alg_use:  
        #   option for alg to use for xpress
        #min_improvement_dump:  
        #   minium bound improvme t
        #Returns:
        #lp_prob: xpress.problem
        #    The solved LP after adjustments.
        #pos_based_forbidden_names:  
        #    Computed after algorihtm
        #red_cost_forbidden_names
        #    Computed after algorihtm
        #total_time_lp
        self.lp_prob=lp_prob
        self.var_dict=var_dict
        self.all_possible_forbidden_names=all_possible_forbidden_names
        self.init_forbiden_names=init_forbiden_names
        self.max_terms_add_per_round=K
        self.verbose=verbose
        self.remove_choice=remove_choice
        self.alg_use=alg_use
        self.min_improvement_dump=min_improvement_dump
        self.epsilon=epsilon
        self.debug_on=debug_on
        self.hist=dict()

        self.hist['lp']=[]
        self.hist['numCurMid']=[]
        self.hist['numCurEnd']=[]
        self.hist['time_iter']=[]
        self.setup_alg()

        self.call_core_alg()
        if debug_on==True:
            self.backup_lp()
        
        if verbose==True:
            input('done call')


    def setup_alg(self):
        self.forbidden_var_names=set()
        self.tot_lp_time=0
        lp_prob=self.lp_prob
        #if self.init_choice==1:
        #    self.forbidden_var_names = set(all_possible_forbidden_names)
        #if self.init_choice==2:
        #    self.forbidden_var_names =set(pos_based_forbidden_names)
        #if self.init_choice==3:
        #    self.forbidden_var_names =set(red_cost_forbidden_names)
        # Mapping from name to variable
        
        self.all_removable_vars = [self.var_dict[name] for name in self.all_possible_forbidden_names]
        # Only keep existing variables
        self.vars_list = self.lp_prob.getVariable()
       # print('all_removable_vars')
       # print(self.all_removable_vars)
       # print('len(all_removable_vars)')
       # print(len(self.all_removable_vars))
       # print('len(all_possible_forbidden_names)')
       #print(len(self.all_possible_forbidden_names))
       # input('all_removable_vars')
        lp_prob.controls.defaultalg = self.alg_use  # primal simplex

        


    def call_core_alg(self):
        self.incumbant_lp_val=np.inf
        self.current_forbidden_vars =set([])# [self.all_vars[name] for name in self.forbidden_var_names]
        self.forbidden_var_names=set([])#self.init_forbiden_names.copy()
        
        vars_2_make_remove=[self.var_dict[name] for name in self.init_forbiden_names]
        self.add_2_forbidden(vars_2_make_remove)

        if  len(self.forbidden_var_names)==0:
            print("No forbidden variables found in model. this must be a typo")
            input('---')
        iter=0
        if self.verbose==True:
            input('about to start the internal lp solver')
        #self.verbose=True
        while True:
            iter=iter+1
            self.compute_bound()
            self.hist['lp'].append(self.lp_obj_val)
            if self.lp_obj_val>self.incumbant_lp_val+0.01:
                input('bound increased error')
            if len(self.forbidden_vars_with_neg_red_cost)<0.5:
                break
            if self.lp_obj_val<self.incumbant_lp_val-self.min_improvement_dump:
                self.incumbant_lp_val=self.lp_obj_val
                if self.remove_choice==2:
                    self.remove_all_non_pos_after_improvemnt()
                if self.remove_choice==3:
                    self.remove_all_pos_red_cost_after_improvemnt()
            self.hist['numCurMid'].append(len(self.forbidden_var_names))


            self.add_neg_red_cost_vars()
            self.hist['numCurEnd'].append(len(self.forbidden_var_names))

            if len(self.forbidden_var_names)>len(self.all_possible_forbidden_names):
                print('len(self.forbidden_var_names)')
                print(len(self.forbidden_var_names))
                print('len(self.all_possible_forbidden_names)')
                print(len(self.all_possible_forbidden_names))
                input('error here')

            if self.verbose==True:
                print('iter')
                print(iter)
                print('self.lp_obj_val')
                print(self.lp_obj_val)
                print('self.incumbant_lp_val')
                print(self.incumbant_lp_val)
                print('len(forbidden_vars_with_neg_red_cost')
                print(len(self.forbidden_vars_with_neg_red_cost) )
                print('len(forbidden_var_names)')
                print(len(self.forbidden_var_names))
                print('---')
                
        if self.verbose==True:
            print('self.hist')
            print(self.hist)
            print('sum(self.hist[time_iter])')
            print(sum(self.hist['time_iter']))
            input('paused')
            

    def get_solution(self):
        
        a1=self.lp_prob
        a2=self.total_solve_time
        a3=self.active_removable_vars
        a4=self.non_pos_red_cost_removable
        return [a1,a2,a3,a4]

    def backup_lp(self):
        lp_prob=self.lp_prob
        vars_list=self.all_removable_vars
        if self.debug_on==True:
            input('debug on ')
            old_val=lp_prob.getObjVal()
            lp_prob.controls.defaultalg = 4  # primal simplex

            lp_prob.chgbounds(vars_list, ['U'] * len(vars_list), [1000000000] * len(vars_list))
            t_bakcup=time.time()
            lp_prob.solve()
            t_bakcup=time.time()-t_bakcup
            new_val=lp_prob.getObjVal()
           
            print('new_val')
            print(new_val)
            print('new_val')
            print(new_val)
            print('t_bakcup')
            print(t_bakcup)
            print('tot_lp_time')
            print(self.tot_lp_time)
            if abs(new_val-old_val)>.001:
                input('error')
            input('done debug')

    def add_2_forbidden(self,selected_remove_vars):
        lp_prob=self.lp_prob
        selected_remove_vars=set(selected_remove_vars)-self.current_forbidden_vars
        selected_remove_vars=list(selected_remove_vars)
        lp_prob.chgbounds(selected_remove_vars, ['U'] * len(selected_remove_vars), [0] * len(selected_remove_vars))
        if self.verbose==True:
            print('removing this many:  ' +str(len(selected_remove_vars)))
            print('--')
        for var in selected_remove_vars:
            self.forbidden_var_names.add(var.name)
            self.current_forbidden_vars.add(var)
        #print(len(selected_remove_vars))
        #input('len(selected_remove_vars)')
    def remove_from_forbidden(self,selected_add_vars):
        lp_prob=self.lp_prob
        lp_prob.chgbounds(selected_add_vars, ['U'] * len(selected_add_vars), [np.inf] * len(selected_add_vars))
        for var in selected_add_vars:
            self.forbidden_var_names.remove(var.name)
            self.current_forbidden_vars.remove(var)
    def remove_all_pos_red_cost_after_improvemnt(self):
        selected_remove_vars=self.pos_red_cost_removable
        self.add_2_forbidden(selected_remove_vars)

    def remove_all_non_pos_after_improvemnt(self):
        selected_remove_vars=self.in_active_removable_vars
        self.add_2_forbidden(selected_remove_vars)

    def add_neg_red_cost_vars(self):
        self.forbidden_vars_with_neg_red_cost.sort(key=lambda x: -x[1], reverse=True)
        selected_forbidden_vars = [v for v, _ in self.forbidden_vars_with_neg_red_cost[:self.max_terms_add_per_round]]
        self.remove_from_forbidden(selected_forbidden_vars)
        
    def compute_bound(self):
        lp_prob=self.lp_prob
        t_start = time.time()
        lp_prob.solve()
        t_end = time.time()
        self.tot_lp_time=self.tot_lp_time+(t_end-t_start)
        self.hist['time_iter'].append(t_end-t_start)
        vals = lp_prob.getSolution()
        self.my_status=lp_prob.getProbStatus()
        if self.my_status!=1:
            input('status failed')
        self.lp_obj_val=lp_prob.getObjVal()
        lp_primal_solution = {
            var.name: vals[i]
            for i, var in enumerate(self.vars_list)
        }

        # Identify forbidden vars with nonzero primal values
        self.active_removable_vars = [
            v
            for v in self.all_removable_vars
            if lp_primal_solution.get(v.name, 0.0) > self.epsilon
        ]

        self.in_active_removable_vars = [
            v
            for v in self.current_forbidden_vars
            if v.name in self.all_possible_forbidden_names and abs(lp_primal_solution.get(v.name, 0.0)) < self.epsilon
        ]

        self.reduced_costs = lp_prob.getRCost()

        # Build mapping
        self.reduced_costs_dict = {var.name: self.reduced_costs[i] for i, var in enumerate(self.vars_list)}

        # Find active forbidden variables by reduced cost
        self.forbidden_vars_with_neg_red_cost = [
            (v, self.reduced_costs_dict.get(v.name, 0.0))
            for v in self.all_removable_vars
            if v.name in self.forbidden_var_names and self.reduced_costs_dict.get(v.name, 0.0) < -self.epsilon
        ]


        self.pos_red_cost_removable = [
            v
            for v in self.all_removable_vars
            if self.reduced_costs_dict.get(v.name, 0.0) > self.epsilon
        ]


        self.non_pos_red_cost_removable = [
            v
            for v in self.all_removable_vars
            if self.reduced_costs_dict.get(v.name, 0.0) < self.epsilon
        ]


