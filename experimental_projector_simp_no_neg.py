
from collections import defaultdict
#from src.common.route import route
from typing import Dict, DefaultDict, Set, List
import numpy as np
import pulp as pl
from pulp import LpProblem, LpVariable, LpMaximize, PULP_CBC_CMD
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xpress as xp
import networkx as nx
import time
from scipy.sparse import csr_matrix
import pulp
import random
import time
import bisect

class projector:

    def __init__(self,my_full_solver,h):
        self.time_dict_proj=dict()
        t1=time.time()
        self.h=h
        self.MF=my_full_solver
        self.my_lp=self.MF.my_lower_bound_LP
        self.agg_node_2_node=self.my_lp.agg_node_2_nodes[h]
        self.graph_node_2_agg_node=self.MF.graph_node_2_agg_node[h]

        self.lp_primal_solution=self.my_lp.lp_primal_solution
        self.this_fg_sink=self.my_lp.graph_node_2_agg_node[h][self.MF.h_2_sink_id[h]]
        self.this_fg_source=self.my_lp.graph_node_2_agg_node[h][self.MF.h_2_source_id[h]]
        self.compact_sink=self.MF.h_2_sink_id[self.h]
        self.compact_source=self.MF.h_2_source_id[self.h]
        self.fg_nodes_non_source_sink=set(self.agg_node_2_node)-set([self.this_fg_sink,self.this_fg_source])
        self.null_action=self.MF.null_action
        self.non_source_sink_agg_nodes=set(self.agg_node_2_node.keys())-set([self.this_fg_sink,self.this_fg_source])

        self.ij_2_fg=self.my_lp.h_ij_2_fg[h]
        self.fg_2_ij=self.my_lp.h_fg_2_ij[h]
        self.hij_2_P_orig=self.my_lp.hij_2_P[h]
        self.dict_PROJ_ineq_con_name_2_rhs=dict()
        self.dict_PROJ_eq_con_name_2_rhs=dict()
        self.dict_PROG_ineq_var_con_lhs=dict()
        self.dict_PROG_eq_var_con_lhs=dict()
        self.dict_PROJ_var_name_2_obj=dict()
        self.dict_PROJ_eq_con_name_2_rhs=dict()

        self.non_source_sink=set(self.graph_node_2_agg_node)-set([self.compact_sink ,self.compact_source])
        self.time_dict_proj['prior']=time.time()-t1
        t1=time.time()
        self.get_non_zero_terms_p()
        self.time_dict_proj['get_non_zero_terms_p']=time.time()-t1
        t1=time.time()
        self.get_non_zero_terms_ij_i()
        self.time_dict_proj['get_non_zero_terms_ij_i']=time.time()-t1
        t1=time.time()
        self.proj_make_vars()
        self.time_dict_proj['proj_make_vars']=time.time()-t1
        t1=time.time()
        self.proj_make_actions_match_exog()
        self.time_dict_proj['proj_make_actions_match_exog']=time.time()-t1
        #t1=time.time()
        #self.proj_make_equv_class_consist()
        #self.time_dict_proj['proj_make_equv_class_consist']=time.time()-t1
        t1=time.time()

        self.proj_make_proj_flow_i_plus_minus()
        self.time_dict_proj['proj_make_proj_flow_i_plus_minus']=time.time()-t1
        t1=time.time()
        debug_on=False
        if debug_on==True:
            #self.proj_make_equiv_flow()

            self.make_primal_feas()
            self.check_solution_feasibility()
        if my_full_solver.jy_opt['use_Xpress']==False:
            self.make_LP()
        else:
            self.make_xpress_LP()
        t1=time.time()
        self.make_new_splits()
        self.time_dict_proj['make_new_splits']=time.time()-t1
        t1=time.time()
        
        total = sum(self.time_dict_proj.values())
        time_percentage_projector = {key: (val / total if total != 0 else 0) for key, val in self.time_dict_proj.items()}
        if self.MF.jy_opt['verbose']>0.5:
            print('self.time_dict_proj')
            print(self.time_dict_proj)
            print('--')
            print('time_percentage_projector')
            print(time_percentage_projector)
            print('h above')
            print(self.h)
            print('----')
            #input('-look im here at end of verbose-')
    def proj_make_equiv_flow(self):

        for f in self.non_source_sink_agg_nodes:
            con_name_f='f_flo_zero_'+str(f)
            self.dict_PROJ_eq_con_name_2_rhs[con_name_f]=0
        for ij in self.non_zero_ij:
            i=ij[0]
            j=ij[1]
            f=self.graph_node_2_agg_node[i]
            g=self.graph_node_2_agg_node[j]
            var_name='Proj_ij_i'+i+'_j=_'+j

            if f==g:
                continue
            if f!=self.this_fg_source:
                con_name_f='f_flo_zero_'+str(f)
                self.dict_PROG_eq_var_con_lhs[tuple([var_name,con_name_f])]=1

            if g!=self.this_fg_sink:
                con_name_g='f_flo_zero_'+g
                self.dict_PROG_eq_var_con_lhs[tuple([var_name,con_name_g])]=-1
      
    def get_non_zero_terms_p(self):
        
        self.non_zero_p=[]
        self.non_zero_p_plus_null=[self.MF.my_lower_bound_LP.null_action]
        lp_primal_solution=self.MF.my_lower_bound_LP.lp_primal_solution
        DEBUG_non_zero_p=[]
        DEBUG_thresh=0.01
        for p in self.MF.all_non_null_action:
            if lp_primal_solution[p]>self.MF.jy_opt['epsilon']:
                self.non_zero_p.append(p)
                self.non_zero_p_plus_null.append(p)
            if lp_primal_solution[p]>DEBUG_thresh:
                DEBUG_non_zero_p.append(p)
        
    def get_non_zero_terms_ij_i(self):
        # Clear results
        self.non_zero_ij = []
        self.Nz_ij_2_q = {}
        self.q_2_NZ_ij = {}
        
        # Precompute a set for fast membership test.
        non_zero_set = set(self.non_zero_p_plus_null)
        
        # Iterate over each key (edge tuple) and its corresponding list in hij_2_P_orig.
        for ij, P_list in self.hij_2_P_orig.items():
            # Filter P_list, keeping only terms that are in non_zero_set.
            filtered = [p for p in P_list if p in non_zero_set]
            if filtered:  # If the filtered list is non-empty:
                # Append (i,j) to non_zero_ij.
                self.non_zero_ij.append(ij)
                # Sort filtered list and convert to a tuple for use as a key.
                sorted_q = tuple(sorted(filtered))
                self.Nz_ij_2_q[ij] = sorted_q
                # Use setdefault to update q_2_NZ_ij in one shot.
                self.q_2_NZ_ij.setdefault(sorted_q, []).append(ij)

    def proj_make_vars(self):


        for i in self.graph_node_2_agg_node:
            my_var_1_name='Proj_slack_pos_'+str(i)
            #my_var_2_name='Proj_slack_neg_'+str(i)
            self.dict_PROJ_var_name_2_obj[my_var_1_name]=1
            #self.dict_PROJ_var_name_2_obj[my_var_2_name]=1
            
        for ij in self.non_zero_ij:
            i=str(ij[0])
            j=str(ij[1])
            my_var='Proj_ij_i'+i+'_j=_'+j
            #this_cost=
            #if self.hij_2_P_orig[ij][0]=:
            #    this_cost=
            self.dict_PROJ_var_name_2_obj[my_var]=self.MF.jy_opt['offset_cost_edge_project']
        #for q in self.q_2_NZ_ij:
        #    for p in q:
        #        my_var='Proj_q'+str(q)+'_p=_'+p
        #        self.dict_PROJ_var_name_2_obj[my_var]=0
    
    def make_primal_feas(self):
        #assumes one customr per node
        
        self.default_primal_solution=dict()
        for var in self.dict_PROJ_var_name_2_obj:
            self.default_primal_solution[var]=0
        for i in self.graph_node_2_agg_node:
            my_var_1_name='Proj_slack_pos_'+str(i)
            #my_var_2_name='Proj_slack_neg_'+str(i)
            self.default_primal_solution[my_var_1_name]=100000
            #self.default_primal_solution[my_var_2_name]=100000
        
        h=self.h
        edges_fg=self.my_lp.h_fg_2_ij[h].keys()
        lp_primal_solution=self.MF.my_lower_bound_LP.lp_primal_solution
        self.non_zero_f=set([])
        self.non_zero_fg=[]
        for e in edges_fg:
            f=e[0]
            g=e[1]
            var_name_fg='EDGE_h='+h+'_f='+f+'_g='+g
            if lp_primal_solution[var_name_fg]>self.MF.jy_opt['epsilon']:
                did_find=0
                for ij in self.my_lp.h_fg_2_ij[h][e]:
                    i=ij[0]
                    j=ij[1]
                    if ij in self.non_zero_ij:
                        did_find=True
                        my_var_ij='Proj_ij_i'+i+'_j=_'+j
                        if self.default_primal_solution[my_var_ij]>0:
                            input('error here')
                        self.default_primal_solution[my_var_ij]=lp_primal_solution[var_name_fg]
                        q=self.Nz_ij_2_q[ij]
                        if len(q)!=1:
                            print('ij')
                            print(ij)
                            print('q')
                            print(q)
                            input('error in this spot1')
                        for p in q:
                            if p not in self.non_zero_p_plus_null or len(q)>1:
                                input('error ths spot 3')
                            var_name_2='Proj_q'+str(q)+'_p=_'+p
                            #p#rint('var_name_2')
                            #print(var_name_2)
                            #print('my_var_ij')
                            #print(my_var_ij)
                            self.default_primal_solution[var_name_2]+=lp_primal_solution[var_name_fg]
                            break
                        break
        


    def proj_make_actions_match_exog(self):

        for p in self.non_zero_p:
            con_name="p_select_"+p
            self.dict_PROJ_eq_con_name_2_rhs[con_name]=self.lp_primal_solution[p]
        
        for ij in self.non_zero_ij:
            i=ij[0]
            j=ij[1]
            my_var='Proj_ij_i'+i+'_j=_'+j
            for p in self.hij_2_P_orig[ij]:
                if p in self.non_zero_p:
                    my_con="p_select_"+p

                    self.dict_PROG_eq_var_con_lhs[tuple([my_var,my_con])]=1

    
    def proj_make_proj_flow_i_plus_minus(self):

        for i in self.non_source_sink:
            
            con_name_1='con_i_slack_pos_'+i
            #con_name_2='con_i_slack_neg_'+i

            self.dict_PROJ_ineq_con_name_2_rhs[con_name_1]=-self.MF.jy_opt['epsilon']
            #self.dict_PROJ_ineq_con_name_2_rhs[con_name_2]=-self.MF.jy_opt['epsilon']


            my_var_1_name='Proj_slack_pos_'+i
            #my_var_2_name='Proj_slack_neg_'+i
            #my_var_2_name='Proj_slack_pos_'+i

            self.dict_PROG_ineq_var_con_lhs[tuple([my_var_1_name,con_name_1])]=1
            #self.dict_PROG_ineq_var_con_lhs[tuple([my_var_2_name,con_name_2])]=1


        for ij in self.non_zero_ij:
            i=str(ij[0])
            j=str(ij[1])
            var_name='Proj_ij_i'+i+'_j=_'+j
            if i!=self.compact_source:
                con_name_1='con_i_slack_pos_'+i
                self.dict_PROG_ineq_var_con_lhs[tuple([var_name,con_name_1])]=1
                
                #con_name_1a='con_i_slack_neg_'+i
                #self.dict_PROG_ineq_var_con_lhs[tuple([var_name,con_name_1a])]=-1
                

            if j!=self.compact_sink:
                con_name_2='con_i_slack_pos_'+j
                self.dict_PROG_ineq_var_con_lhs[tuple([var_name,con_name_2])]=-1

                #con_name_2a='con_i_slack_neg_'+j
                #self.dict_PROG_ineq_var_con_lhs[tuple([var_name,con_name_2a])]=1

    def check_solution_feasibility(self):
        print('----')
        print('----')
        print('----')
        print('----')
        print('----')

        print('CHEKCING SOLUTION')
        X=self.default_primal_solution
        dict_var_name_2_obj=self.dict_PROJ_var_name_2_obj
        dict_var_con_2_lhs_exog=self.dict_PROG_ineq_var_con_lhs
        dict_con_name_2_LB=self.dict_PROJ_ineq_con_name_2_rhs
        dict_var_con_2_lhs_eq=self.dict_PROG_eq_var_con_lhs
        dict_con_name_2_eq=self.dict_PROJ_eq_con_name_2_rhs
        """
        Check if the solution provided in dictionary X is feasible for the LP.
        
        Parameters:
        X : dict
            Dictionary mapping variable names to their proposed values.
        dict_var_con_2_lhs_exog : dict
            Dictionary with keys (var_name, con_name) mapping to the coefficient
            for inequality (exogenous) constraints.
        dict_con_name_2_LB : dict
            Dictionary mapping constraint names to their lower bound (right-hand side) 
            for inequality constraints.
        dict_var_con_2_lhs_eq : dict
            Dictionary with keys (var_name, con_name) mapping to the coefficient
            for equality constraints.
        dict_con_name_2_eq : dict
            Dictionary mapping constraint names to their fixed value for equality constraints.
        tol : float, optional
            Tolerance for checking equality constraint feasibility.
        
        Returns:
        bool: True if the candidate solution is feasible, False otherwise.
        """
        
        feasible = True
        tol=.0001

        # --- Check inequality constraints ---
        ineq_expressions = {}
        for (var_name, con_name), coeff in dict_var_con_2_lhs_exog.items():
            ineq_expressions.setdefault(con_name, 0)
            ineq_expressions[con_name] += coeff * X.get(var_name, 0)

        for con_name, lhs_value in ineq_expressions.items():
            # For each inequality constraint, check if LHS >= lower bound.
            if con_name in dict_con_name_2_LB:
                lb = dict_con_name_2_LB[con_name]
                if lhs_value < lb - tol:
                    print(f"Inequality constraint '{con_name}' is violated: LHS = {lhs_value}, required >= {lb}")
                    feasible = False
                    input('error here ')

        # --- Check equality constraints ---
        eq_expressions = {}
        for (var_name, con_name), coeff in dict_var_con_2_lhs_eq.items():
            eq_expressions.setdefault(con_name, 0)
            eq_expressions[con_name] += coeff * X.get(var_name, 0)
            
        for con_name, lhs_value in eq_expressions.items():
            if con_name in dict_con_name_2_eq:
                target = dict_con_name_2_eq[con_name]
                if abs(lhs_value - target) > tol:
                    print(f"Equality constraint '{con_name}' is violated: LHS = {lhs_value}, required = {target}")
                    feasible = False
                    for (var, c_name), coeff in dict_var_con_2_lhs_eq.items():
                        if c_name == con_name:
                            var_value = X.get(var, 0)
                            term_value = coeff * var_value
                            print(f"  Variable '{var}': coefficient = {coeff}, value = {var_value}, product = {term_value}")
                
                    input('error here')

        if feasible:
            print("The solution is feasible!")
            #input('---')
        else:
            print("The solution is NOT feasible.")
            input('--')
            
        return feasible
               

    def make_LP(self):
        dict_var_name_2_obj=self.dict_PROJ_var_name_2_obj
        dict_var_con_2_lhs_exog=self.dict_PROG_ineq_var_con_lhs
        dict_con_name_2_LB=self.dict_PROJ_ineq_con_name_2_rhs
        dict_var_con_2_lhs_eq=self.dict_PROG_eq_var_con_lhs
        dict_con_name_2_eq=self.dict_PROJ_eq_con_name_2_rhs
        # --- Build the LP model ---
        t2=time.time()
        my_times_proj=[]
        lp_prob = pulp.LpProblem("MyLP", pulp.LpMinimize)

        # Create decision variables (all non-negative)
        t1=time.time()
        var_dict = {}
        for var_name, coeff in dict_var_name_2_obj.items():
            var_dict[var_name] = pulp.LpVariable(var_name, lowBound=0)
        my_times_proj.append(time.time()-t1)
        t1=time.time()
        # Define the objective function (minimize sum(obj_coeff * var))
        lp_prob += pulp.lpSum(dict_var_name_2_obj[var_name] * var_dict[var_name]
                            for var_name in dict_var_name_2_obj), "Objective"
        my_times_proj.append(time.time()-t1)
        t1=time.time()
        # --- Add inequality constraints (>=) ---
        # Group terms for each inequality constraint.
        ineq_expressions = {}
        for (var_name, con_name), coeff in dict_var_con_2_lhs_exog.items():
            ineq_expressions.setdefault(con_name, 0)
            ineq_expressions[con_name] += coeff * var_dict[var_name]
        my_times_proj.append(time.time()-t1)
        t1=time.time()
        # Add each inequality constraint to the model.
        for con_name, expr in ineq_expressions.items():
            if con_name in dict_con_name_2_LB:
                lp_prob += expr >= dict_con_name_2_LB[con_name], con_name #+ "_ineq"
        my_times_proj.append(time.time()-t1)
        t1=time.time()
        # --- Add equality constraints ---
        # Group terms for each equality constraint.
        eq_expressions = {}
        for (var_name, con_name), coeff in dict_var_con_2_lhs_eq.items():
            eq_expressions.setdefault(con_name, 0)
            eq_expressions[con_name] += coeff * var_dict[var_name]
        my_times_proj.append(time.time()-t1)
        t1=time.time()
        # Add each equality constraint to the model.
        for con_name, expr in eq_expressions.items():
            if con_name in dict_con_name_2_eq:
                lp_prob += expr == dict_con_name_2_eq[con_name], con_name #+ "_eq"
        my_times_proj.append(time.time()-t1)
        t1=time.time()
        # --- Solve the LP ---
        # Using the default CBC solver here.
        self.time_dict_proj['lp_prior']=time.time()-t2
        start_time = time.time()
        solver = pulp.PULP_CBC_CMD(msg=False)
        lp_prob.solve(solver)
        end_time = time.time()
        self.time_dict_proj['lp_time']=end_time-start_time
        t2=time.time()
        my_times_proj.append(end_time-start_time)
        self.lp_time=end_time-start_time
        self.lp_status=pulp.LpStatus[lp_prob.status]
        if self.lp_status=='Infeasible':
            input('ERROR infeasible')
        t1=time.time()
        self.lp_prob=lp_prob
        self.lp_primal_solution=dict()
        for var_name, var in var_dict.items():
            self.lp_primal_solution[var_name]=var.varValue

        my_times_proj.append(time.time()-t1)
        t1=time.time()
        
        self.lp_objective= pulp.value(lp_prob.objective)
        self.lp_dual_solution=dict()
        for con_name, constraint in lp_prob.constraints.items():

            self.lp_dual_solution[con_name]=constraint.pi
        my_times_proj.append(time.time()-t1)

       # print('my_times_proj')
       # print(my_times_proj)
       # print('my_times_proj/sum')
       # print(np.array(my_times_proj)/np.sum(np.array(my_times_proj)))
       # print('lpPortion')
       # print((end_time-start_time)/np.sum(np.array(my_times_proj)))
       # print('---')
        self.time_dict_proj['lp_post']=time.time()-t2

    def make_xpress_LP(self):

       # input('-in express lp--')
        import xpress as xp
        xp.init('C:/xpressmp/bin/xpauth.xpr')
        t2=time.time()
        dict_var_name_2_obj = self.dict_PROJ_var_name_2_obj
        dict_var_con_2_lhs_exog = self.dict_PROG_ineq_var_con_lhs
        dict_con_name_2_LB = self.dict_PROJ_ineq_con_name_2_rhs
        dict_var_con_2_lhs_eq = self.dict_PROG_eq_var_con_lhs
        dict_con_name_2_eq = self.dict_PROJ_eq_con_name_2_rhs

        # --- Build the LP model ---
        lp_prob = xp.problem("MyLP")
        lp_prob.setOutputEnabled(self.MF.jy_opt['verbose']>0.5)

        # Create decision variables (all non-negative) and store them in a list.
        var_dict = {}
        vars_list = []  # list of variables to add to the model
        if 1<0:
            for var_name, coeff in dict_var_name_2_obj.items():
                # Create a variable with lower bound 0.
                v = lp_prob.addVariable(name=var_name, lb=0)
                var_dict[var_name] = v
                vars_list.append(v)
        else:
            vars_list = [xp.var(name=name, lb=0) for name in dict_var_name_2_obj]

            # 2) register *each* one with the model
            lp_prob.addVariable(*vars_list)   # ← note the * here!

            # 3) now you can build var_dict
            var_dict = {v.name: v for v in vars_list}
        # Define the objective function (minimize sum of coeff * variable).
        # Converting the generator to a list to be safe.
        objective = xp.Sum([dict_var_name_2_obj[var_name] * var_dict[var_name]
                            for var_name in dict_var_name_2_obj])
        lp_prob.setObjective(objective, sense=xp.minimize)
        # --- Add inequality constraints (>=) ---
        # Group terms for each inequality constraint.
        ineq_expressions =  defaultdict(float)
        did_find_2 = False
        self.time_dict_proj['pre_XP_lp1']=time.time()-t2
        t2=time.time()
        for (var_name, con_name), coeff in dict_var_con_2_lhs_exog.items():
            ineq_expressions[con_name] += coeff * var_dict[var_name]
            
        self.time_dict_proj['pre_XP_lp2']=time.time()-t2
        t2=time.time()
        #for con_name, expr in ineq_expressions.items():
        #    con = xp.constraint(expr >= dict_con_name_2_LB[con_name], name=con_name)
        #    lp_prob.addConstraint(con)
        constraints = [
            xp.constraint(expr >= dict_con_name_2_LB[con_name], name=con_name)
            for con_name, expr in ineq_expressions.items()
        ]

        lp_prob.addConstraint(constraints)
        self.time_dict_proj['pre_XP_lp3']=time.time()-t2
        t2=time.time()
        eq_expressions =  defaultdict(float)
        for (var_name, con_name), coeff in dict_var_con_2_lhs_eq.items():
           
            eq_expressions[con_name] += coeff * var_dict[var_name]
        self.time_dict_proj['pre_XP_lp4']=time.time()-t2
        t2=time.time()
        # Add each equality constraint to the model.
        for con_name, expr in eq_expressions.items():
            #if con_name in dict_con_name_2_eq:
            con_eq = xp.constraint(expr == dict_con_name_2_eq[con_name], name=con_name)
            lp_prob.addConstraint(con_eq)
        self.time_dict_proj['pre_XP_lp5']=time.time()-t2
        t2=time.time()
        # --- Solve the LP ---
        lp_prob.controls.defaultalg = self.MF.jy_opt['proj_solver']

        self.time_dict_proj['pre_XP_lp_6']=time.time()-t2
        start_time = time.time()
        lp_prob.solve()
        end_time = time.time()

        self.lp_time = end_time - start_time
        self.time_dict_proj['lp_time']=self.lp_time
        t3=time.time()
        self.lp_prob = lp_prob
        #for var_name, var in var_dict.items():
        #    self.lp_primal_solution[var_name] = lp_prob.getSolution(var_name)

        self.lp_status = lp_prob.getProbStatus()
        self.lp_objective = lp_prob.getObjVal()
        self.lp_primal_solution = dict()
        self.time_dict_proj['post_XLP_1']=time.time()-t3
        t3=time.time()
        #
        vals = lp_prob.getSolution(vars_list)
        self.lp_primal_solution = {
            var.name: vals[i]
            for i, var in enumerate(vars_list)
        }
        self.time_dict_proj['post_XLP_2']=time.time()-t3
        t3=time.time()
        if 1<0:
            self.lp_dual_solution = {
                con.name: lp_prob.getDual(con)
                for con in lp_prob.getConstraint()
            }
        else:
            cons = lp_prob.getConstraint()
            # 2) One C‐call to fetch all duals in the same order
            duals = lp_prob.getDuals(cons)
            # 3) Build your dict in a single Python loop
            self.lp_dual_solution = {con.name: d for con, d in zip(cons, duals)}
        self.time_dict_proj['post_XLP_3']=time.time()-t3
        t3=time.time()
        if self.lp_status == 'Infeasible':
            input('HOLD')

        self.time_dict_proj['post_XLP_4']=time.time()-t3
        
    def make_new_splits(self):

        #get max value
        self.NEW_node_2_agg_node=self.graph_node_2_agg_node.copy()
        #max_val=max(self.graph_node_2_agg_node.values())
        start_value=0
        extra_string='rand_'+str(random.randint(0,100000000))+'_'
        i_2_dual=dict()
        for i_orig in self.non_source_sink:
            i=i_orig[:]
            if self.MF.jy_opt['use_Xpress']<0.5:
                i= i.replace(" ", "_")

                i= i.replace("[", "_")
                i= i.replace("]", "_")
            con_name_1='con_i_slack_pos_'+i
            #con_name_2='con_i_slack_neg_'+i
            if  con_name_1 not in self.lp_dual_solution:

                print('self.lp_dual_solution.keys()')
                print(self.lp_dual_solution.keys())
                print('missing ')
                print(con_name_1)
                print('i')
                print(i)
                print('type(con_name_1)')
                print(type(con_name_1))
                print('type(i)')
                print(type(i))
                print('ddd')
                input('hold')
            i_2_dual[i_orig]=self.lp_dual_solution[con_name_1]#-self.lp_dual_solution[con_name_2]
        
        
        count_change=0
        

        self.f_2_mean_val=dict()
        self.f_2_min_val=dict()
        self.f_2_max_val=dict()
        self.do_split_f=[]
        count_find=dict()
        for i in self.graph_node_2_agg_node:
            count_find[i]=0
        for f in self.non_source_sink_agg_nodes:
            my_sum=0
            my_min=np.inf
            my_max=-np.inf
            all_terms=[]
            all_names=[]
            for i in self.agg_node_2_node[f]:
                count_find[i]=count_find[i]+1
                this_term=i_2_dual[i]
                my_sum=my_sum+this_term#i_2_dual[i]
                my_max=max([my_max,this_term])
                my_min=min([my_min,this_term])
                all_names.append(i)
                all_terms.append(this_term)

            self.f_2_mean_val[f]=my_sum/len(self.agg_node_2_node[f])
            self.f_2_min_val[f]=my_min#my_sum/len(self.agg_node_2_node[f])
            self.f_2_max_val[f]=my_max#my_sum/len(self.agg_node_2_node[f])
            if self.f_2_max_val[f]-self.f_2_min_val[f]>self.MF.jy_opt['threshold_split']:#.0001:
                self.do_split_f.append(f)
                X=all_terms
                Y=all_names
                combined = list(zip(X, Y))

                # Sort the combined list based on the first element (values from X)
                combined_sorted = sorted(combined, key=lambda pair: pair[0])

                # Unzip the sorted pairs back into two lists
                X_sorted, Y_sorted = zip(*combined_sorted)

                # Convert tuples back to lists
                X_sorted = list(X_sorted)
                Y_sorted = list(Y_sorted)
                Q = [elem for pair in zip(X_sorted, Y_sorted) for elem in pair]
                #print('Q')
                #print(Q)
                #print('X_sorted')
                #print(X_sorted)
                #print('Y_sorted')
                #print(Y_sorted)
                #print('f')
                #print(f)
                #print('-------')
                #input('---')
        for i in self.graph_node_2_agg_node:
            if count_find[i]!=1 and i!=self.compact_source and i!=self.compact_sink:
                print('i')
                print(i)
                print('count_find[i]')
                print(count_find[i])
                print('self.graph_node_2_agg_node[i]')
                print(self.graph_node_2_agg_node[i])
                print('self.agg_node_2_node[self.graph_node_2_agg_node[i]]')
                print(self.agg_node_2_node[self.graph_node_2_agg_node[i]])
                input('error not found righ amount of tiems')
        if self.lp_objective>.0001 and len(self.do_split_f)==0:
            print('self.do_split_f')
            print(self.do_split_f)
            print('self.lp_objective')
            print(self.lp_objective)
            input('errror here')
        #for f in self.do_split_f:
        #    print('f')
        #    print(f)
        #    print('self.f_2_max_val[f]-self.f_2_min_val[f]')
        #    print(self.f_2_max_val[f]-self.f_2_min_val[f])

        #input('--')
        if 1<0:
            start_value=0
            for f in self.do_split_f:
                start_value=start_value+1
                count_pos=0
                count_tot=len(self.agg_node_2_node[f])
                for i in self.agg_node_2_node[f]:
                    if i_2_dual[i]>self.f_2_mean_val[f]:
                        self.NEW_node_2_agg_node[i]=extra_string+'_'+str(start_value)
                        count_change=count_change+1
                        count_pos=count_pos+1
        else:
            start_value=0
            num_thesh_use=self.MF.jy_opt['num_thresh_split_projector']
            for f in self.do_split_f:
                start_value=start_value+num_thesh_use
                count_pos=0
                extra_str_f=str(random.randint(0,100000000))
                tmp_dict=dict()
                for i in self.agg_node_2_node[f]:
                    tmp_dict[i]=i_2_dual[i]
                [chosen, new_dict]=self.quantize_dict_to_index(tmp_dict,num_thesh_use)
                for i in self.agg_node_2_node[f]:
                    self.NEW_node_2_agg_node[i]=extra_string+'_'+extra_str_f+'_'+str(new_dict[i])
                    count_change=count_change+1
                    count_pos=count_pos+1


    def quantize_dict_to_index(self,orig_dict, K):
    # 1) round all values to 3dp and get sorted uniques
        num_digits_keep=self.MF.jy_opt['roundingDiscretization_num_digits_keep']
        levels = sorted({round(v,num_digits_keep ) for v in orig_dict.values()})

        # 2) sample up to K uniformly‐spaced levels
        if len(levels) > K:
            N = len(levels)
            if K == 1:
                chosen = [levels[N//2]]
            else:
                chosen = [
                    levels[int(round(i * (N-1) / (K-1)))]
                    for i in range(K)
                ]
            #print('--')
            #print(levels)
            #print('chosen')
            #print(chosen)
            #input('levels  drop')
        else:
            chosen = levels
            #print('--')
            #print(levels)
            
            #input('levels no drop')

        chosen.sort()  # just in case

        # 3) snap each entry to the index of the nearest chosen level
        index_map = {}
        for key, val in orig_dict.items():
            r = round(val, num_digits_keep)
            i = bisect.bisect_left(chosen, r)

            # collect candidate indices
            idxs = []
            if i > 0:
                idxs.append(i-1)
            if i < len(chosen):
                idxs.append(i)

            # pick the idx whose chosen[idx] is closest to r
            best_idx = min(idxs, key=lambda j: abs(chosen[j] - r))
            index_map[key] = best_idx

        return chosen, index_map