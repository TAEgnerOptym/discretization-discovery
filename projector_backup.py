
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


class projector:

    def __init__(self,my_full_solver,h):
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
        
        self.ij_2_fg=self.my_lp.h_ij_2_fg[h]
        self.fg_2_ij=self.my_lp.h_fg_2_ij[h]
        self.hij_2_P_orig=self.my_lp.hij_2_P[h]
        self.dict_PROJ_ineq_con_name_2_rhs=dict()
        self.dict_PROJ_eq_con_name_2_rhs=dict()
        self.dict_PROG_ineq_var_con_lhs=dict()
        self.dict_PROG_eq_var_con_lhs=dict()
        self.dict_PROJ_var_name_2_obj=dict()

        self.get_non_zero_terms_p()
        self.get_non_zero_terms_f_fg()
        self.get_non_zero_terms_ij_i()
        self.get_ij_2_p_terms()
        self.proj_make_vars()
        self.proj_make_RHS_projector()
        self.proj_make_actions_match_exog()
        self.proj_make_flow_match_fg()
        self.proj_make_equv_class_consist()
        #self.proj_make_equiv_flow()

        self.proj_make_proj_flow_i_plus_minus()
        self.make_LP()
        self.make_new_splits()


    def get_non_zero_terms_p(self):
        
        self.non_zero_p=[]
        self.non_zero_p_plus_null=[self.MF.my_lower_bound_LP.null_action]
        lp_primal_solution=self.MF.my_lower_bound_LP.lp_primal_solution
        for p in self.MF.all_non_null_action:
            if lp_primal_solution[p]>self.MF.jy_opt['epsilon']:
                self.non_zero_p.append(p)
                self.non_zero_p_plus_null.append(p)
    def get_non_zero_terms_f_fg(self):
        h=self.h
        edges_fg=self.my_lp.h_fg_2_ij[h].keys()
        lp_primal_solution=self.MF.my_lower_bound_LP.lp_primal_solution
        h=self.h
        self.non_zero_f=set([])
        self.non_zero_fg=[]
        for e in edges_fg:
            f=e[0]
            g=e[1]
            var_name='EDGE_h='+h+'_f='+f+'_g='+g
            if lp_primal_solution[var_name]>self.MF.jy_opt['epsilon']:
                
                self.non_zero_f.add(f)
                self.non_zero_f.add(g)
                self.non_zero_fg.append(tuple([f,g]))

        if self.this_fg_sink in self.non_zero_f:

            self.non_zero_f.remove(self.this_fg_sink)
        if self.this_fg_source in self.non_zero_f:

            self.non_zero_f.remove(self.this_fg_source)
        for f in self.non_zero_f:
            self.non_zero_fg.append(tuple([f,f]))
        print('self.non_zero_fg')
        print(self.non_zero_fg)
        print('self.non_zero_f')
        print(self.non_zero_f)
        input('----')
    def get_non_zero_terms_ij_i(self):
        self.non_zero_i=set([])
        self.non_zero_ij=[]
        h=self.h
        for f in self.non_zero_f:
            for i in self.agg_node_2_node[f]:

                self.non_zero_i.add(i)
        for fg in self.non_zero_fg:
            for e in self.fg_2_ij[fg]:
                i=e[0]
                j=e[1]
                self.non_zero_ij.append(tuple([i,j]))

    def get_ij_2_p_terms(self):
        self.Nz_ij_2_p=dict()
        self.q_2_NZ_ij=dict()
        self.ij_2_remove=dict()
        self.i_2_remove=dict()
        for ij in self.non_zero_ij:
            self.Nz_ij_2_p[ij]=dict()
            q=[]
            i=ij[0]
            j=ij[1]
            
            for p in self.hij_2_P_orig[ij]:
                if p in self.non_zero_p_plus_null:
                    q.append(p)
            if len(q)>0:
                q=tuple(sorted(q))
                #print('q')
                #print(q)
                if q not in self.q_2_NZ_ij:
                    self.q_2_NZ_ij[q]=[]
                self.q_2_NZ_ij[q].append(ij)
                self.Nz_ij_2_p[ij]=q
            else:
                self.ij_2_remove.add(ij)
        did_find=dict()#self.non_zero_i.copy()
        for ij in self.Nz_ij_2_p:
            i=ij[0]
            j=ij[1]
            if i not in did_find:
                did_find[i]=0
            if j not in did_find:
                did_find[j]=0
            if ij not in self.ij_2_remove:
                i=ij[0]
                j=ij[1]
                did_find[i]=1
                did_find[j]=1

        for i in did_find:
            if did_find[i]==0:
                self.non_zero_i.remove[i]

        for ij in self.ij_2_remove:
            self.non_zero_ij.remove[ij]
            self.Nz_ij_2_p.remove[ij]

        self.Nz_fg_2_nz_ij=dict()
        for fg in self.non_zero_fg:
            self.Nz_fg_2_nz_ij[fg]=set([])
            for ij in self.fg_2_ij[fg]:
                if ij not in self.ij_2_remove:
                    self.Nz_fg_2_nz_ij[fg].add(ij)

    def proj_make_vars(self):
        for i in self.non_zero_i:
            my_var_1_name='Proj_slack_pos_'+i
            my_var_2_name='Proj_slack_neg_'+i
            self.dict_PROJ_var_name_2_obj[my_var_1_name]=1
            self.dict_PROJ_var_name_2_obj[my_var_2_name]=1

        for ij in self.non_zero_ij:
            i=ij[0]
            j=ij[1]
            my_var='Proj_ij_i'+i+'_j=_'+j
            self.dict_PROJ_var_name_2_obj[my_var]=0
        for q in self.q_2_NZ_ij:
            for p in q:
                my_var='Proj_q'+str(q)+'_p=_'+p
                self.dict_PROJ_var_name_2_obj[my_var]=0
    
    def proj_make_RHS_projector(self):

        #con Name p selected
        self.dict_PROJ_eq_con_name_2_rhs=dict()
        for p in self.non_zero_p:
            con_name="p_select_"+p
            self.dict_PROJ_eq_con_name_2_rhs[con_name]=self.lp_primal_solution[p]

        #con nam fg and ij match
        h=self.h
        for fg in self.non_zero_fg:
            
            con_name="fg_select_"+str(fg)
            f=fg[0]
            g=fg[1]
            if f!=g:
                edge_var='EDGE_h='+h+'_f='+str(f)+'_g='+str(g)
                self.dict_PROJ_eq_con_name_2_rhs[con_name]=self.lp_primal_solution[edge_var]

        #con name ij and selected actions aligh
        for q in self.q_2_NZ_ij:
            con_name='q_ij_agree_'+str(q)
            self.dict_PROJ_eq_con_name_2_rhs[con_name]=0
        if 0>1:
            for f in self.non_zero_f:
                con_name='f_flo_zero_'+str(f)
                self.dict_PROJ_eq_con_name_2_rhs[con_name]=0
        #
        for i in self.non_zero_i:
            con_name_1='con_i_slack_pos_'+str(i)
            con_name_2='con_i_slack_neg_'+str(i)
            print('con_name_1')
            print(con_name_1)
            print('con_name_2')
            print(con_name_2)
            print('i')
            print(i)
            #input('---')
            self.dict_PROJ_ineq_con_name_2_rhs[con_name_1]=-self.MF.jy_opt['epsilon']
            self.dict_PROJ_ineq_con_name_2_rhs[con_name_2]=-self.MF.jy_opt['epsilon']

    def proj_make_actions_match_exog(self):

        for q in self.q_2_NZ_ij:
            for p in q:
                if p !=self.null_action:
                    my_var='Proj_q'+str(q)+'_p=_'+p
                    my_con="p_select_"+p
                    self.dict_PROG_eq_var_con_lhs[tuple([my_var,my_con])]=1
    
    def proj_make_flow_match_fg(self):

        for fg in self.non_zero_fg:
            con_name="fg_select_"+str(fg)
            f=fg[0]
            g=fg[1]
            if f!=g:
                for ij in self.Nz_fg_2_nz_ij[fg]:
                    i=ij[0]
                    j=ij[1]
                    my_var='Proj_ij_i'+i+'_j=_'+j
                    self.dict_PROG_eq_var_con_lhs[tuple([my_var,con_name])]=1
        
    def proj_make_equv_class_consist(self):

        for q in self.q_2_NZ_ij:
            con_name='q_ij_agree_'+str(q)
            for ij in self.q_2_NZ_ij[q]:

                i=ij[0]
                j=ij[1]
                var_name='Proj_ij_i'+i+'_j=_'+j
                self.dict_PROG_eq_var_con_lhs[tuple([var_name,con_name])]=1
            for p in q:
                var_name_2='Proj_q'+str(q)+'_p=_'+p
                self.dict_PROG_eq_var_con_lhs[tuple([var_name_2,con_name])]=-1
    
    def proj_make_equiv_flow(self):
        input('NO GOOD NOT HERE')
        for ij in self.non_zero_ij:
            i=ij[0]
            j=ij[1]
            f=self.graph_node_2_agg_node[i]
            g=self.graph_node_2_agg_node[j]
            var_name='Proj_ij_i'+i+'_j=_'+j

    
            if f!=self.this_fg_source:
                con_name_f='f_flo_zero_'+f
                self.dict_PROG_eq_var_con_lhs[tuple([var_name,con_name_f])]=1

            if g!=self.this_fg_sink:
                con_name_g='f_flo_zero_'+g
                self.dict_PROG_eq_var_con_lhs[tuple([var_name,con_name_g])]=1
            
    def proj_make_proj_flow_i_plus_minus(self):

        for i in self.non_zero_i:
            con_name_1='con_i_slack_pos_'+i
            con_name_2='con_i_slack_neg_'+i

            my_var_1_name='Proj_slack_pos_'+i
            my_var_2_name='Proj_slack_neg_'+i
            self.dict_PROG_ineq_var_con_lhs[tuple([my_var_1_name,con_name_1])]=1
            self.dict_PROG_ineq_var_con_lhs[tuple([my_var_2_name,con_name_2])]=1


        for ij in self.non_zero_ij:
            i=ij[0]
            j=ij[1]
            var_name='Proj_ij_i'+i+'_j=_'+j
            if i!=self.compact_source:
                con_name_1='con_i_slack_pos_'+i
                self.dict_PROG_ineq_var_con_lhs[tuple([var_name,con_name_1])]=1
                
                con_name_1a='con_i_slack_neg_'+i
                self.dict_PROG_ineq_var_con_lhs[tuple([var_name,con_name_1a])]=-1
                

            if j!=self.compact_sink:
                con_name_2='con_i_slack_pos_'+j
                self.dict_PROG_ineq_var_con_lhs[tuple([var_name,con_name_2])]=-1

                con_name_2a='con_i_slack_neg_'+j
                self.dict_PROG_ineq_var_con_lhs[tuple([var_name,con_name_2a])]=1

                

    def make_LP(self):
        print('-----')
        dict_var_name_2_obj=self.dict_PROJ_var_name_2_obj
        dict_var_con_2_lhs_exog=self.dict_PROG_ineq_var_con_lhs
        dict_con_name_2_LB=self.dict_PROJ_ineq_con_name_2_rhs
        dict_var_con_2_lhs_eq=self.dict_PROG_eq_var_con_lhs
        dict_con_name_2_eq=self.dict_PROJ_eq_con_name_2_rhs
        # --- Build the LP model ---
        lp_prob = pulp.LpProblem("MyLP", pulp.LpMinimize)

        # Create decision variables (all non-negative)
        var_dict = {}
        for var_name, coeff in dict_var_name_2_obj.items():
            var_dict[var_name] = pulp.LpVariable(var_name, lowBound=0)

        # Define the objective function (minimize sum(obj_coeff * var))
        lp_prob += pulp.lpSum(dict_var_name_2_obj[var_name] * var_dict[var_name]
                            for var_name in dict_var_name_2_obj), "Objective"

        # --- Add inequality constraints (>=) ---
        # Group terms for each inequality constraint.
        ineq_expressions = {}
        for (var_name, con_name), coeff in dict_var_con_2_lhs_exog.items():
            ineq_expressions.setdefault(con_name, 0)
            ineq_expressions[con_name] += coeff * var_dict[var_name]

        # Add each inequality constraint to the model.
        for con_name, expr in ineq_expressions.items():
            if con_name in dict_con_name_2_LB:
                #print('con_name')
                #print(con_name)
                #input('---')
                if 1>0 or con_name[0:len('con_i_slack')]=='con_i_slack':
                    print('con_name')
                    print(con_name)
                    print('expr')
                    print(expr)
                    print('dict_con_name_2_LB[con_name]')
                    print(dict_con_name_2_LB[con_name])
                    #input('---')
                lp_prob += expr >= dict_con_name_2_LB[con_name], con_name #+ "_ineq"

        # --- Add equality constraints ---
        # Group terms for each equality constraint.
        eq_expressions = {}
        for (var_name, con_name), coeff in dict_var_con_2_lhs_eq.items():
            eq_expressions.setdefault(con_name, 0)
            eq_expressions[con_name] += coeff * var_dict[var_name]
    
        # Add each equality constraint to the model.
        print('----')
        print('----')
        print('----')
        for con_name, expr in eq_expressions.items():
            if con_name in dict_con_name_2_eq:

                print('con_name')
                print(con_name)
                print('expr')
                print(expr)
                print('dict_con_name_2_eq[con_name]')
                print(dict_con_name_2_eq[con_name])
                #input('---')
                lp_prob += expr == dict_con_name_2_eq[con_name], con_name #+ "_eq"

        # --- Solve the LP ---
        # Using the default CBC solver here.
        solver = pulp.PULP_CBC_CMD(msg=True)
        lp_prob.solve(solver)
        self.lp_status=pulp.LpStatus[lp_prob.status]
        if self.lp_status=='Infeasible':
            input('ERROR infeasible')
        self.lp_prob=lp_prob
        self.lp_primal_solution=dict()
        print('outputting')
        for var_name, var in var_dict.items():
            self.lp_primal_solution[var_name]=var.varValue
            if var.varValue>.001:
                print('var_name')
                print(var_name)
                print('var.varValue')
                print(var.varValue)
            
        
        self.lp_objective= pulp.value(lp_prob.objective)
        self.lp_dual_solution=dict()
        for con_name, constraint in lp_prob.constraints.items():
            #if con_name not in dict_con_name_2_LB and con_name not in dict_con_name_2_eq:
            #    print(lp_prob.constraints.keys())
            #    print ('con_name')
            #    print(con_name)
            #    input('error here')
            self.lp_dual_solution[con_name]=constraint.pi
        print('self.lp_status')
        print(self.lp_status)
        print('self.lp_objective')
        print(self.lp_objective)
        input('---')

    def make_new_splits(self):

        #get max value
        self.NEW_node_2_agg_node=self.graph_node_2_agg_node.copy()
        #max_val=max(self.graph_node_2_agg_node.values())
        start_value=0
        extra_string='rand_'+str(random.randint(0,100000000))+'_'
        i_2_dual=dict()
        for i_orig in self.non_zero_i:
            i=i_orig[:]
            i= i.replace(" ", "_")
            con_name_1='con_i_slack_pos_'+i
            con_name_2='con_i_slack_neg_'+i
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
            i_2_dual[i_orig]=self.lp_dual_solution[con_name_1]-self.lp_dual_solution[con_name_2]
        for f in self.non_zero_f:
            start_value=start_value+1
            for i in self.agg_node_2_node[f]:
                if i in i_2_dual and i_2_dual[i]>self.MF.jy_opt['epsilon']:
                    self.NEW_node_2_agg_node[i]=extra_string+'_'+str(start_value)