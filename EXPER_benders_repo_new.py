import functools

import random
import re
from collections import defaultdict
from solve_gurobi_lp import solve_gurobi_lp_bounds
from solve_gurobi_lp import solve_gurobi_lp_bounds_benders_pareto
import numpy as np
import sys
from itertools import chain, combinations
import pickle
sys.path.append("pre_process")
import math
from naive_pre import *

from typing import Dict, Hashable, Tuple

MIN_LP_OBJECTIVE_CUT=0.1
COVER_EPSILON=0.0001
OFFSET_COST_CUT=0.0001
EPSILON_STANDARD=0.0001
EPSILON_RHS_SUB=0.000001
EPSILON_EDGE=0.00001
EPSILON_MULT_PARETO_OBJ=0.0001
USE_RAND=0
MAX_SIZE_CHOOSE_K=230
VAL_STOP_ADDING_CUTS=100000000
MY_SIZES_USE=[9]

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    


class benders_cut_generator:

    def __init__(self,MF,sub_prob_name,sub_prob_y_obj,A_ineq_x,A_ineq_y,A_eq_y,rhs_ineq,my_sub_prob):
        self.sub_prob_name=sub_prob_name
        self.MF=MF
        self.my_sub_prob=my_sub_prob
        self.A_ineq_x=A_ineq_x
        self.sub_prob_y_obj=sub_prob_y_obj
        self.A_ineq_y=A_ineq_y
        self.A_eq_y=A_eq_y
        self.rhs_ineq=rhs_ineq

        self.dict_var_name_2_obj=self.sub_prob_y_obj
        self.dict_var_con_2_lhs_exog=self.A_ineq_y
        self.dict_var_con_2_lhs_eq=self.A_eq_y

        #self.dict_var_con_2_lhs_exog_pareto=self.my_sub_prob.A_ineq_y_pareto
        #self.sub_prob_y_obj_Pareto=self.my_sub_prob.sub_prob_y_obj_Pareto
        #self.rhs_ineq_pareto=self.my_sub_prob.sub_prob_y_obj_Pareto

        self.dict_con_name_2_eq=dict()
        for var_con in self.A_eq_y:
            con=var_con[1]
            self.dict_con_name_2_eq[con]=0
        self.dict_var_name_2_LB=dict()
        self.dict_var_name_2_UB=dict()
        for var_name in sub_prob_y_obj:
            self.dict_var_name_2_LB[var_name]=0
            self.dict_var_name_2_UB[var_name]=np.inf
   

    def call_lp_pareto(self,x):
        #print('part 1 of cut')
        [primal_solution_ORIG,dual_solution_ORIG,lp_objective_ORIG,time_opt_ORIG]=self.call_lp(x)
        #print('part 2 of cut')

        if lp_objective_ORIG<MIN_LP_OBJECTIVE_CUT:
            return [primal_solution_ORIG,dual_solution_ORIG,lp_objective_ORIG,time_opt_ORIG]
        
        PARETO_dict_var_con_2_lhs_exog=self.dict_var_con_2_lhs_exog.copy()
        PARETO_dict_var_name_2_obj=self.dict_var_name_2_obj.copy()

        z_var='pareto_var'
        PARETO_dict_var_name_2_obj[z_var]=-(1-EPSILON_MULT_PARETO_OBJ)*lp_objective_ORIG
        for con_name in self.dict_con_name_2_LB:
            val_1=-self.dict_con_name_2_LB[con_name]
            my_tup=tuple([z_var,con_name])
            PARETO_dict_var_con_2_lhs_exog[my_tup]=val_1
        eta=dict()
        for my_term in x:
            eta[my_term]=x[my_term]*0 
        for var_name in x:
            if var_name.startswith('act'):
                if var_name not in self.MF.action_2_cost or (var_name in self.MF.delta_name_2_ub and self.MF.delta_name_2_ub[var_name]<EPSILON_STANDARD):
                    continue
                else:
                    rand_term=1
                    if USE_RAND>0.5:
                        rand_term=(0.5*(1+np.random.rand()))
                    pareto_term=EPSILON_EDGE*rand_term
                    eta[var_name]=pareto_term
                    if eta[var_name]<0:
                        input('error sign')
        PARETO_dict_con_name_2_LB=dict()

        for my_key in self.rhs_ineq:
            PARETO_dict_con_name_2_LB[my_key]=0

        for my_dual_var_x_pair in self.A_ineq_x:
            var_name=my_dual_var_x_pair[0]
            con_name=my_dual_var_x_pair[1]
            my_mult=-self.A_ineq_x[my_dual_var_x_pair]
            val=eta[var_name]
            PARETO_dict_con_name_2_LB[con_name]+=my_mult*val
        
        #print('PARETO_dict_con_name_2_LB')
        #p#rint(PARETO_dict_con_name_2_LB)
        #i#nput('---')
        self.dict_var_name_2_LB[z_var]=.9999
        self.dict_var_name_2_UB[z_var]=1.0001
        #print('part 3 of cut')

        out_solution=solve_gurobi_lp_bounds_benders_pareto(PARETO_dict_var_name_2_obj,
                    PARETO_dict_var_con_2_lhs_exog,
                    PARETO_dict_con_name_2_LB,
                    self.dict_var_con_2_lhs_eq,
                    #self.dict_con_name_2_eq,dict(),dict())
                    self.dict_con_name_2_eq,self.dict_var_name_2_LB,self.dict_var_name_2_UB)
        #print('part 4 of cut')

        dual_solution=out_solution['dual_solution']
        primal_solution=out_solution['primal_solution']
        lp_objective=out_solution['objective']
        time_opt=out_solution['time_opt']
        time_opt_tot=time_opt+time_opt_ORIG

        check_obj=0
        check_orig_backup=0
        for v in self.dict_var_name_2_obj:
            check_obj+=primal_solution[v]*self.dict_var_name_2_obj[v]
            check_orig_backup+=primal_solution_ORIG[v]*self.dict_var_name_2_obj[v]
        if abs(check_obj-lp_objective_ORIG)>lp_objective_ORIG*0.1:
            print('check_obj')
            print(check_obj)
            print('check_orig_backup')
            print(check_orig_backup)
            print('lp_objective_ORIG')
            print(lp_objective_ORIG)
            print('primal_solution[z_var]')
            print(primal_solution[z_var])
            print('PARETO_dict_var_name_2_obj[z_var]')
            print(PARETO_dict_var_name_2_obj[z_var])
            print('z_var')
            print(z_var)
            input('error')
        #else:
        #    print('passed')
        #    input('---')
        #print('done cut')
        return [primal_solution,dual_solution,check_obj,time_opt_tot]


    def call_lp(self,x):

        dict_con_name_2_LB=self.rhs_ineq.copy()
        for  my_dual_var_x_pair in self.A_ineq_x:
            var_name=my_dual_var_x_pair[0]
            con_name=my_dual_var_x_pair[1]
            my_mult=-self.A_ineq_x[my_dual_var_x_pair]
            val=x[var_name]#+self.MF.jy_opt['ParetoEps']*(np.random.rand()+0.05)
            dict_con_name_2_LB[con_name]+=my_mult*val
        for var in self.dict_var_name_2_obj:
            if var in self.my_sub_prob.var_2_internal_2_act:
                uv=self.my_sub_prob.var_2_internal_2_act[var]
                u=uv[0]
                v=uv[1]
                var_name='act_'+str(u)+'_'+str(v)
                if var_name not in self.MF.action_2_cost or (var_name in self.MF.delta_name_2_ub and self.MF.delta_name_2_ub[var_name]<EPSILON_STANDARD):
                    self.dict_var_name_2_UB[var]=0
                    #input('did find')
                    
        self.dict_con_name_2_LB=dict_con_name_2_LB
        #print('calling LP')
        out_solution=solve_gurobi_lp_bounds(self.dict_var_name_2_obj,
                    self.dict_var_con_2_lhs_exog,
                    self.dict_con_name_2_LB,
                    self.dict_var_con_2_lhs_eq,
                    self.dict_con_name_2_eq,dict(),dict())
                    #self.dict_con_name_2_eq,self.dict_var_name_2_LB,self.dict_var_name_2_UB)
        dual_solution=out_solution['dual_solution']
        primal_solution=out_solution['primal_solution']
        lp_objective=out_solution['objective']
        time_opt=out_solution['time_opt']
        return [primal_solution,dual_solution,lp_objective,time_opt]
    def generate_benders_cut(self,x,OPT_X_input=None):
        
        self.in_x=x
        self.OPT_X_input=OPT_X_input
        tot_cut_value=0
        #[primal_solution,dual_solution,lp_objective,time_opt]=self.call_lp(x)
        [primal_solution,dual_solution,lp_objective,time_opt]=self.call_lp_pareto(x)
        self.primal_solution=primal_solution
        self.dual_solution=dual_solution
        self.lp_objective=lp_objective
        did_add=False
        if lp_objective>MIN_LP_OBJECTIVE_CUT:
            did_add=True
            tot_cut_value=tot_cut_value+lp_objective
            dict_x_2_coeff=dict()
            for act in self.MF.all_non_null_action:
                dict_x_2_coeff[act]=0
            cut_RHS=0
            for my_dual_var in self.rhs_ineq:
                cut_RHS+=self.rhs_ineq[my_dual_var]*dual_solution[my_dual_var]
            for my_dual_var_x_pair in self.A_ineq_x:
                my_act=my_dual_var_x_pair[0]
                my_dual_var=my_dual_var_x_pair[1]
                my_mult=self.A_ineq_x[my_dual_var_x_pair]
                my_term=dual_solution[my_dual_var]*my_mult
                dict_x_2_coeff[my_act]=dict_x_2_coeff[my_act]+my_term
                
            
            new_cut_name='Benders_cut_new_'+self.sub_prob_name+'_'+str(len(self.MF.exog_name_2_rhs))+'_'+str(np.floor(np.random.rand()*10000))
            cut_RHS=cut_RHS-OFFSET_COST_CUT
            self.MF.all_exog.append(cut_RHS)
            self.MF.full_input_dict['allExogNames'].append(cut_RHS)
            self.MF.D['allExogNames'].append(new_cut_name)

            self.MF.exog_name_2_rhs[new_cut_name]=cut_RHS
            self.MF.D['exogName2Rhs'][new_cut_name]=cut_RHS
            self.MF.full_input_dict['exogName2Rhs'][new_cut_name]=cut_RHS

            for act in dict_x_2_coeff:
                my_tuple=tuple([act,new_cut_name])
                new_val=dict_x_2_coeff[act]
                if abs(new_val)>EPSILON_STANDARD:
                    self.MF.action_con_2_contrib[my_tuple]=new_val
                    self.MF.D['actionCon2Contrib'][my_tuple]=new_val
                    self.MF.full_input_dict['actionCon2Contrib'][my_tuple]=new_val
            my_LHS_frac=0
            for act in dict_x_2_coeff:
                term_add=(x[act]*dict_x_2_coeff[act])
                my_LHS_frac=my_LHS_frac+term_add
            if my_LHS_frac>cut_RHS:
                print('my_LHS_frac')
                print(my_LHS_frac)
                print('cut_RHS')
                print(cut_RHS)
                input('wrong no sense ')
            
        return lp_objective,did_add,time_opt

class benders_repo_new:

    def __init__(self,my_full_prob):
        #input('hi')
        
        self.MF=my_full_prob

        self.MF.jy_opt.update({
            "MIN_LP_OBJECTIVE_CUT":      MIN_LP_OBJECTIVE_CUT,
            "COVER_EPSILON":             COVER_EPSILON,
            "OFFSET_COST_CUT":           OFFSET_COST_CUT,
            "EPSILON_STANDARD":          EPSILON_STANDARD,
            "EPSILON_RHS_SUB":           EPSILON_RHS_SUB,
            "EPSILON_EDGE":              EPSILON_EDGE,
            "EPSILON_MULT_PARETO_OBJ":   EPSILON_MULT_PARETO_OBJ,
            "USE_RAND":                  USE_RAND,
            "MAX_SIZE_CHOOSE_K":         MAX_SIZE_CHOOSE_K,
            "VAL_STOP_ADDING_CUTS":      VAL_STOP_ADDING_CUTS,
            "MY_SIZES_USE":              MY_SIZES_USE,
        })
        if not hasattr(self.MF, "act_2_uv"):
            self.MF.act_2_uv = {}

            for act in self.MF.all_non_null_action:
                try:
                    _, u, v = act.split("_")
                    self.MF.act_2_uv[act] = [int(u), int(v)]
                except Exception as e:
                    raise ValueError(f"Not in 'act_u_v' format: {act!r}") from e
        self.size_neigh_use_benders=9
        L=self.size_neigh_use_benders
        self.m_sz_pairs=[]

        
        
        

        self.A_eq_y=dict()
        self.A_eq_x=dict()
        self.rhs_ineq=dict()

        print('Generating sub-problems')
        self.generate_sub_problems()
        print('Done Generating sub-problems')

    def generate_cuts(self,x_solution,OPT_X_input=None):
        
        self.x_solution=x_solution
        debug_on=False
        if debug_on==True:
            with open("PlayHere.pkl", "wb") as f:
                print('saving prior')
                pickle.dump(self, f)
                print('done saving prior')
        tot_cut_value=0
        TOT_gen_cut=0
        tot_time_opt=0
        max_time_opt=0

        for my_bend_prob in random.sample(self.my_list_benders_cut_generator,len(self.my_list_benders_cut_generator)):
 #           print('generating cut ')
            print('my_bend_prob.sub_prob_name')
            print(my_bend_prob.sub_prob_name)
#            print('----')
            [this_cut_value,did_gen_cut,this_time_opt]=my_bend_prob.generate_benders_cut(x_solution,OPT_X_input)
            if did_gen_cut==True:
                TOT_gen_cut=TOT_gen_cut+1
                tot_cut_value=tot_cut_value+this_cut_value
            tot_time_opt=tot_time_opt+this_time_opt
            max_time_opt=max([max_time_opt,this_time_opt])
            print('this_cut_value:  '+str(this_cut_value))
            print('tot_cut_value,TOT_gen_cut]:  '+str([tot_cut_value,TOT_gen_cut]))
            print('tot_time_opt.  '+str([this_time_opt,tot_time_opt]))
            if VAL_STOP_ADDING_CUTS<tot_cut_value:
                break
        print('[tot_cut_value,TOT_gen_cut]')
        print([tot_cut_value,TOT_gen_cut])

        return [tot_cut_value,TOT_gen_cut,tot_time_opt,max_time_opt]

    def generate_sub_problems(self):
        self.my_sub_prob=[]
        self.my_list_benders_cut_generator=[]
        my_VRP=self.MF.D['my_VRP']
        self.Nc=self.MF.my_VRP.num_cust
        all_sets=set()

        for this_sz in MY_SIZES_USE:
            [ng_neigh_by_cust_power,junk]=naive_get_LA_neigh(my_VRP,this_sz)
        #self.ng_neigh_by_cust_power=ng_neigh_by_cust_power
            for u in range(0,self.Nc):
                my_set=set(ng_neigh_by_cust_power[u]).union([u])
                if len(my_set)>2:
                    my_set=frozenset(my_set)
                    all_sets.add(my_set)
        counter=0
        for my_set in all_sets:

            my_sub_prob_input=sub_problem(self.MF,my_set)
            M=my_sub_prob_input
            self.my_sub_prob.append(my_sub_prob_input)

            new_cut_gen=benders_cut_generator(self.MF,M.sub_prob_name,M.var_2_cost,M.A_ineq_x,M.A_ineq_y,M.A_eq_y,M.rhs_ineq,my_sub_prob_input)
            self.my_list_benders_cut_generator.append(new_cut_gen)
            counter=counter+1
            print('my_set_cust:  '+str(my_set))
            print('counter:  '+str(counter))
            

class sub_problem:
    def __init__(self,MF,my_set_cust):

        
        #for kk in range(0,40):
        #    self.get_my_sz_pairs_by_size(kk)
            #print('kk')
            #print(kk)
            #print('self.m_sz_pairs')
            #print(self.m_sz_pairs)
            #input('---')



        self.MF=MF
        self.my_set_cust=my_set_cust
        self.get_my_sz_pairs_by_size(len(my_set_cust))

        
        self.my_ng_graph=Benders_NG_graph(my_set_cust,MF)
        self.ng_edge_source_2_cust=self.my_ng_graph.ng_edge_source_2_cust
        self.ng_edge_cust_2_sink=self.my_ng_graph.ng_edge_cust_2_sink
        self.ng_edges=self.my_ng_graph.ng_edges
        self.ng_nodes=self.my_ng_graph.ng_nodes
        self.ng_nodes_minus_source_sink=self.my_ng_graph.ng_nodes_minus_source_sink
        self.ng_source=self.my_ng_graph.source_node
        self.ng_sink=self.my_ng_graph.sink_node
        self.ng_edges_non_source_sink=self.my_ng_graph.ng_edges_non_source_sink
        self.get_divisor_subset_tuples()
        self.A_eq_y=dict()
        self.A_ineq_y=dict()
        self.sub_prob_name='my_sub_prob'+str(my_set_cust)
        self.A_ineq_x=dict()
        self.rhs_ineq=dict()

        self.make_vars_cost()
        self.make_cover_con()
        self.make_flow_in_out()
        self.make_matching_constrs_reg_edge()
        self.make_matching_constrs_source()
        self.make_matching_constrs_sink()

        self.make_valid_ineq_con()
        self.make_dual_lp_PRE()

    def make_dual_lp_PRE(self):

        self.DUAL_A_ineq=dict()

        self.DUAL_r_ineq=dict()
        self.DUAL_var_LB=dict()
        self.DUAL_var_B=dict()


    def get_my_sz_pairs_by_size(self, K):
        m_sz_pairs = []

        num_cust_use_pos = np.arange(2,K)#np.array([3])
        num_cust_use_neg = K-np.arange(0,K-1)#K - np.array([0, 1])

        num_cust_use = np.concatenate([num_cust_use_pos, num_cust_use_neg])
        num_cust_use = {k for k in num_cust_use if k > 2 and k <= K}
        #print('num_cust_use')
        #print(num_cust_use)
        for k in num_cust_use:
            this_sz=math.comb(int(K),int(k))
            if this_sz>MAX_SIZE_CHOOSE_K:
                #print('[k,K,MAX_SIZE_k_K,this_sz]')
                ##print([k,K,MAX_SIZE_k_K,this_sz])
                #input('look')
                continue
            for r in range(2, k + 1):
                m_sz_pairs.append((r, k))
        #print('m_sz_pairs')
        #print(m_sz_pairs)
        #print('---')
        X = list(m_sz_pairs)
        do_keep=[]

        for (r1,k1) in X:
            this_do_keep=True

            if k1%r1==0:
                this_do_keep=False

                continue
            for (r2,k2) in X:
                #dom 
                if r1==r2 and k2>k1 and (k1//r1)==(k2//r2):
                    this_do_keep=False
                    #print('killing 1')
                    #print(' r2,k2)')
                    #print((r2,k2))
                    #print('killing r1,k1)')
                    #print((r1,k1))
                    break
                if k1==k2 and r1>r2 and (k1//r1)==(k2//r2):
                    this_do_keep=False
                    #print('killing 2')
                    #print(' r2,k2)')
                    #print((r2,k2))
                    #print('killing r1,k1)')
                    #print((r1,k1))
                    break
                #if  k1==k2 and r2<r1 and r1%r2==0:
                #    t1=r1//k1
                #    t2=r2//k2
                 #   if 
            if this_do_keep==True:
                do_keep.append((r1,k1))
        do_keep_2=[]
        for (r1,k1) in do_keep:
            this_do_keep=True
            for (r2,k2) in do_keep:

                if k2==k1 and r1>r2 and r1%r2==0:
                    this_do_keep=False
            if this_do_keep==True:
                do_keep_2.append((r1,k1))
        #print('do_keep')
        #print(do_keep)
        #3print('do_keep_2')
        #p#rint(do_keep_2)
        
        self.m_sz_pairs = do_keep_2

    def get_divisor_subset_tuples(self):
        """
        N: iterable of items
        M: dict or list, where M[k] = [f, g] with g an int (subset size)
        Returns: dict k -> list of subsets of N (as tuples) of size g
        """
        N = list(self.my_set_cust)


        self.AllSubsetsBySize = {}
        self.subset_and_divisor=[]
        for k in self.m_sz_pairs:
            my_divisor=k[0]
            my_sz=k[1]
            if my_sz not in self.AllSubsetsBySize:
                self.AllSubsetsBySize[k] = [tuple(c) for c in combinations(N, my_sz)]
            
            for my_mini_set in self.AllSubsetsBySize[k]:
                this_tup=tuple([my_mini_set,my_divisor])
                self.subset_and_divisor.append(this_tup)

    def make_matching_constrs_source(self):
        for v in self.my_set_cust:
            con_name='Source_agree_match_'+str(v)
            var_name='slack_pos_cust_'+str(v)
            my_tup_source=tuple([var_name,con_name])
            self.rhs_ineq[con_name]=0#-self.MF.jy_opt['ParetoEps']
            if con_name=="Source_agree_match_frozenset()":
                input('err here 0')
            self.A_ineq_y[my_tup_source]=1

        for ng_edge in self.ng_edge_source_2_cust:
            i=ng_edge[0]
            j=ng_edge[1]
            v=j[0]
            var_name='ng_EDGE_'+str(ng_edge)
            con_name='Source_agree_match_'+str(v)
            if con_name=="Source_agree_match_frozenset()":
                print('my_set_cust')
                print(self.my_set_cust)
                input('err here 1')
            my_tup_edge=tuple([var_name,con_name])
            self.A_ineq_y[my_tup_edge]=-1
        for my_act in self.MF.act_2_uv:
            uv=self.MF.act_2_uv[my_act]
            u=uv[0]
            v=uv[1]
            if u not in self.my_set_cust and v in self.my_set_cust:
                con_name='Source_agree_match_'+str(v)
                var_name=my_act
                my_tup=tuple([var_name,con_name])
                self.A_ineq_x[my_tup]=1

    def make_matching_constrs_sink(self):
        for u in self.my_set_cust:
            con_name='Sink_agree_match_'+str(u)
            var_name='slack_neg_cust_'+str(u)
            my_tup_sink=tuple([var_name,con_name])
            self.rhs_ineq[con_name]=0#-self.MF.jy_opt['ParetoEps']
            self.A_ineq_y[my_tup_sink]=1

        for ng_edge in self.ng_edge_cust_2_sink:
            i=ng_edge[0]
            u=i[0]
            var_name='ng_EDGE_'+str(ng_edge)
            con_name='Sink_agree_match_'+str(u)

            my_tup_edge=tuple([var_name,con_name])
            self.A_ineq_y[my_tup_edge]=-1
        for my_act in self.MF.act_2_uv:
            uv=self.MF.act_2_uv[my_act]
            u=uv[0]
            v=uv[1]
            if u in self.my_set_cust and v not in self.my_set_cust:
                con_name='Sink_agree_match_'+str(u)
                var_name=my_act
                my_tup=tuple([var_name,con_name])
                self.A_ineq_x[my_tup]=1

    def make_matching_constrs_reg_edge(self):
        for my_act in self.MF.all_non_null_action:
            uv=self.MF.act_2_uv[my_act]
            
            u=uv[0]
            v=uv[1]
            
            if u in self.my_set_cust and v in self.my_set_cust:
                con_name='agree_match_'+str(u)+'_'+str(v)
                var_name='act_'+str(u)+'_'+str(v)
                if var_name not in self.MF.action_2_cost:
                    continue
                if var_name in self.MF.delta_name_2_ub and self.MF.delta_name_2_ub[var_name]<EPSILON_STANDARD:
                    continue
                my_tup=tuple([var_name,con_name])
                self.rhs_ineq[con_name]=0#-self.MF.jy_opt['ParetoEps']
                self.A_ineq_x[my_tup]=1
        for edge_tup in self.ng_edges_non_source_sink:
            var_name='ng_EDGE_'+str(edge_tup)

            i=edge_tup[0]
            j=edge_tup[1]
            u=i[0]
            v=j[0]
            ACT_NAME='act_'+str(u)+'_'+str(v)
            if ACT_NAME not in self.MF.action_2_cost:
                input('cant hapen')
            if ACT_NAME in self.MF.delta_name_2_ub and self.MF.delta_name_2_ub[ACT_NAME]<EPSILON_STANDARD:
                input('cant hapen')
            con_name='agree_match_'+str(u)+'_'+str(v)
            if con_name not in self.rhs_ineq:
                print('con_name err here')
                print(con_name)
                input('---')
            my_tup=tuple([var_name,con_name])

            self.A_ineq_y[my_tup]=-1

    def make_cover_con(self):
        #input('maing cover stuff')
        for u in self.my_set_cust:
            con_name='cover_con_'+str(u)
            self.rhs_ineq[con_name]=1-COVER_EPSILON
        for edge_tup in self.ng_edges:
            var_name='ng_EDGE_'+str(edge_tup)
            i=edge_tup[0]
            j=edge_tup[1]
            if i!=self.ng_source:
                u=i[0]
                con_name='cover_con_'+str(u)
                
                my_tup=tuple([var_name,con_name])
                self.A_ineq_y[my_tup]=1
    

    def make_valid_ineq_con(self):
        EPS = EPSILON_RHS_SUB  # or EPSILON_RHS_SUB if it's a module const

        # --- 1) Build a stable bit index for all customers (do once) ---
        if not hasattr(self, "_cust_bit_idx"):
            U = set()
            for subset, _div in self.subset_and_divisor:
                U.update(subset)        # works for tuple/list/set/frozenset

            for e in self.ng_edge_cust_2_sink:
                U.update(e[0][2])       # same here
            self._cust_bit_idx = {c: i for i, c in enumerate(U)}
        idx = self._cust_bit_idx

        @functools.lru_cache(maxsize=None)
        def to_mask_frozen(fz):
            m = 0
            for c in fz:
                m |= 1 << idx[c]
            return m

        def to_mask(S):
            # use frozenset for caching
            return to_mask_frozen(frozenset(S))

        # --- 2) Precompute masks & names for edges (once) ---
        edges = self.ng_edge_cust_2_sink
        edge_masks = [to_mask(e[0][2]) for e in edges]
        edge_names = [f"ng_EDGE_{e}" for e in edges]  # if this is slow/huge, use a compact name

        # Local aliases (fewer attribute lookups)
        A = self.A_ineq_y
        RHS = self.rhs_ineq

        # --- 3) Build constraints quickly ---
        for subset, divisor in self.subset_and_divisor:
            sub_mask = to_mask(subset)
            # NOTE: str(subset) can be very slow/long; prefer a compact name:
            con_name = f"my_SRI_d{divisor}_h{hash(frozenset(subset))}"
            # original name would be: con_name = 'my_SRI_' + str((subset, divisor))

            # RHS: -floor(len(subset)/divisor) - EPS
            RHS[con_name] = -(len(subset) // divisor) - EPS

            # Fill A_ineq_y only for non-zero entries
            for m_edge, var_name in zip(edge_masks, edge_names):
                inter_sz = (m_edge & sub_mask).bit_count()
                if inter_sz >= divisor:              # else floor=0 → skip
                    mult = -(inter_sz // divisor)    # -floor
                    if mult:                         # guard (redundant given >= divisor)
                        A[(var_name, con_name)] = mult
        
    def OLD_make_valid_ineq_con(self):
        for q in self.subset_and_divisor:

            my_subset=q[0]
            my_divisor=q[1]
            con_name='my_SRI_'+str(q)

            self.rhs_ineq[con_name]=-np.floor(len(my_subset)/my_divisor)-EPSILON_RHS_SUB

            for ng_edge in self.ng_edge_cust_2_sink:

                my_inter_sz=len(ng_edge[0][2].intersection(my_subset))
                my_mult=-np.floor(my_inter_sz/my_divisor)
                var_name='ng_EDGE_'+str(ng_edge)
                my_tup=tuple([var_name,con_name])
                self.A_ineq_y[my_tup]=my_mult

    def make_vars_cost(self):
        MF=self.MF
        self.var_2_cost=dict()

        if MF.null_action in MF.all_non_null_action:
            input('error here')

        for u in self.my_set_cust:
            var_pos_slack_act='slack_pos_cust_'+str(u)
            var_neg_slack_act='slack_neg_cust_'+str(u)
            self.var_2_cost[var_pos_slack_act]=1
            self.var_2_cost[var_neg_slack_act]=1

        self.var_2_internal_2_act=dict()
        for edge_tup in self.ng_edges:
            edge_name='ng_EDGE_'+str(edge_tup)
            self.var_2_cost[edge_name]=0
            if edge_tup in self.my_ng_graph.ng_internal_edge_2_act:
                self.var_2_internal_2_act[edge_name]=self.my_ng_graph.ng_internal_edge_2_act[edge_tup]
       
    def make_flow_in_out(self):
        for ng_node in self.ng_nodes_minus_source_sink:
            con_name='flow_in_out_'+str(ng_node)
        for ng_edge in self.ng_edges:
            i=ng_edge[0]
            j=ng_edge[1]
            if i !=self.ng_source:
                con_name='flow_in_out_'+str(i)
                var_name='ng_EDGE_'+str(ng_edge)
                my_tup=tuple([var_name,con_name])
                self.A_eq_y[my_tup]=-1
            if j !=self.ng_sink:
                con_name='flow_in_out_'+str(j)
                var_name='ng_EDGE_'+str(ng_edge)
                my_tup=tuple([var_name,con_name])
                self.A_eq_y[my_tup]=1

class   Benders_NG_graph:

    def __init__(self,my_set_cust,MF):
        self.my_set_cust=my_set_cust
        self.MF=MF
        self.make_all_nodes()
        self.make_feas_edges()
        #self.clean_feas_edges()
    def  make_all_nodes(self):
        Nc=self.MF.my_VRP.num_cust

        self.source_node=tuple([Nc,frozenset([]),frozenset([])])
        self.sink_node=tuple([Nc+1,frozenset([]),frozenset([])])

    
        my_nodes=[]
        my_nodes_by_sz=dict()


        for k in range(0,len(self.my_set_cust)):
            my_nodes_by_sz[k]=[]
        for u in self.my_set_cust:
            rest = [x for x in self.my_set_cust if x != u]
            for j in powerset(rest):
                s1=frozenset(sorted(list(j)))
               
                s2=list(s1)+[u]
                s2=frozenset(sorted(list(s2)))

                new_node=tuple([u,s1,s2])
                node_dem=0
                for w in s2:
                    node_dem+=self.MF.my_VRP.dem[w]
                if node_dem<=self.MF.my_VRP.vehicle_capacity:
                    my_nodes.append(new_node)
                    my_nodes_by_sz[len(s1)].append(new_node)

        self.my_nodes_by_sz=my_nodes_by_sz
        self.my_nodes=my_nodes
    def make_feas_edges(self):
        self.my_orderings_by_node=dict()
        my_nodes_by_sz=self.my_nodes_by_sz
        my_edges=set()

        self.ng_edge_source_2_cust=[]
        self.ng_edges=[]
        self.ng_internal_edge_2_act=dict()
        self.ng_edges_non_source_sink=[]
        self.early_depart_time_by_node=dict()
        self.edges_removed=[]
        for i  in self.my_nodes_by_sz[0]:
            e=tuple([self.source_node,i])
            my_edges.add(e)
            self.ng_edge_source_2_cust.append(e)
            self.ng_edges.append(e)
            u=i[0]
            self.early_depart_time_by_node[i]=self.MF.my_VRP.early_start[u]
        for k in range(1,len(self.my_set_cust)):
            for i in self.my_nodes_by_sz[k]:

                visited_excluding_last=i[1]
                self.early_depart_time_by_node[i]=-np.inf
                u=i[0]
                for w in visited_excluding_last:
                    var_name='act_'+str(w)+'_'+str(u)
                    if var_name not in self.MF.action_2_cost:
                        continue
                    if var_name in self.MF.delta_name_2_ub and self.MF.delta_name_2_ub[var_name]<EPSILON_STANDARD:
                        #input('found one NOT AN ERROR ')
                        continue
                    tmp=set(visited_excluding_last)-set([w])
                    tmp_1=frozenset(sorted(list(tmp)))
                    tmp_2=visited_excluding_last.copy()
                    tmp_2=frozenset(sorted(list(tmp_2)))
                    node_pred=tuple([w,tmp_1,tmp_2])
                    if node_pred not in self.my_nodes:#[node_pred]:
                        continue
                    early_w=self.compute_depart_time(i[0],w,self.early_depart_time_by_node[node_pred])
                    self.early_depart_time_by_node[i]=max([self.early_depart_time_by_node[i],early_w])
                    if early_w>=0:
                        e=tuple([node_pred,i])
                        self.ng_edges.append(e)
                        self.ng_edges_non_source_sink.append(e)
                        self.ng_internal_edge_2_act[e]=tuple([w,u])
                    else:
                        eREM=tuple([node_pred,i])
                        self.edges_removed.append(eREM)
        self.ng_nodes=[]
        self.ng_nodes_minus_source_sink=[]
        self.ng_nodes.append(self.source_node)
        self.ng_nodes.append(self.sink_node)
        self.ng_edge_cust_2_sink=[]
        for my_node in self.my_nodes:
            if self.early_depart_time_by_node[my_node]>=0:
                self.ng_nodes.append(my_node)
                self.ng_nodes_minus_source_sink.append(my_node)
                e=tuple([my_node,self.sink_node])
                self.ng_edges.append(e)
                self.ng_edge_cust_2_sink.append(e)
        
    def compute_depart_time(self,v,u,time_dep_u):
        if time_dep_u<0:
            out=-np.inf
            return out
        out=time_dep_u-self.MF.my_VRP.dist_mat_full[u,v]
        
        out=min(out,self.MF.my_VRP.early_start[v])
        act='act_'+str(u)+'_'+str(v)
        if out <self.MF.my_VRP.late_start[v]:
            out=-np.inf
        if act in self.MF.delta_name_2_ub and self.MF.delta_name_2_ub[act]<EPSILON_STANDARD:
            out=-np.inf
            
        return out

    