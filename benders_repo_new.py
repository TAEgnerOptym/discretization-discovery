import random
import math
import re
from collections import defaultdict
from solve_gurobi_lp import solve_gurobi_lp_bounds
import numpy as np
import sys
from itertools import chain, combinations
import pickle
sys.path.append("pre_process")
import time
from naive_pre import *
EPSILON=0.0001
COVER_CON_EPSILON=0#0.00001
EPSILON_VALID_INEQ=0#0.000001
EPSILON_MAX_VAL_CON=0.01
EPSILON_MULT=0.01
from typing import Dict, Hashable, Tuple
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    

def to_nested(A: Dict[Tuple[Hashable, Hashable], float]) -> Dict[Hashable, Dict[Hashable, float]]:
    """
    Convert flat dict A[(var_name, con_name)] -> coeff
    into nested dict A_fast[con_name][var_name] -> coeff.
    """
    A_fast: Dict[Hashable, Dict[Hashable, float]] = {}
    for (var, con), val in A.items():
        if val == 0:               # optional: skip stored zeros
            continue
        A_fast.setdefault(con, {})[var] = val
    return A_fast

# --- Step 1 usage ---
# Inputs: flat dicts keyed by (var_name, con_name)
# A_ineq_x[(uv, q)], A_ineq_y[(z, q)], A_eq_y[(z, q)]

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
        #self.A_fast_ineq_x = to_nested(self.A_ineq_x)
        #self.A_fast_ineq_y = to_nested(self.A_ineq_y)
        #self.A_fast_eq_y   = to_nested(self.A_eq_y)

        self.dict_var_name_2_obj=self.sub_prob_y_obj
        self.dict_var_con_2_lhs_exog=self.A_ineq_y
        self.dict_var_con_2_lhs_eq=self.A_eq_y
        self.dict_con_name_2_eq=dict()
        for var_con in self.A_eq_y:
            con=var_con[1]
            self.dict_con_name_2_eq[con]=0
        self.dict_var_name_2_LB=dict()
        self.dict_var_name_2_UB=dict()
        for var_name in sub_prob_y_obj:
            self.dict_var_name_2_LB[var_name]=0
            self.dict_var_name_2_UB[var_name]=np.inf
        #step 1
        #take in A_ineq_x which has input form A[tuple(var_name,con_name)] and make a dictionary of dictionarys that has
            #A_fast_ineq_x is a dict with elements A_fast_ineq_x[con_name] which is a dict with elements A_fast_ineq_x[con_name][var_name]
        
        #do that for A_ineq_y,A_eq_y as well.  use a funciton so that we do this onc.e. 
        #step 2
        #take in A_ineq_x which has input form A[tuple(var_name,con_name)] and make a dictionary of dictionarys that has
            #A_fast is a dict with elements A_fast[con_name] which is a dict with elements A_fast[con_name][var_name]


    def call_lp(self,x):
        phase1_time=time.time()
        dict_con_name_2_LB=self.rhs_ineq.copy()
        tot_add_by_con_name_no_eps=defaultdict(float)
        tot_add_by_con_name=defaultdict(float)
        tot_add_by_con_name_only_eps=defaultdict(float)
        for  my_dual_var_x_pair in self.A_ineq_x:
            var_name=my_dual_var_x_pair[0]
            if var_name in self.MF.delta_name_2_ub:# and self.MF.delta_name_2_ub[var_name]<EPSILON_STANDARD):
                #if self.MF.delta_name_2_ub[var_name]!=0:
                #    input('error here')
                continue
            con_name=my_dual_var_x_pair[1]
            my_mult=-self.A_ineq_x[my_dual_var_x_pair]
            rand_val=EPSILON_MULT
            if self.MF.jy_opt['BEND_USE_RAND']>0.5:
                rand_val=EPSILON_MULT*(np.random.rand+0.1)
            val=x[var_name]+rand_val#self.MF.jy_opt[]
            val_no_eps=x[var_name]
            
            tot_add_by_con_name[con_name]+=my_mult*val
            tot_add_by_con_name_no_eps[con_name]+=my_mult*val_no_eps
            tot_add_by_con_name_only_eps[con_name]+=my_mult*rand_val
        for con_name in tot_add_by_con_name:
            #dict_con_name_2_LB[con_name]-=EPSILON_MAX_VAL_CON#max([-0.01,tot_add_by_con_name[con_name]])
            dict_con_name_2_LB[con_name]+=tot_add_by_con_name_no_eps[con_name]#tot_add_by_con_name[con_name]
            dict_con_name_2_LB[con_name]-=self.MF.jy_opt['NO_PARETO_EPSILON_MAX_VAL_CON']#max([-0.01,tot_add_by_con_name[con_name]])
            #dict_con_name_2_LB[con_name]+=tot_add_by_con_name_only_eps[con_name]#tot_add_by_con_name[con_name]
            #print('con_name')
            #print(con_name)
            #print('tot_add_by_con_name[con_name]')
            #print(tot_add_by_con_name[con_name])
            #print('self.rhs_ineq[con_name]')
            #print(self.rhs_ineq[con_name])
            #print('dict_con_name_2_LB[con_name]')
            #print(dict_con_name_2_LB[con_name])
            #input('---')
        for var in self.dict_var_name_2_obj:
            if var in self.my_sub_prob.var_2_internal_2_act:
                uv=self.my_sub_prob.var_2_internal_2_act[var]
                u=uv[0]
                v=uv[1]
                var_name='act_'+str(u)+'_'+str(v)
                if var_name not in self.MF.action_2_cost or (var_name in self.MF.delta_name_2_ub):
                    self.dict_var_name_2_UB[var]=0
                   
        
        self.dict_con_name_2_LB=dict_con_name_2_LB
        #print('calling LP')
        phase1_time=time.time()-phase1_time
        out_solution=solve_gurobi_lp_bounds(self.dict_var_name_2_obj,
                    self.dict_var_con_2_lhs_exog,
                    self.dict_con_name_2_LB,
                    self.dict_var_con_2_lhs_eq,
                    self.dict_con_name_2_eq,self.dict_var_name_2_LB,self.dict_var_name_2_UB)
        dual_solution=out_solution['dual_solution']
        primal_solution=out_solution['primal_solution']
        lp_objective=out_solution['objective']
        time_opt=out_solution['time_opt']
        print('phase 1 time = '+str(phase1_time))
        return [primal_solution,dual_solution,lp_objective,time_opt]
    def generate_benders_cut(self,x,OPT_X_input=None):
        print('calling LP maker')
        self.in_x=x
        self.OPT_X_input=OPT_X_input
        tot_cut_value=0
        [primal_solution,dual_solution,lp_objective,time_opt]=self.call_lp(x)
        print('Done calling LP maker')

        self.primal_solution=primal_solution
        self.dual_solution=dual_solution
        self.lp_objective=lp_objective
        did_add=False
        if lp_objective>self.MF.jy_opt['BEND_MIN_LP_OBJECTIVE_CUT']:
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
            cut_RHS=cut_RHS-self.MF.jy_opt['NO_PARETO_NUMERICAL_BEND_OFFSET_COST_CUT']
            self.MF.all_exog.append(cut_RHS)
            self.MF.full_input_dict['allExogNames'].append(cut_RHS)
            self.MF.D['allExogNames'].append(new_cut_name)

            self.MF.exog_name_2_rhs[new_cut_name]=cut_RHS
            self.MF.D['exogName2Rhs'][new_cut_name]=cut_RHS
            self.MF.full_input_dict['exogName2Rhs'][new_cut_name]=cut_RHS

            for act in dict_x_2_coeff:
                my_tuple=tuple([act,new_cut_name])
                new_val=dict_x_2_coeff[act]
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
            if OPT_X_input!=None:
                my_LHS=0
                my_LHS_frac=0
                for act in dict_x_2_coeff:
                    my_LHS=my_LHS+(OPT_X_input[act]*dict_x_2_coeff[act])
                    my_LHS_frac=my_LHS_frac+(x[act]*dict_x_2_coeff[act])
                if cut_RHS<my_LHS_frac:
                    input('ok no sense here')
                if cut_RHS>my_LHS:
                    print('non_neg dict2 ')
                    for d in self.dict_con_name_2_LB:
                        if abs(self.dict_con_name_2_LB[d])>0:
                            print([d,str(self.dict_con_name_2_LB[d])])
                    
                    print('self.my_sub_prob.my_ng_graph.DEBUG_my_candid_edges')
                    print(self.my_sub_prob.my_ng_graph.DEBUG_my_candid_edges)
                    print('self.my_sub_prob.my_ng_graph.is_special')
                    print(self.my_sub_prob.my_ng_graph.is_special)

                    #
                    with open("NEWINTERPlayHere.pkl", "wb") as f:
                        pickle.dump(self, f)
                    input('error here')
        print('Done call benders cut LP maker')

        return lp_objective,did_add,time_opt


        

class benders_repo_new:

    def __init__(self,my_full_prob,OPT_X_SOL=None):
        #input('hi')
        self.MF=my_full_prob

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
        #self.m_sz_pairs=[]
        #self.m_sz_pairs.append(tuple([6,10]))
        #self.m_sz_pairs.append(tuple([6,9]))
        #self.m_sz_pairs.append(tuple([2,3]))
        ##self.m_sz_pairs.append(tuple([2,5]))
        #self.m_sz_pairs.append(tuple([2,9]))
        #self.m_sz_pairs.append(tuple([3,10]))
        #self.m_sz_pairs.append(tuple([4,10]))
        self.A_eq_y=dict()
        self.A_eq_x=dict()
        self.rhs_ineq=dict()

        print('Generating sub-problems')
        self.generate_sub_problems(OPT_X_SOL)
        print('Done Generating sub-problems')


    def generate_cuts(self,x_solution,OPT_X_input=None):
        
        tot_cut_value=0
        TOT_gen_cut=0
        tot_time_opt=0
        max_time_opt=0
        counter=0
        num_sub_probs=len(self.my_list_benders_cut_generator)
        for my_bend_prob in random.sample(self.my_list_benders_cut_generator,len(self.my_list_benders_cut_generator)):
            print('----')
            print('generating cut counter = '+str(counter)+'. out of '+str(num_sub_probs))
            print('my_bend_prob.sub_prob_name = '+str(my_bend_prob.sub_prob_name))
            counter=counter+1
            [this_cut_value,did_gen_cut,this_time_opt]=my_bend_prob.generate_benders_cut(x_solution,OPT_X_input)
            if did_gen_cut==True:
                TOT_gen_cut=TOT_gen_cut+1
                tot_cut_value=tot_cut_value+this_cut_value
            tot_time_opt=tot_time_opt+this_time_opt
            max_time_opt=max([max_time_opt,this_time_opt])
            print('this_cut_value:  '+str(this_cut_value))
            print('tot_cut_value,TOT_gen_cut]:  '+str([tot_cut_value,TOT_gen_cut]))
            print('tot_time_opt,tot_time_opt,max_time_opt.  '+str([this_time_opt,tot_time_opt,max_time_opt]))
            if tot_cut_value>self.MF.jy_opt['NO_PARETO_BEND_VAL_STOP_ADDING_CUTS']:
                break
        #print('self.my_list_benders_cut_generator')
        #print(self.my_list_benders_cut_generator)
        #input('--')
        return [tot_cut_value,TOT_gen_cut,tot_time_opt,max_time_opt]

    def generate_sub_problems(self,OPT_X_SOL):
        self.my_sub_prob=[]
        self.my_list_benders_cut_generator=[]
        my_VRP=self.MF.D['my_VRP']
        self.Nc=self.MF.my_VRP.num_cust

        [ng_neigh_by_cust_power,junk]=naive_get_LA_neigh(my_VRP,self.size_neigh_use_benders)
        self.ng_neigh_by_cust_power=ng_neigh_by_cust_power
        all_sets=set()
        for u in range(0,self.Nc):
            my_set=set(ng_neigh_by_cust_power[u]).union([u])
            my_set=frozenset(my_set)
            all_sets.add(my_set)
        counter=0
        num_sets=len(all_sets)
        for my_set in all_sets:

            my_sub_prob_input=sub_problem(self.MF,my_set,)
            M=my_sub_prob_input
            self.my_sub_prob.append(my_sub_prob_input)

            new_cut_gen=benders_cut_generator(self.MF,M.sub_prob_name,M.var_2_cost,M.A_ineq_x,M.A_ineq_y,M.A_eq_y,M.rhs_ineq,my_sub_prob_input)
            self.my_list_benders_cut_generator.append(new_cut_gen)
            counter=counter+1
            print('my_set_cust:  '+str(my_set))
            print('counter:  '+str(counter)+ " out of  "+str(num_sets))
            if OPT_X_SOL!=None:
                
                [lp_objective,did_add]=new_cut_gen.generate_benders_cut(OPT_X_SOL)
                counter=counter+1
                if did_add==False or lp_objective>EPSILON:
                    print('lp_objective')
                    print(lp_objective)
                    print('did_add')
                    print(did_add)
                    print('--my primal solution -')
                    for p in new_cut_gen.primal_solution:
                        print([p+'   '+str(new_cut_gen.primal_solution[p])])
                    print('new_cut_gen.lp_objective')
                    print(new_cut_gen.lp_objective)
                    input('CUT HAS AN ISSUE FAILS HERE on cut')

class sub_problem:
    def __init__(self,MF,my_set_cust):
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
        #print('p1')
        self.make_vars_cost()
        #print('p2')
        self.make_cover_con()
        #print('p3')
        #self.make_flow_in_out()
        self.fast_make_flow_in_out()
        #print('p4')
        self.make_matching_constrs_reg_edge()
       # print('p5')
        self.make_matching_constrs_source()
        #print('p6')
        self.make_matching_constrs_sink()
        #print('p7')

        #self.make_valid_ineq_con()
        self.COPIED_make_valid_ineq_con()
        #print('p8')
    
    def get_my_sz_pairs_by_size(self, K):
        m_sz_pairs = []

        num_cust_use_pos = np.arange(2,K)#np.array([3])
        num_cust_use_neg = K-np.arange(0,K-1)#K - np.array([0, 1])

        num_cust_use = np.concatenate([num_cust_use_pos, num_cust_use_neg])
        num_cust_use = {k for k in num_cust_use if k > 2 and k <= K}
        for k in num_cust_use:
            this_sz=math.comb(int(K),int(k))
            if this_sz>self.MF.jy_opt['BEND_MAX_SIZE_CHOOSE_K']:
                continue
            for r in range(2, k + 1):
                m_sz_pairs.append((r, k))
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
                    break
                if k1==k2 and r1>r2 and (k1//r1)==(k2//r2):
                    this_do_keep=False
                    break
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
            self.rhs_ineq[con_name]=-EPSILON
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
            self.rhs_ineq[con_name]=-EPSILON
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
                if var_name in self.MF.delta_name_2_ub:
                    continue
                my_tup=tuple([var_name,con_name])
                self.rhs_ineq[con_name]=-EPSILON
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
            if ACT_NAME in self.MF.delta_name_2_ub and self.MF.delta_name_2_ub[ACT_NAME]<EPSILON:
                input('cant hapen')
            con_name='agree_match_'+str(u)+'_'+str(v)
            if con_name not in self.rhs_ineq:

                print('con_name err here')
                print(con_name)
                input('---')
            
            my_tup=tuple([var_name,con_name])

            self.A_ineq_y[my_tup]=-1

    def make_cover_con(self):
        for u in self.my_set_cust:
            con_name='cover_con_'+str(u)
            self.rhs_ineq[con_name]=1-COVER_CON_EPSILON#.9999
        for edge_tup in self.ng_edges:
            var_name='ng_EDGE_'+str(edge_tup)
            i=edge_tup[0]
            j=edge_tup[1]
            if i!=self.ng_source:
                u=i[0]
                con_name='cover_con_'+str(u)
                
                my_tup=tuple([var_name,con_name])
                self.A_ineq_y[my_tup]=1


    
    def COPIED_make_valid_ineq_con(self):
        debug_on=True
        ng_edge_2_str=dict()
        for ng_edge in self.ng_edge_cust_2_sink:
            ng_edge_2_str[ng_edge]='ng_EDGE_'+str(ng_edge)


        unique_divisors = {d for (_, d) in self.subset_and_divisor}
        unique_cust_sets = {s for (s,_) in self.subset_and_divisor}
        num_cust_in_subset=len(self.my_set_cust)#int(max(unique_divisors))
        dict_div_intersz_2_val=dict()

        for my_divisor in unique_divisors:
            dict_div_intersz_2_val[my_divisor]=dict()
            for my_input_size in range(0,1+num_cust_in_subset):#range(0,unique_seconds+1):
                dict_div_intersz_2_val[my_divisor][my_input_size]=-(my_input_size//my_divisor)

        my_inter_size=dict()
        if 1<0:
            for S in unique_cust_sets:
                my_inter_size[S]=defaultdict(float)
                for ng_edge in self.ng_edge_cust_2_sink:
                    inter_sz=len(ng_edge[0][2].intersection(S))
                    if inter_sz>0:
                        my_inter_size[S][ng_edge]=inter_sz
        else:
            _edges = list(self.ng_edge_cust_2_sink)
            _edge_sets = [es if isinstance((es := e[0][2]), set) else set(es) for e in _edges]

            my_inter_size = {}

            for S in unique_cust_sets:
                S_set = S if isinstance(S, set) else set(S)
                # build only nonzero intersections for this S
                d = {e: sz for e, es in zip(_edges, _edge_sets) if (sz := len(es & S_set)) > 0}
                if d:  # keep only non-empty entries (matches your original behavior)
                    my_inter_size[S] = defaultdict(float, d)
        
        #print('ap5')
        
        for q in self.subset_and_divisor:
            my_subset=q[0]
            my_divisor=q[1]
            S=my_subset
            D=my_divisor
            con_name='my_SRI_'+str(q)

            did_find=False
            tot_found=0
            num_found=0
            if 1<0:
                for ng_edge in self.ng_edge_cust_2_sink:

                    this_inter_sz=my_inter_size[S][ng_edge]
                    my_mult=dict_div_intersz_2_val[D][this_inter_sz]
                    if abs(my_mult)>0:
                        var_name=ng_edge_2_str[ng_edge]
                        my_tup=tuple([var_name,con_name])
                        self.A_ineq_y[my_tup]=my_mult
                        did_find=True
                        num_found=num_found+1
                        tot_found=tot_found+abs(my_mult)
            else:
                edge_inter = my_inter_size.get(S, {})                 # {ng_edge: inter_sz} for this S
                val_by_sz  = dict_div_intersz_2_val[D]                # {inter_sz: multiplier}
                to_name    = ng_edge_2_str                            # {ng_edge: "ng_EDGE_<...>"}

                # Build all nonzero coefficients in one pass
                edge_terms = {
                    (to_name[e], con_name): m
                    for e, sz in edge_inter.items()
                    if (m := val_by_sz.get(sz, 0)) != 0
                }

                # Bulk write
                self.A_ineq_y.update(edge_terms)

                # Fast counters
                did_find  = bool(edge_terms)
                num_found = len(edge_terms)
                tot_found = sum(abs(v) for v in edge_terms.values())
            if did_find==True:
                self.rhs_ineq[con_name]=-(len(my_subset)//my_divisor)

                if 1<0 and abs(tot_found)+0.3<=abs(self.rhs_ineq[con_name]):
                    print('my_subset')
                    print(my_subset)
                    print('my_divisor')
                    print(my_divisor)
                    print('tot_found')
                    print(tot_found)
                    print('self.rhs_ineq')
                    print(self.rhs_ineq)
                    input('not wrong but how common is this ')
    def make_valid_ineq_con(self):
        for q in self.subset_and_divisor:

            my_subset=q[0]
            my_divisor=q[1]
            con_name='my_SRI_'+str(q)

            self.rhs_ineq[con_name]=-np.floor(len(my_subset)/my_divisor)-EPSILON_VALID_INEQ

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

    def fast_make_flow_in_out(self):
        ns, nk = self.ng_source, self.ng_sink

        # Precompute names
        con_name = {n: f"flow_in_out_{n}" for n in self.ng_nodes_minus_source_sink}
        edge_var = {e: f"ng_EDGE_{e}" for e in self.ng_edges}

        # Build all "in" and "out" coefficients via comprehensions
        in_terms = {
            (edge_var[e], con_name[e[0]]): -1
            for e in self.ng_edges
            if e[0] != ns and e[0] in con_name
        }
        out_terms = {
            (edge_var[e], con_name[e[1]]): 1
            for e in self.ng_edges
            if e[1] != nk and e[1] in con_name
        }

        # Single bulk update
        self.A_eq_y.update(in_terms)
        self.A_eq_y.update(out_terms)

class   Benders_NG_graph:

    def __init__(self,my_set_cust,MF):
        self.my_set_cust=my_set_cust
        self.MF=MF
        self.make_all_nodes()
        self.make_feas_edges()
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
                    if var_name in self.MF.delta_name_2_ub and self.MF.delta_name_2_ub[var_name]<EPSILON:
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
                   
        

        self.ng_nodes=[]
        self.ng_nodes_minus_source_sink=[]
        self.ng_nodes.append(self.source_node)
        self.ng_nodes.append(self.sink_node)
        self.ng_edge_cust_2_sink=[]
        #print('startign infeas')
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
        if act in self.MF.delta_name_2_ub and self.MF.delta_name_2_ub[act]<EPSILON:
            out=-np.inf
            
        return out

    