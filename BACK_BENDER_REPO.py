import re
from collections import defaultdict
from solve_gurobi_lp import solve_gurobi_lp_bounds
import numpy as np
import sys
from itertools import chain, combinations
import pickle
sys.path.append("pre_process")

from naive_pre import *
EPSILON=0.00001

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

        dict_con_name_2_LB=self.rhs_ineq.copy()
        for  my_dual_var_x_pair in self.A_ineq_x:
            var_name=my_dual_var_x_pair[0]
            con_name=my_dual_var_x_pair[1]
            my_mult=-self.A_ineq_x[my_dual_var_x_pair]
            
            val=x[var_name]+EPSILON#self.MF.jy_opt[]
            if self.MF.jy_opt['BEND_USE_RAND']>0.5:
                val=x[var_name]+EPSILON*(np.random.rand+0.1)
            dict_con_name_2_LB[con_name]+=my_mult*val
        for var in self.dict_var_name_2_obj:
            if var in self.my_sub_prob.var_2_internal_2_act:
                uv=self.my_sub_prob.var_2_internal_2_act[var]
                u=uv[0]
                v=uv[1]
                var_name='act_'+str(u)+'_'+str(v)
                if var_name not in self.MF.action_2_cost or (var_name in self.MF.delta_name_2_ub and self.MF.delta_name_2_ub[var_name]<0.001):
                    self.dict_var_name_2_UB[var]=0
                    #print('removing var')
                    #print('var')
                    #print(var)
                    #print('var_name')
                    #print(var_name)
                    #input('NOT AN ERROR JUST CHECKING found one')
                #else:
                #    input('NOT AN ERROR JUST CHECKING  OK identified but ok')
            #else:
            #    if False==var.startswith('slack'):
            #        print('--')
            #        print('self.my_sub_prob.var_2_internal_2_act')
             #       print(self.my_sub_prob.var_2_internal_2_act)
             #       print('var')
             #       print(var)
             #       input('hihi')
        #print('dict_con_name_2_LB')
        #print(dict_con_name_2_LB)
        #print('non_neg dict')
        #for d in dict_con_name_2_LB:
        #    if abs(dict_con_name_2_LB[d])>0.01:
        #        print([d,str(dict_con_name_2_LB[d])])
        #print('call LP')
        
        self.dict_con_name_2_LB=dict_con_name_2_LB
        #print('calling LP')
        out_solution=solve_gurobi_lp_bounds(self.dict_var_name_2_obj,
                    self.dict_var_con_2_lhs_exog,
                    self.dict_con_name_2_LB,
                    self.dict_var_con_2_lhs_eq,
                    self.dict_con_name_2_eq,self.dict_var_name_2_LB,self.dict_var_name_2_UB)
        dual_solution=out_solution['dual_solution']
        primal_solution=out_solution['primal_solution']
        lp_objective=out_solution['objective']
        time_opt=out_solution['time_opt']
        #print('primal_solution')
        #print(primal_solution)
        #print('dict_var_name_2_obj')
        #print(self.dict_var_name_2_obj)
        #print('dual_solution')
        #print(dual_solution)
        #print('lp_objective')
        #print(lp_objective)
        #print('done clalling LP')
        return [primal_solution,dual_solution,lp_objective,time_opt]
    def generate_benders_cut(self,x,OPT_X_input=None):
        #print('generate benders cut')
        self.in_x=x
        self.OPT_X_input=OPT_X_input
        tot_cut_value=0
        [primal_solution,dual_solution,lp_objective,time_opt]=self.call_lp(x)
        self.primal_solution=primal_solution
        self.dual_solution=dual_solution
        self.lp_objective=lp_objective
        did_add=False
        if lp_objective>0.01:
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
                #uv=self.MF.act_2_uv[my_act]
                #u=uv[0]
                #v=uv[1]
                #print('my_dual_var_x_pair')
                #print(my_dual_var_x_pair)
                #con_name='agree_match_'+str(u)+'_'+str(v)
                my_mult=self.A_ineq_x[my_dual_var_x_pair]
                my_term=dual_solution[my_dual_var]*my_mult
                dict_x_2_coeff[my_act]=dict_x_2_coeff[my_act]+my_term
                #if abs(my_term)>0.00001:
                #    print('my_mult')
                #    print(my_mult)
                #    print('my_term')
                #    print(my_term)
                #    print('dual_solution[my_dual_var]')
                #    print(dual_solution[my_dual_var])
                #    print('my_dual_var')
                #    print(my_dual_var)

                    #input('looking')
            #print("x act (nonzero only)", {k:v for k,v in x.items() if v != 0 and k.startswith('act')})
            #print("x (nonzero only)", {k:v for k,v in x.items() if v != 0})
            #print("my primal (nonzero only)", {k:v for k,v in primal_solution.items() if v != 0})
            #print("my dict_x_2_coeff (nonzero only)", {k:v for k,v in dict_x_2_coeff.items() if v != 0})
            ##print("dual_solution (nonzero only)", {k:v for k,v in dual_solution.items() if v != 0})
            #print('cut_RHS:   '+str(cut_RHS))
            #print("--")
            
            new_cut_name='Benders_cut_new_'+self.sub_prob_name+'_'+str(len(self.MF.exog_name_2_rhs))+'_'+str(np.floor(np.random.rand()*10000))
            cut_RHS=cut_RHS-.0001
            #print('len(self.MF.exog_name_2_rhs)')
            #print(len(self.MF.exog_name_2_rhs))
            self.MF.all_exog.append(cut_RHS)
            self.MF.full_input_dict['allExogNames'].append(cut_RHS)
            self.MF.D['allExogNames'].append(new_cut_name)

            self.MF.exog_name_2_rhs[new_cut_name]=cut_RHS
            self.MF.D['exogName2Rhs'][new_cut_name]=cut_RHS
            self.MF.full_input_dict['exogName2Rhs'][new_cut_name]=cut_RHS

            for act in dict_x_2_coeff:
                my_tuple=tuple([act,new_cut_name])
                new_val=dict_x_2_coeff[act]
                if abs(new_val)>0.00001:
                    self.MF.action_con_2_contrib[my_tuple]=new_val
                    self.MF.D['actionCon2Contrib'][my_tuple]=new_val
                    self.MF.full_input_dict['actionCon2Contrib'][my_tuple]=new_val
            my_LHS_frac=0
            for act in dict_x_2_coeff:
                term_add=(x[act]*dict_x_2_coeff[act])
                my_LHS_frac=my_LHS_frac+term_add
            #    if abs(dict_x_2_coeff[act])>0.0001:
            #        print(['act'+act+' term_add  '+ str(term_add)+'    x[act]  '+str(x[act])+' dict_x_2_coeff[act] '+str(dict_x_2_coeff[act])])
            if my_LHS_frac>cut_RHS:
                print('my_LHS_frac')
                print(my_LHS_frac)
                print('cut_RHS')
                print(cut_RHS)
                input('wrong no sense ')
            
            #print('my_LHS_frac')
            #print(my_LHS_frac)
            #print('cut_RHS')
            #print(cut_RHS)
            #print('self.my_sub_prob.my_set_cust')
            #print(self.my_sub_prob.my_set_cust)
            #print('new_cut_name')
            #print(new_cut_name)
            #input('--PASSED -')

            if OPT_X_input!=None:
                my_LHS=0
                my_LHS_frac=0
                print('analyzing')
                for act in dict_x_2_coeff:
                    #if abs(dict_x_2_coeff[act])>0.00001:
                    #    print('OPT_X_input[act]')
                    #    print(OPT_X_input[act])
                    #    print('dict_x_2_coeff[act]')
                    #    print(dict_x_2_coeff[act])
                    #    print('act')
                    #    print(act)
                    #    print('---')
                    #if  OPT_X_input[act]>0.0001:
                    #    print('OPT_X_input[act]')
                    #    print(OPT_X_input[act])
                    #    print('dict_x_2_coeff[act]')
                    #    print(dict_x_2_coeff[act])
                    #    input('OK look here')
                    my_LHS=my_LHS+(OPT_X_input[act]*dict_x_2_coeff[act])
                    my_LHS_frac=my_LHS_frac+(x[act]*dict_x_2_coeff[act])
                #print('dict_x_2_coeff')
                #print(dict_x_2_coeff)
                for d in dual_solution:
                    if  d.startswith("flow_in_out_")==False:#and abs(dual_solution[d])>0.0001:
                        print([d+'   '+str(dual_solution[d])])
                #print('my_LHS')
                #print(my_LHS)
                #print('my_LHS_frac')
                #print(my_LHS_frac)
                #print('cut_RHS')
                #print(cut_RHS)
                #print('lp_objective')
                #print(lp_objective)
                #print('my_LHS-my_rhs')
                #print(my_LHS-cut_RHS)
                #print('my_LHS_frac')
                #print(my_LHS_frac)
                #input('---')
                #print('sub_prob_name')

                print(self.sub_prob_name)
                for act in self.MF.all_non_null_action:
                    if OPT_X_input[act]>0.5:
                        print(act+'. OPT_X_input[act]  '+str(OPT_X_input[act]),'dict val '+str(dict_x_2_coeff[act]))
                if cut_RHS<my_LHS_frac:
                    input('ok no sense here')
                if cut_RHS>my_LHS:
                    print('non_neg dict2 ')
                    for d in self.dict_con_name_2_LB:
                        if abs(self.dict_con_name_2_LB[d])>0.01:
                            print([d,str(self.dict_con_name_2_LB[d])])
                    
                    print('self.my_sub_prob.my_ng_graph.DEBUG_my_candid_edges')
                    print(self.my_sub_prob.my_ng_graph.DEBUG_my_candid_edges)
                    print('self.my_sub_prob.my_ng_graph.is_special')
                    print(self.my_sub_prob.my_ng_graph.is_special)

                    #
                    with open("NEWINTERPlayHere.pkl", "wb") as f:
                        pickle.dump(self, f)
                    input('error here')
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
        self.m_sz_pairs=[]
        self.m_sz_pairs.append(tuple([6,10]))
        self.m_sz_pairs.append(tuple([6,9]))
        self.m_sz_pairs.append(tuple([2,3]))
        #self.m_sz_pairs.append(tuple([2,5]))
        self.m_sz_pairs.append(tuple([2,9]))
        self.m_sz_pairs.append(tuple([3,10]))
        self.m_sz_pairs.append(tuple([4,10]))

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
        for my_bend_prob in self.my_list_benders_cut_generator:
            print('generating cut ')
            print('my_bend_prob.sub_prob_name')
            print(my_bend_prob.sub_prob_name)
            print('----')
            [this_cut_value,did_gen_cut,this_time_opt]=my_bend_prob.generate_benders_cut(x_solution,OPT_X_input)
            if did_gen_cut==True:
                TOT_gen_cut=TOT_gen_cut+1
                tot_cut_value=tot_cut_value+this_cut_value
            tot_time_opt=tot_time_opt+this_time_opt
            max_time_opt=max([max_time_opt,this_time_opt])
            print('this_cut_value:  '+str(this_cut_value))
            print('tot_cut_value,TOT_gen_cut]:  '+str([tot_cut_value,TOT_gen_cut]))
            print('tot_time_opt.  '+str([this_time_opt,tot_time_opt]))
        print('[tot_cut_value,TOT_gen_cut]')
        print([tot_cut_value,TOT_gen_cut])
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

        for my_set in all_sets:

            my_sub_prob_input=sub_problem(self.MF,my_set,self.m_sz_pairs)
            M=my_sub_prob_input
            self.my_sub_prob.append(my_sub_prob_input)
            #    def __init__(self,MF,sub_prob_name,sub_prob_y_obj,A_ineq_x,A_ineq_y,A_eq_y,rhs_ineq):

            new_cut_gen=benders_cut_generator(self.MF,M.sub_prob_name,M.var_2_cost,M.A_ineq_x,M.A_ineq_y,M.A_eq_y,M.rhs_ineq,my_sub_prob_input)
            self.my_list_benders_cut_generator.append(new_cut_gen)
            counter=counter+1
            print('my_set_cust:  '+str(my_set))
            print('counter:  '+str(counter))
            if OPT_X_SOL!=None:
                print('my_set')
                print(my_set)
                print('counter')
                print(counter)
                [lp_objective,did_add]=new_cut_gen.generate_benders_cut(OPT_X_SOL)
                counter=counter+1
                if did_add==False or lp_objective>0.001:
                    print('lp_objective')
                    print(lp_objective)
                    print('did_add')
                    print(did_add)
                    print('--my primal solution -')
                    for p in new_cut_gen.primal_solution:
                        #if new_cut_gen.primal_solution[p]>0.0001:
                        print([p+'   '+str(new_cut_gen.primal_solution[p])])
                    print('new_cut_gen.lp_objective')
                    print(new_cut_gen.lp_objective)
                    input('CUT HAS AN ISSUE FAILS HERE on cut')

class sub_problem:
    def __init__(self,MF,my_set_cust,m_sz_pairs):
        self.MF=MF
        self.my_set_cust=my_set_cust
        self.m_sz_pairs=m_sz_pairs

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
                if var_name in self.MF.delta_name_2_ub and self.MF.delta_name_2_ub[var_name]<0.001:
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
            if ACT_NAME in self.MF.delta_name_2_ub and self.MF.delta_name_2_ub[ACT_NAME]<0.001:
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
            self.rhs_ineq[con_name]=.9999
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
        for q in self.subset_and_divisor:

            my_subset=q[0]
            my_divisor=q[1]
            con_name='my_SRI_'+str(q)

            self.rhs_ineq[con_name]=-np.floor(len(my_subset)/my_divisor)-0.000001

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
                    #print('new_node')
                    #print(new_node)
                    #input('killing by size ')
                    #if s2==frozenset([0,1,2,3,4,5,6]):
                    #    input('look here')
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
                
                #print('i')
                #print(i)
                
                visited_excluding_last=i[1]
                self.early_depart_time_by_node[i]=-np.inf
                u=i[0]
                for w in visited_excluding_last:
                    var_name='act_'+str(w)+'_'+str(u)
                    if var_name not in self.MF.action_2_cost:
                        continue
                    if var_name in self.MF.delta_name_2_ub and self.MF.delta_name_2_ub[var_name]<0.001:
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
                        #print('removing')
                        eREM=tuple([node_pred,i])
                        #print(eREM)
                        #print('removing')
                        self.edges_removed.append(eREM)
        

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
                #if len(self.my_orderings_by_node[my_node])>0.5:
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
        if act in self.MF.delta_name_2_ub and self.MF.delta_name_2_ub[act]<0.001:
            out=-np.inf
            
        return out

    