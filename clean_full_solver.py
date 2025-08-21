import pickle
from collections import defaultdict
from typing import Dict, DefaultDict, Set, List
import numpy as np
import pulp as pl
from pulp import LpProblem, LpVariable, LpMaximize, PULP_CBC_CMD
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xpress as xp
import networkx as nx
import time
import heapq
from scipy.sparse import csr_matrix
from lower_bound_LP_milp import lower_bound_LP_milp
from pre_process.naive_pre import *

#from exper_lower_bound_LP_MILP import lower_bound_LP_milp
import pulp
from compressor import compressor
from class_new_valid import complete_separater_end_to_end 
from power_set import power_set
from baseline_solver import baseline_solver
import json
from New_valid_sep.check_valid_round_2 import check_valid_round_2
from projector_on_lb import projector_on_lb
class full_solver:

    def __init__(self,full_input_dict,jy_opt,output_file_path,all_actions_inclumbent=None,actions_ignore=None,hist_terms_phase_one=None,init_disc=None):
        print('type(full_input_dict)')
        print(type(full_input_dict))
        self.all_actions_inclumbent=None
        
        self.D=full_input_dict
        self.count_cutting_planes=0
        self.jy_opt=jy_opt
        self.output_file_path=output_file_path
        self.full_input_dict=full_input_dict
        self.my_VRP=full_input_dict['my_VRP']

        self.all_delta=full_input_dict['allDelta']
        self.all_actions_not_source_sink_connected=full_input_dict['all_actions_not_source_sink_connected']
        
        #self.all_graph_names:  names of all of the graphs
        self.all_graph_names=full_input_dict['allGraphNames']
        #graph_name_2_nodes:  given a graph_name gives  you the nodes  names
        self.graph_name_2_nodes=full_input_dict['graphName2Nodes']
        #self.all_actions:  list of the names of all actions
        self.all_actions=full_input_dict['allActions']
        #self.null_action:  name of the null action
        self.null_action=full_input_dict['nullAction']
        #self.all_exog; list of the names of all exogenous constraints
        self.all_exog=full_input_dict['allExogNames']
        #all_non_null_action:  list of the all_non_null_action
        self.all_non_null_action=full_input_dict['allNonNullAction']
        #self.exog_name_2_rhs:  mapping of exogenous contraitns to RHS
        self.exog_name_2_rhs=full_input_dict['exogName2Rhs']
        #self.all_integCon:  all of the contraints used to intregrate with the psi constriants 
        self.all_integCon=full_input_dict['allIntegCon']
        #self.all_primitive_vars;  list of all primitive varaiables names of psi
        self.all_primitive_vars=full_input_dict['allPrimitiveVars']
        # dictionary to mapping compact variable name to cost
        self.action_2_cost=full_input_dict['action2Cost']
        # dictionary to mapping compact variable name and constriant name to contirbutoin
        self.action_con_2_contrib=full_input_dict['actionCon2Contrib']
        #delta_con_2_contrib:   dictionary to mapping delta,constraint 2 contrib
        self.delta_con_2_contrib=full_input_dict['deltaCon2Contrib']
        #action_integCon_2_contrib
        self.action_integCon_2_contrib=full_input_dict['actionIntegCon2Contrib']
        #given any action, and interCon maps it to a contribution
        self.prim_integCon_2_contrib=full_input_dict['primIntegCon2Contrib']
        #gien any h,i,j it maps it to the ids for p
        #indexed first by h then by [i,j]
        self.hij_2_P=full_input_dict['hij2P']
        #maps the names of the graphs to the id for the source
        self.h_2_source_id=full_input_dict['h2SourceId']
        #maps the names of the graphs to the id for the sink
        self.h_2_sink_id=full_input_dict['h2sinkid']
        #self.init_agg_graph_node_2_agg_node:  rganize first by h then by  node name then by the aggregated node
        self.graph_node_2_agg_node=full_input_dict['initGraphNode2AggNode']
        self.orig_init_graph_node_2_agg_node=dict()
        for h in self.graph_node_2_agg_node:
            self.orig_init_graph_node_2_agg_node[h]=dict()
            for i in self.graph_node_2_agg_node[h]:
                self.orig_init_graph_node_2_agg_node[h][i]=str(self.graph_node_2_agg_node[h][i])
        if init_disc!=None:
            self.graph_node_2_agg_node=init_disc
        if all_actions_inclumbent==None:
            self.all_actions_inclumbent=set(self.all_actions)-set(self.all_actions_not_source_sink_connected)
        else:
            self.all_actions_inclumbent=all_actions_inclumbent
        if actions_ignore==None:
            self.actions_ignore=[]
        else:
            self.actions_ignore=actions_ignore.copy()

        self.graph_names=full_input_dict['allGraphNames']

        
        self.delta_name_2_ub=full_input_dict['delta_name_2_ub']
        self.delta_name_2_lb=full_input_dict['delta_name_2_lb']
        self.ineq_replaced_by_lb_ub=full_input_dict['ineq_replaced_by_lb_ub']

        self.history_dict=dict()
        self.history_dict['lblp_lower']=[]
        self.history_dict['ub_lp']=[]
        self.history_dict['prob_sizes_at_start']=[]
        self.history_dict['prob_sizes_mid']=[]
        self.history_dict['did_compress']=[]
        self.history_dict['lp_time_project']=[]
        self.history_dict['lp_time_LB']=[]
        self.history_dict['ilp_time']=[]
        self.hist_terms_phase_one=hist_terms_phase_one

        self.history_dict['history_of_graphs_by_iter']=[]

        self.apply_complete_algorithm()
        with open(self.output_file_path, 'w') as file:
            json.dump(self.history_dict, file)

    def apply_splitting_2(self):
        incumbent_lp=np.inf
        tinyPos=0.0000001
        if len(self.history_dict['ub_lp'])>0:
            incumbent_lp=self.history_dict['ub_lp'][-1]
        for p in self.action_2_cost:
            if self.my_lower_bound_LP.lp_primal_solution[p]>tinyPos:#0.0001:
                self.all_actions_inclumbent.add(p)
        actions_ignore=set(self.all_non_null_action)-self.all_actions_inclumbent
        if 'null_action' in actions_ignore:
            actions_ignore.remove('null_action')
        my_proj=projector_on_lb(self,actions_ignore)
        duality_gap=my_proj.lp_objective-self.my_lower_bound_LP.lp_objective
        self.incumbant_solution=defaultdict(float)
        if my_proj.lp_objective<incumbent_lp-duality_gap/100:
            if self.jy_opt['do_remove_actions_from_incumbant']==True:
                self.all_actions_inclumbent=set([])
            for p in self.action_2_cost:
                if p in my_proj.lp_primal_solution and my_proj.lp_primal_solution[p]>tinyPos:#0.00001:
                    self.all_actions_inclumbent.add(p)
                    self.incumbant_solution[p]=my_proj.lp_primal_solution[p]
        objective_gain=my_proj.lp_objective-self.my_lower_bound_LP.lp_objective
        #print('objective_gain')
        #print(objective_gain)
        #print('my_proj.num_do_split')
        #print(my_proj.num_do_split)
        #input('---')
        self.my_proj=my_proj
        self.history_dict['ub_lp'].append(my_proj.lp_objective)
 
        my_Lp_time=my_proj.lp_time
        did_split=False
        if objective_gain>0.1 and my_proj.num_do_split>0.5:#%self.jy_opt['epsilon']:
            self.graph_node_2_agg_node=my_proj.NAIVE_graph_node_2_agg_node
            
            did_split=True

        return did_split,my_proj.lp_objective,my_Lp_time

    def count_size(self,supress_output=True):
        my_count_size=dict()
        if supress_output==False:
            print('Graph Sizes (in terms of number of compressed nodes) are below:  ')
        for h in self.graph_names:
            my_count_size[h]=len(set(self.graph_node_2_agg_node[h].values()))
            if supress_output==False:
                print('h:   '+h+' is of size:   '+str(my_count_size[h]))
        return my_count_size

    def prepare_ILP_solution(self):
        
        if self.jy_opt['do_ilp']>0.5:
            my_ilp_sol=self.my_lower_bound_ILP.milp_solution
            out_sol=dict()
            if self.jy_opt['use_delta_in_milp']==True:
                for my_delta in self.all_delta:
                    out_sol[my_delta]=my_ilp_sol[my_delta]

            tot_cost=0
            for my_act in self.all_actions:
                out_sol[my_act]=my_ilp_sol[my_act]
                if my_ilp_sol[my_act]>0.5:
                    tot_cost=tot_cost+self.action_2_cost[my_act]

            for my_prim in self.all_primitive_vars:
                if my_prim in my_ilp_sol:
                    if my_ilp_sol[my_prim]>0.01:
                        out_sol[my_prim]=my_ilp_sol[my_prim]
            self.history_dict['jy_opt']=self.jy_opt
            self.history_dict['output_ilp_solution']=out_sol
        

    
    def augment_history_graphs(self):

        new_hist=dict()
        for h in self.all_graph_names:
            new_hist[h]=dict()
            for i in self.graph_node_2_agg_node[h]:
                f=self.graph_node_2_agg_node[h][i]
                f_str=f[:]
                i_str=i[:]
                new_hist[h][i_str]=f_str

        self.history_dict['history_of_graphs_by_iter'].append(new_hist)

    def apply_complete_algorithm(self):
        self.incumbant_lp=-np.inf
        iter=0
        did_split=True
        self.current_LP_solution=[]
        did_add_ineq=True
        self.call_init_separ()
        #input('--')
        while iter<self.jy_opt['max_iterations_loop_compress_project'] and (did_split==True or did_add_ineq==True):
            self.time_list_outer=dict()
            iter=iter+1
            t1=time.time()

            prob_sizes_at_start=self.count_size()
            self.time_list_outer['part0']=time.time()-t1
            self.my_lower_bound_LP=lower_bound_LP_milp(self,self.graph_node_2_agg_node,False,False)            
            self.current_LP_solution=self.my_lower_bound_LP.lp_primal_solution#.copy()
            lblp_time=self.my_lower_bound_LP.lp_time
            new_lp_value=self.my_lower_bound_LP.lp_objective
            if self.incumbant_lp<new_lp_value-self.jy_opt['min_inc_2_compress']: #and iter>0:
                self.graph_node_2_agg_node=self.my_lower_bound_LP.NAIVE_graph_node_2_agg_node
                did_compress_call=True
                self.incumbant_lp=new_lp_value
                if self.jy_opt['restore_after_each_step']>0.5:
                    self.split_based_init()
                    print('splitting up after')
                else:
                    print('NOG SPLITTING up after')

            [did_split,proj_objective_componentLps,proj_time_component_lps]=self.apply_splitting_2()
            if did_split==False:
                #[did_add_ineq,new_exog_terms,new_action_contrib]=self.separate_zero_val_terms(self.my_lower_bound_LP.lp_primal_solution)
                [did_add_ineq,new_exog_terms,new_action_contrib]=self.separate_zero_val_terms_internal(self.my_lower_bound_LP.lp_primal_solution)
                #if did_add_ineq==True:
                #    print('MAYBE;  NOT SURE YET; This really should not occur I dont think i mea')
                #    input('hih')
            if did_split==False and did_compress_call==False and did_add_ineq==False:
                self.graph_node_2_agg_node=self.my_lower_bound_LP.NAIVE_graph_node_2_agg_node
                if self.jy_opt['do_split_based_init']>0.5:
                    self.split_based_init()
                    print('AT TERM splitting up after')

                else:
                    print('AT TERM NOT splitting up after')

            t1=time.time()

            self.history_dict['lblp_lower'].append(new_lp_value)
            self.history_dict['prob_sizes_at_start'].append(prob_sizes_at_start)
            self.history_dict['did_compress'].append(did_compress_call)
            self.history_dict['lp_time_project'].append(proj_time_component_lps)
            self.history_dict['lp_time_LB'].append(lblp_time)
            if self.jy_opt['save_graph_each_iter']>0.5:
                self.augment_history_graphs()
            
            print('ITER FINISHE:  '+str(iter))
            print('new_lp_value=  '+str(new_lp_value))
            print('did_compress_call:  '+str(did_compress_call))
            print('lp project time '+str(self.history_dict['lp_time_project'][-1]))
            print('lplb time '+str(self.history_dict['lp_time_LB'][-1]))
            print('prob_sizes_at_start')
            print(prob_sizes_at_start)

        if self.jy_opt['do_ilp']>0.5:
            self.call_ILP_solver()
        if self.jy_opt['run_baseline']:
            self.call_baseline()
        if self.jy_opt['do_ilp']>0.5:

            self.prepare_ILP_solution()
    
    

    def call_ILP_solver(self):
        if self.jy_opt['do_split_based_init']>0.5:
            self.split_based_init()
       

        self.my_lower_bound_ILP=lower_bound_LP_milp(self,self.graph_node_2_agg_node,True,False)
        new_Ilp_value=self.my_lower_bound_ILP.milp_solution_objective_value
        self.history_dict['OUR_ilp_objective']=new_Ilp_value
        self.history_dict['OUR_MIP_Lower_Bound']=self.my_lower_bound_ILP.MIP_lower_bound
        self.history_dict['OUR_ilp_time']=self.my_lower_bound_ILP.milp_time
        self.history_dict['OUR_gurobi_MILP_str']=self.my_lower_bound_ILP.gurobi_MILP_str
        print('final solution objective')
        print(new_Ilp_value)
    def call_baseline(self):
        print('running baseline')
        if (self.jy_opt['in_demo_mode']==True):
            input('Press enter about to start the running of the baseline ILP')
        self.jy_opt['max_ILP_time']=1000000
        my_base=baseline_solver(self,True,False)
        self.history_dict['BASE_ILP_sol_obj']=my_base.milp_solution_objective_value
        self.history_dict['BASE_milp_solution']=my_base.milp_solution
        self.history_dict['BASE_milp_time']=my_base.milp_time
        self.history_dict['BASE_MIP_lower_bound'] = my_base.MIP_lower_bound#model.ObjBound



    def split_based_init(self):
        self.NEW_graph_node_2_agg_node=dict()
        for h in self.graph_names:
            self.NEW_graph_node_2_agg_node[h]=dict()
            non_sourc_sink_nodes_of_h=set(self.graph_node_2_agg_node[h])-set([self.h_2_sink_id[h],self.h_2_source_id[h]])

            self.NEW_graph_node_2_agg_node[h][self.h_2_sink_id[h]]=self.graph_node_2_agg_node[h][self.h_2_sink_id[h]]
            self.NEW_graph_node_2_agg_node[h][self.h_2_source_id[h]]=self.graph_node_2_agg_node[h][self.h_2_source_id[h]]
            for i in non_sourc_sink_nodes_of_h:
                old_name=self.graph_node_2_agg_node[h][i]
                init_name=self.orig_init_graph_node_2_agg_node[h][i]
                new_name=old_name+"_"+init_name
                
                self.NEW_graph_node_2_agg_node[h][i]=new_name
        self.graph_node_2_agg_node=self.NEW_graph_node_2_agg_node
    

    def separate_zero_val_terms(self,primal_sol_lp):
        #input('IN HERE')
        if hasattr(self,'did_init_separ')==False:
            self.call_init_separ()
        D=self.D
        costs=D['action2Cost']
        G=self.G
        ub_keys = set(self.delta_name_2_ub)

        for g, zs in self.Z_by_group.items():
            self.Z_by_group[g] = zs - ub_keys
        for g, zs in self.Z_by_group_inside.items():
            self.Z_by_group_inside[g] = zs - ub_keys

        self.cost_val = {
            g: min(costs[act] for act in self.Z_by_group[g]) if self.Z_by_group[g] else float("inf")
            for g in G
        }
        amount_inside = {
            g: (sum(primal_sol_lp[act] for act in self.Z_by_group[g]))
            for g in G
        }
        K=10
        filtered = [
            ((1 - amount_inside[g]) * self.cost_val[g], g)
            for g in amount_inside
            if amount_inside[g] < 0.99
        ]
        #print('len(filtered)')
        #print(len(filtered))
        #input('--')
        largest_k = heapq.nlargest(K, filtered)

        result = [g for _, g in largest_k]
        new_exog_terms=dict()
        new_action_contrib=dict()
        for g in result:
            con_name_ineq='NewLB_'+str(g)
            self.full_input_dict['exogName2Rhs'][con_name_ineq]=1
            self.full_input_dict['allExogNames'].append(con_name_ineq)
            new_exog_terms[con_name_ineq]=1
            print('adding ')
            print('con_name_ineq')
            print(con_name_ineq)
            for act in self.Z_by_group[g]:
                self.full_input_dict['actionCon2Contrib'][tuple([act,con_name_ineq])]=1#self.delta_con_2_contrib[v_con]
                new_action_contrib[tuple([act,con_name_ineq])]=1
                print('tuple([act,con_name_ineq])')
                print(tuple([act,con_name_ineq]))
        did_add_ineq=False
        if len(result)>0.5:
            did_add_ineq=True
           # print('did_find_separ')
           # print(did_add_ineq)
           # input('did_find_separ')


       # if did_find_separ:
        #    for con_name in new_exog_terms:
        #        self.full_input_dict['exogName2Rhs'][con_name]=new_exog_terms[]
        #        self.full_input_dict['allExogNames'].append(con_name)
            #    for (act,con_name) in new_action_contrib:
            #        self.dict_var_con_2_lhs_exog[(act,con_name)]=new_action_contrib[(act,con_name)]

        return [did_add_ineq,new_exog_terms,new_action_contrib]
        
    
    def separate_zero_val_terms_internal(self,primal_sol_lp):
        #input('IN HERE')
        if hasattr(self,'did_init_separ')==False:
            self.call_init_separ()
        D=self.D
        costs=D['action2Cost']
        G=self.G
        ub_keys = set(self.delta_name_2_ub)

        for g, zs in self.Z_by_group.items():
            self.Z_by_group[g] = zs - ub_keys
        for g, zs in self.Z_by_group_inside.items():
            self.Z_by_group_inside[g] = zs - ub_keys

        self.cost_val = {
            g: min(costs[act] for act in self.Z_by_group[g]) if self.Z_by_group[g] else float("inf")
            for g in G
        }
        amount_inside = {
            g: (sum(primal_sol_lp[act] for act in self.Z_by_group_inside[g]))
            for g in G
        }
        K=10
        filtered = [
            ((amount_inside[g]+0.99999-len(g)) * self.cost_val[g], g)
            for g in amount_inside
            if (amount_inside[g]+0.99999-len(g))>0.01
        ]
        #print('len(filtered)')
        #print(len(filtered))
        #input('--')
        largest_k = heapq.nlargest(K, filtered)

        result = [g for _, g in largest_k]
        new_exog_terms=dict()
        new_action_contrib=dict()
        for g in result:
            con_name_ineq='NewLB_'+str(g)
            self.full_input_dict['exogName2Rhs'][con_name_ineq]=-len(g)+.999
            self.full_input_dict['allExogNames'].append(con_name_ineq)
            new_exog_terms[con_name_ineq]=-len(g)+.999
            print('adding ')
            print('con_name_ineq')
            print(con_name_ineq)
            print('new con val')
            print(self.full_input_dict['exogName2Rhs'][con_name_ineq])
            for act in self.Z_by_group_inside[g]:
                self.full_input_dict['actionCon2Contrib'][tuple([act,con_name_ineq])]=-1#self.delta_con_2_contrib[v_con]
                new_action_contrib[tuple([act,con_name_ineq])]=-1
                print('tuple([act,con_name_ineq])')
                print(tuple([act,con_name_ineq]))
        did_add_ineq=False
        if len(result)>0.5:
            did_add_ineq=True
            #input('ADDING HRE LOOK.  IS THIS SYSTEM OPERATIONAL CHECK')
        return [did_add_ineq,new_exog_terms,new_action_contrib]
        

    
    def call_init_separ(self):
        self.did_init_separ=True
        my_VRP=self.D['my_VRP']
        Nc=self.my_VRP.num_cust

        self.dict_pred_gain=dict()
        [ng_neigh_by_cust_power,junk]=naive_get_LA_neigh(my_VRP,self.jy_opt['LAB_MP_neigh_use_power'])
        [ng_neigh_by_cust_all,junk]=naive_get_LA_neigh(my_VRP,Nc)
        G=set()
        for u in range(0,Nc):
            #QZ=38
            #print('ng_neigh_by_cust_power[QZ]')
            #print(ng_neigh_by_cust_power[QZ])
            #print('len(ng_neigh_by_cust_power[QZ])')
            #print(len(ng_neigh_by_cust_power[QZ]))
            neighborhood = set(ng_neigh_by_cust_power[u]) | {u}
            for g in power_set(neighborhood):
                if len(g)>1:  # skip empty set and size 1 sets
                    G.add(frozenset(g))
            #print('neighborhood[u]')
            #print(neighborhood)
            #print('u')
            #print(u)
            #if u==QZ:
            #    print('neighborhood[u]-1')
            #    print({v + 1 for v in neighborhood})
            #    print('u+1')
            #    print(u+1)
            #    input('---')
            for k in range(1,Nc):
                my_terms=ng_neigh_by_cust_all[u][0:k]
                G.add(frozenset(my_terms))
            
        G.add(frozenset(np.arange(0,Nc)))

        if 1<0:
            G=set()
            C1 = frozenset({0,1,2,3,4,5,6,7})
            C2 = frozenset({8,9,10,11,12,13,14,15,16})
            C3 = frozenset({17,18,19,20,21,22,23,24})
            C12=frozenset(set(C1).union(set(C2)))
            C13=frozenset(set(C1).union(set(C3)))
            C23=frozenset(set(C2).union(set(C3)))
            C123=frozenset(np.arange(0,25))
            G.add(C1)
            G.add(C2)
            G.add(C3)
            G.add(C12)
            G.add(C13)
            G.add(C23)
            G.add(C123)
            #G=set()
            #TL = frozenset({4, 2, 0, 44, 7, 45, 3, 6, 5, 1})
            #G.add(TL)
            #TC = frozenset({42, 39, 35, 34, 43, 36, 41, 38, 37, 40})
            #G.add(TC)

            #BL = frozenset({16, 46, 13, 11, 15, 14, 10, 9, 12, 8})
            #G.add(BL)

            #BC = frozenset({23, 21, 19, 48, 18, 24, 22, 20, 47, 17})
            #G.add(BC)

            #BR = frozenset({49, 33, 30, 28, 26, 31, 29, 27, 25, 32})
            #G.add(BR)

        self.Z_by_group = defaultdict(set)  # g → set of act_u_v
        self.Z_by_group_inside = defaultdict(set)  # g → set of act_u_v
        all_non_null_action=set(self.D['allNonNullAction'])-set(['null_action'])
        u_in_groups = defaultdict(set)
        v_not_in_groups = defaultdict(set)

        for g in G:
            for u in g:
                u_in_groups[u].add(g)
            for v in range(0,Nc+2):  # include depot if needed
                if v not in g:
                    v_not_in_groups[v].add(g)

        # Step 2: Build Z_by_group using set intersection
        num_add=0

        for act in all_non_null_action:
            num_add=num_add+1
            _, u_str, v_str = act.split("_")
            u, v = int(u_str), int(v_str)

            relevant_groups = u_in_groups[u] & v_not_in_groups[v]
            relevant_groups_inside = u_in_groups[u] & u_in_groups[v]
            for g in relevant_groups:
                self.Z_by_group[g].add(act)
            for g in relevant_groups_inside:
                self.Z_by_group_inside[g].add(act)
        self.G=G
        
