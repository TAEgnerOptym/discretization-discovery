import ast
import copy
import re

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
from benders_repo_new import benders_repo_new
from pre_process.ng_neigh_fancy_paper import *

#from EXPER_benders_repo_new import benders_repo_new
class full_solver:

    def __init__(self,full_input_dict,jy_opt,output_file_path,all_actions_inclumbent=None,actions_ignore=None,hist_terms_phase_one=None,init_disc=None,OPT_SOL=None,Actions_of_given_sol=None):
        print('type(full_input_dict)')
        print(type(full_input_dict))
        self.all_actions_inclumbent=None
        
            
        self.OPT_SOL=OPT_SOL
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
        self.graph_name_2_nodes=full_input_dict['graphName2Nodes']
        self.graphNameNode_2_cust=full_input_dict['graphNameNode_2_cust']
        #print('graphNameNode_2_cust')
        #print(self.graphNameNode_2_cust['timeGraph'])
        #input('---')
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
        #if all_actions_inclumbent==None:
        #    self.all_actions_inclumbent=set(self.all_actions)-set(self.all_actions_not_source_sink_connected)
        #else:
        #    self.all_actions_inclumbent=all_actions_inclumbent
        if actions_ignore==None:
            self.actions_ignore=[]
        else:
            self.actions_ignore=actions_ignore.copy()

        self.graph_names=full_input_dict['allGraphNames']

        self.all_source_sink_actions=set(self.all_actions)-set(self.all_actions_not_source_sink_connected)
        self.init_default_solution_is_empty=True
        self.Actions_of_given_sol=self.jy_opt['Actions_of_given_sol'].copy()
        if len(self.Actions_of_given_sol)==0:
            self.Actions_of_given_sol=self.all_source_sink_actions
            self.init_default_solution_is_empty=False
        self.objective_of_initial_sol=0
        for my_act in self.Actions_of_given_sol:
            self.objective_of_initial_sol+=self.action_2_cost[my_act]
        
        self.all_actions_inclumbent=set(self.Actions_of_given_sol.copy())

        if self.jy_opt['ub_use_remove']>self.objective_of_initial_sol:
            self.jy_opt['ub_use_remove']=self.objective_of_initial_sol+0.001
            
            print('self.Actions_of_given_sol')
            print(self.Actions_of_given_sol)
            print('self.objective_of_initial_sol')
            print(self.objective_of_initial_sol)
            #input('ok im here not a problem just to flag me')
        self.delta_name_2_ub=full_input_dict['delta_name_2_ub']
        self.delta_name_2_lb=full_input_dict['delta_name_2_lb']
        self.ineq_replaced_by_lb_ub=full_input_dict['ineq_replaced_by_lb_ub']

        self.history_dict=dict()
        self.history_dict['lblp_lower']=[]
        self.history_dict['cuttingPlaneBendInfo']=[]
        self.history_dict['time_compress']=[]
        self.history_dict['ub_lp']=[]
        self.history_dict['prob_sizes_at_start']=[]
        self.history_dict['prob_sizes_after_compress']=[]
        self.history_dict['prob_sizes_after_split']=[]
        self.history_dict['did_compress']=[]
        self.history_dict['lp_time_project']=[]
        self.history_dict['lp_time_LB']=[]
        self.history_dict['ilp_time']=[]
        self.hist_terms_phase_one=hist_terms_phase_one

        self.history_dict['history_of_graphs_by_iter']=[]

        #self.my_bender_repo=benders_repo_new(self)
        #input('---')
        if 'ngGraph' in self.D['hij2P']:
            self.ORIG_ng_graph_h_ijp=self.D['hij2P']['ngGraph'].copy()
            self.ORIG_h_ijp=copy.deepcopy(self.D['hij2P'])
            
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
        #self.count_size(False)
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
            actions_used=[]
            for my_act in self.all_actions:
                #out_sol[my_act]=my_ilp_sol[my_act]
                if my_ilp_sol[my_act]>0.5:
                    tot_cost=tot_cost+self.action_2_cost[my_act]
                    out_sol[my_act]=1#my_ilp_sol[my_act]
                    actions_used.append(my_act)
            #for my_prim in self.all_primitive_vars:
            #    if my_prim in my_ilp_sol:
            #        if my_ilp_sol[my_prim]>0.01:
            #            out_sol[my_prim]=my_ilp_sol[my_prim]
            if (tot_cost-self.history_dict['OUR_ilp_objective'])>0.001:
                input('wierd')
            else:
                print('GOOD CORRECT. tot_cost')
                print(tot_cost)
            self.history_dict['jy_opt']=self.jy_opt
            self.history_dict['output_ilp_solution']=out_sol

            self.history_dict['Actions_of_given_sol']=str(self.Actions_of_given_sol)
            self.history_dict['actions_used']=actions_used
            
            self.generate_paths_from_actions(out_sol)
            #print('ready')
            self.check_feas_and_cost()


            self.history_dict['init_default_solution_is_empty']=self.init_default_solution_is_empty
            self.history_dict['Actions_of_given_sol']='none just used source and sink terms'#str(self.Actions_of_given_sol)

            if self.init_default_solution_is_empty>0.5:
                self.history_dict['Actions_of_given_sol']=str(self.Actions_of_given_sol)

            self.history_dict['objective_of_initial_sol']=self.objective_of_initial_sol
            self.history_dict['my_paths']=str(self.my_paths)
    
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
        did_add_ineq=False
        did_ever_generate_cut=False
        self.jy_opt['LAB_MP_ON']=0
        #self.call_init_separ()
        #input('--')
        num_calls_ineq=0
        while iter<self.jy_opt['max_iterations_loop_compress_project'] and (did_split==True or did_add_ineq==True):
            did_add_ineq=False

            self.time_list_outer=dict()
            iter=iter+1
            
            t1=time.time()

            prob_sizes_at_start=self.count_size()
            self.time_list_outer['part0']=time.time()-t1
            self.my_lower_bound_LP=lower_bound_LP_milp(self,self.graph_node_2_agg_node,False,False)            
            self.current_LP_solution=self.my_lower_bound_LP.lp_primal_solution#.copy()
            lblp_time=self.my_lower_bound_LP.lp_time
            new_lp_value=self.my_lower_bound_LP.lp_objective
            if iter>2 and new_lp_value<-0.1+max(self.history_dict['lblp_lower']):
                print('old val list')
                print(self.history_dict['lblp_lower'])
                print('new_lp_value')
                print(new_lp_value)
                input('error here')
            if self.incumbant_lp<new_lp_value-self.jy_opt['min_inc_2_compress']: #and iter>0:
                if self.jy_opt['use_classic_compress']>0.5:
                    self.graph_node_2_agg_node=self.my_lower_bound_LP.NAIVE_graph_node_2_agg_node
                    self.history_dict['time_compress'].append(0)
                else:
                    print('starting comrpession fancy')
                    time_add=0
                    if 1>0:
                        prob_sizes_after_compress=self.count_size()
                        print('STGE 1=prob_sizes_after_compress')
                        print(prob_sizes_after_compress)
                        self.graph_node_2_agg_node=self.my_lower_bound_LP.NAIVE_graph_node_2_agg_node                    
                        if self.jy_opt['restore_after_each_step']>0.5:
                            print('doing split in here')
                            self.split_based_init()
                        self.my_lower_bound_LP=lower_bound_LP_milp(self,self.graph_node_2_agg_node,False,False)            
                        #self.graph_node_2_agg_node=self.my_lower_bound_LP.NAIVE_graph_node_2_agg_node                    
                        #self.my_lower_bound_LP.make_agg_node_2_nodes()
                        #if self.jy_opt['restore_after_each_step']>0.5:
                        #    print('doing split in here')
                        #    self.split_based_init()

                        self.current_LP_solution=self.my_lower_bound_LP.lp_primal_solution#.copy()
                        time_add=self.my_lower_bound_LP.lp_time
                        if abs(self.my_lower_bound_LP.lp_objective-new_lp_value)>0.1:
                            print('self.my_lower_bound_LP.lp_objective')
                            print(self.my_lower_bound_LP.lp_objective)
                            print('new_lp_value')
                            print(new_lp_value)
                            input('ERRR here')
                    self.my_compressor=compressor(self) 
                    print('time_add')
                    print(time_add)
                    #self.my_compressor.lp_time+=time_add
                    print('done  comrpession fancy')
                    
                    self.history_dict['time_compress'].append(self.my_compressor.lp_time)
                    self.graph_node_2_agg_node=self.my_compressor.NEW_graph_node_2_agg_node
                    prob_sizes_after_compress=self.count_size()
                    print('STGE 2=prob_sizes_after_compress')
                    print(prob_sizes_after_compress)
                    print('---')
                prob_sizes_after_compress=self.count_size()
                did_compress_call=True
                self.incumbant_lp=new_lp_value
                if self.jy_opt['restore_after_each_step']>0.5:
                    self.split_based_init()
                    print('splitting up after')
                else:
                    print('NOG SPLITTING up after')
                #print('003 just Before')
                #self.count_size(False)
            #print('002 just Before')
            #self.count_size(False)
            [did_split,proj_objective_componentLps,proj_time_component_lps]=self.apply_splitting_2()
            #print('001 just Before')
            #self.count_size(False)
            this_cutting_plane_info={'tot_cut_value':0,'TOT_gen_cut':0,'tot_time_opt':0,'max_time_opt':0}
            if did_ever_generate_cut==False:
                self.history_dict['ROOT_LP_PRIOR_ADDING_CUTS']=new_lp_value
            if did_split==False and self.jy_opt['ub_use_remove']-new_lp_value>0.001 and self.jy_opt['use_ineq']>0.5:
                
                did_ever_generate_cut=True
                #[did_add_ineq,new_exog_terms,new_action_contrib]=self.separate_zero_val_terms(self.my_lower_bound_LP.lp_primal_solution)
                self.my_bender_repo=benders_repo_new(self)

                [tot_cut_value,TOT_gen_cut,tot_time_opt,max_time_opt]=self.my_bender_repo.generate_cuts(self.my_lower_bound_LP.lp_primal_solution,self.OPT_SOL)
                this_cutting_plane_info={'tot_cut_value':tot_cut_value,'TOT_gen_cut':TOT_gen_cut,'tot_time_opt':tot_time_opt,'max_time_opt':max_time_opt}
                print('tot_cut_value:  '+ str(tot_cut_value)+'. TOT_gen_cut: '+str(TOT_gen_cut))
                #print(tot_cut_value)
                if TOT_gen_cut>0.5:
                    did_add_ineq=True
                if did_add_ineq==True:
                    self.all_actions_inclumbent=self.all_actions_inclumbent.union(self.Actions_of_given_sol)
                
                    #input('LOOK ME ')
                num_calls_ineq=num_calls_ineq+1
            #print('-001 just Before')
            #self.count_size(False)
            prob_sizes_after_split=self.count_size()

            if did_split==False  and did_add_ineq==False :
                #input("AT DONE DONE")
                if self.jy_opt['use_classic_compress_last']>0.5:
                    #print('in here')
                    #print('-002 just Before')
                    self.count_size(False)
                    self.graph_node_2_agg_node=self.my_lower_bound_LP.NAIVE_graph_node_2_agg_node
                    #print('-003 just Before')
                    #self.count_size(False)
                else:
                    print('FFINAL starting comrpession fancy')
                    time_add=0
                    if 1>0:
                        prob_sizes_after_compress=self.count_size()
                        print('FINAL STGE 1=prob_sizes_after_compress')
                        print(prob_sizes_after_compress)
                        self.graph_node_2_agg_node=self.my_lower_bound_LP.NAIVE_graph_node_2_agg_node
                        self.my_lower_bound_LP.graph_node_2_agg_node=self.graph_node_2_agg_node#self.my_lower_bound_LP.NAIVE_graph_node_2_agg_node
                    
                        self.my_lower_bound_LP=lower_bound_LP_milp(self,self.graph_node_2_agg_node,False,False)            
                        self.current_LP_solution=self.my_lower_bound_LP.lp_primal_solution#.copy()
                        time_add=self.my_lower_bound_LP.lp_time
                        if abs(self.my_lower_bound_LP.lp_objective-new_lp_value)>0.01:
                            input('ERRR here')
                    self.my_compressor=compressor(self) 
                    self.my_compressor.lp_time+=time_add
                    print('FIANL. done  comrpession fancy')
                    
                    self.history_dict['time_compressLAST']=(self.my_compressor.lp_time)
                    self.graph_node_2_agg_node=self.my_compressor.NEW_graph_node_2_agg_node
                    prob_sizes_after_compress=self.count_size()
                    print('FINAL STGE 2=prob_sizes_after_compress')
                    print(prob_sizes_after_compress)
                    print('-FINAL--')
                    print('---')
                    print('---')
                    print('-FINAL--')
                    print('---')
                    print('---')
                    print('-FINAL--')
                if self.jy_opt['do_split_based_init']>0.5:
                    self.split_based_init()
                    print('AT TERM splitting up after')

                else:
                    print('AT TERM NOT splitting up after')

            t1=time.time()
            

            self.history_dict['lblp_lower'].append(new_lp_value)
            self.history_dict['cuttingPlaneBendInfo'].append(this_cutting_plane_info)
            self.history_dict['prob_sizes_at_start'].append(prob_sizes_at_start)
            self.history_dict['prob_sizes_after_compress'].append(prob_sizes_after_compress)
            self.history_dict['prob_sizes_after_split'].append(prob_sizes_after_split)
            self.history_dict['did_compress'].append(did_compress_call)
            self.history_dict['lp_time_project'].append(proj_time_component_lps)
            self.history_dict['lp_time_LB'].append(lblp_time)
            if self.jy_opt['save_graph_each_iter']>0.5:
                self.augment_history_graphs()
            
            print('ITER FINISHE:  '+str(iter))
            print('new_lBLP_value=  '+str(new_lp_value))
            print('new ubLP value= '+ str(self.history_dict['ub_lp'][-1]))
            print('did_compress_call:  '+str(did_compress_call))
            print('lp project time '+str(self.history_dict['lp_time_project'][-1]))
            print('lp compress time '+str(self.history_dict['time_compress'][-1]))
            print('lplb time '+str(self.history_dict['lp_time_LB'][-1]))
            print('prob_sizes_at_start:  '+ str(prob_sizes_at_start))
            #print()
            print('prob_sizes_after_compress:  '+str(prob_sizes_after_compress))
            print(prob_sizes_after_compress)
            print('prob_sizes_after_split:  '+ str(prob_sizes_after_split))
            #print(prob_sizes_after_split)
            print('[did_add_ineq = '+str(did_add_ineq)+ '   did_split ='+str(did_split))
            #print([did_add_ineq,did_split])
            

        self.history_dict['FinalSizeBeforeILP']=self.count_size()
        print('self.history_dict[FinalSizeBeforeILP]')
        print(self.history_dict['FinalSizeBeforeILP'])
        print('self.history_dict[FinalSizeBeforeILP]')
        #input('about to call ilp')
        if self.jy_opt['do_ilp']>0.5:
            self.call_ILP_solver()
        
        if self.jy_opt['run_baseline']:
            for act in self.all_non_null_action:
            #if act in usedTerms:
            #    num_used=num_used+1
            #    continue
                if act in self.delta_name_2_ub:#and self.full_prob.delta_name_2_ub[act]<0.001:
                    del self.delta_name_2_ub[act]
            self.call_baseline()
        if self.jy_opt['do_ilp']>0.5:

            self.prepare_ILP_solution()
    
    

    def call_ILP_solver(self):
        if self.jy_opt['do_split_based_init']>0.5:
            self.split_based_init()
       

        self.my_lower_bound_ILP=lower_bound_LP_milp(self,self.graph_node_2_agg_node,True,False)
        new_Ilp_value=self.my_lower_bound_ILP.milp_solution_objective_value
        if self.jy_opt['ub_use_remove']>0:
            if new_Ilp_value>0.001+self.jy_opt['ub_use_remove']:
                print('WOrse that known optimal is generated')
                #input('error here ')
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
        #self.jy_opt['max_ILP_time']=1000000
        my_base=baseline_solver(self,True,False)
        self.history_dict['BASE_ILP_sol_obj']=my_base.milp_solution_objective_value
        self.history_dict['BASE_milp_solution']=my_base.milp_solution
        self.history_dict['BASE_milp_time']=my_base.milp_time
        self.history_dict['BASE_MIP_lower_bound'] = my_base.MIP_lower_bound#model.ObjBound
        self.history_dict['BASE_OUT_STR']=my_base.BASE_OUT_STR


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
    

    def generate_paths_from_actions(self,out_sol):
        """
        Build VRP routes from self.all_actions_inclumbent, which contains names like 'act_u_v'
        (or a dict mapping action->value; only truthy values kept).

        Allows:
        - START (= self.Nc) to have multiple successors (many vehicles).
        - END   (= self.Nc+1) to have multiple predecessors.
        Enforces:
        - For any node x not in {START, END}:
            outdegree(x) <= 1 and indegree(x) <= 1

        Returns:
        routes: List[List[int]], each route is [START, ..., END].
        Raises:
        ValueError if degree constraints (above) are violated or cycles exist outside depots.
        """
        Nc   = self.my_VRP.num_cust
        START, END = Nc, Nc + 1

        # ---- 1) normalize input to (u,v) edges ----
        raw = out_sol
        if isinstance(raw, dict):
            names = [k for k, v in raw.items() if v]
        elif isinstance(raw, (set, list, tuple)):
            names = list(raw)
        else:
            raise TypeError("all_actions_inclumbent must be dict/set/list/tuple")

        edge_re = re.compile(r"^act_(\-?\d+)_(\-?\d+)$")
        edges = []
        for a in names:
            if isinstance(a, str):
                m = edge_re.match(a.strip())
                if not m:
                    continue
                u, v = int(m.group(1)), int(m.group(2))
            elif isinstance(a, (tuple, list)) and len(a) == 2:
                u, v = int(a[0]), int(a[1])
            else:
                continue
            edges.append((u, v))

        if not edges:
            return []

        # ---- 2) build adjacency with depot exceptions ----
        succ = defaultdict(list)  # u -> [v,...]
        pred = defaultdict(list)  # v -> [u,...]

        for u, v in edges:
            succ[u].append(v)
            pred[v].append(u)

        # Degree checks with START/END exceptions
        for u, vs in succ.items():
            if u != START and len(vs) > 1:
                raise ValueError(f"Node {u} has multiple successors {vs} (only START may have many).")
        for v, us in pred.items():
            if v != END and len(us) > 1:
                raise ValueError(f"Node {v} has multiple predecessors {us} (only END may have many).")

        # ---- 3) assemble routes: one per arc out of START ----
        routes = []
        visited = set()

        # Direct START->END (empty vehicle) is allowed
        starts = succ.get(START, [])
        # De-duplicate if duplicates exist (shouldn't, but cheap safety)
        starts = list(dict.fromkeys(starts))

        def next_of(x):
            vs = succ.get(x, [])
            if not vs:
                return None
            # for non-START, we validated len <= 1
            return vs[0] if x != START else None  # never called for START in traversal

        for first in starts:
            route = [START, first]
            cur = first
            while cur != END:
                if cur in visited:
                    raise ValueError(f"Cycle detected through node {cur}.")
                visited.add(cur)

                # must have 0 or 1 successor (by validation)
                vs = succ.get(cur, [])
                if not vs:
                    # route terminates prematurely (no arc to END)
                    break
                if len(vs) > 1:
                    # shouldn’t happen due to earlier check
                    raise ValueError(f"Internal node {cur} has multiple successors {vs}.")
                nxt = vs[0]
                route.append(nxt)
                cur = nxt

            # Only keep proper depot-to-depot routes
            if route[0] == START and route[-1] == END:
                routes.append(route)

        # ---- 4) (optional) include stray chains not connected to START ----
        # If your incumbent may contain partial paths not starting at START,
        # you can try to collect them as well. Typically in VRP they shouldn’t exist.
        # Uncomment if desired:
        #
        # for u, vs in succ.items():
        #     if u == START:
        #         continue
        #     if u not in pred:  # no predecessor -> chain start
        #         r = [u]
        #         cur = u
        #         while True:
        #             vs2 = succ.get(cur, [])
        #             if not vs2:
        #                 break
        #             cur = vs2[0]
        #             r.append(cur)
        #         routes.append(r)

       
        self.my_paths=routes

        
    def check_feas_and_cost(self):

        total_cost=0
        target_cost=self.history_dict['OUR_ilp_objective']
        num_cust=self.my_VRP.num_cust
        dem_full=self.my_VRP.dem_full
        early=self.my_VRP.early_start
        late=self.my_VRP.late_start
        dist_plus_service=self.my_VRP.dist_mat_full
        covered_cust=np.zeros(num_cust)
        for my_path in self.my_paths:
                #cur_loc=path_in[0]
            path_len=len(my_path)
            cur_time_rem=np.inf
            cur_cap_rem=self.my_VRP.vehicle_capacity
            if my_path[0]!=num_cust:
                input('error here does not start withdepot')
            if my_path[path_len-1]!=num_cust+1:
                input('error here does not end withdepot')
            for i in range(0,path_len-1):
                cur_loc=my_path[i]
                next_loc=my_path[i+1]
                my_act='act_'+str(cur_loc)+'_'+str(next_loc)
                if my_act not in self.history_dict['output_ilp_solution']:
                    input('error here not present in solution')
                cur_cap_rem-=dem_full[cur_loc]
                cur_time_rem-=dist_plus_service[cur_loc,next_loc]
                if cur_cap_rem<dem_full[next_loc]-0.0001:
                    input('error here capacity disobeyed in solution')
                if next_loc<num_cust-0.5 and cur_time_rem<late[next_loc]-0.0001:
                    input('error here time disobeyed in solution')
                if  next_loc<num_cust-0.5 and cur_time_rem>early[next_loc]:
                    cur_time_rem=early[next_loc]
                if i>0 and cur_loc>num_cust-0.5:
                    input('depot in wrong spot')
                if cur_loc<num_cust-0.5 and covered_cust[cur_loc]==1:
                    input('covered more than once')
                if cur_loc<num_cust-0.5:
                    covered_cust[cur_loc]=1
                total_cost+=self.action_2_cost[my_act]
        if np.min(covered_cust)<0.5:
            input('not lining up')
        if abs(total_cost-target_cost)>0.001:
            input('error in cost')
        print(' ALL CLEAN EVERYTHING MATCHES UP')