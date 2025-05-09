
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
from scipy.sparse import csr_matrix
from lower_bound_LP_milp import lower_bound_LP_milp
#from exper_lower_bound_LP_MILP import lower_bound_LP_milp
import pulp
from compressor import compressor
#from experimental_compressor_additive import compressor
#from projector import projector
#from experimental_projector_simp import projector
from experimental_projector_simp_eq import projector
#from  import projector
#from experimental_projector_simp_no_neg import projector

#from experimental_projector_simp_no_neg_w_removal import projector
from baseline_solver import baseline_solver
import json
class full_solver:

    def __init__(self,full_input_dict,jy_opt,output_file_path):
        print('type(full_input_dict)')
        print(type(full_input_dict))
        self.jy_opt=jy_opt
        self.output_file_path=output_file_path
        self.full_input_dict=full_input_dict
        #self.all_delta:  list of the ids of all delta terms 
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
                
        self.graph_names=full_input_dict['allGraphNames']
        self.TOT_time_component_lps=dict()
        for h in self.graph_names:
            self.TOT_time_component_lps[h]=0

        
        self.delta_name_2_ub=full_input_dict['delta_name_2_ub']
        self.delta_name_2_lb=full_input_dict['delta_name_2_lb']
        self.ineq_replaced_by_lb_ub=full_input_dict['ineq_replaced_by_lb_ub']

        self.history_dict=dict()
        self.history_dict['lblp_lower']=[]
        self.history_dict['prob_sizes_at_start']=[]
        self.history_dict['prob_sizes_mid']=[]
        self.history_dict['did_compress']=[]
        self.history_dict['lp_time_compress']=[]
        self.history_dict['lp_time_project']=[]
        self.history_dict['lp_time_LB']=[]
        self.history_dict['ilp_time']=[]
        self.history_dict['lp_value_project']=[]
        self.history_dict['lp_value_compress']=[]
        self.history_dict['sum_lp_value_project']=[]
        self.history_dict['sum_lp_time_project']=[]
        self.history_dict['history_of_graphs_by_iter']=[]
        
        self.apply_complete_algorithm()

    def apply_splitting(self):
        did_split=False
        #input('before splitting')
        count_prior_split=dict()
        count_after_split=dict()
        objective_componentLps=dict()
        time_component_lps=dict()
        for h in self.graph_names:
            print('Starting SPlit+'+h)
            my_proj=projector(self,h)
            print('DONE  SPlit+'+h)

            objective=my_proj.lp_objective
            objective_componentLps[h]=my_proj.lp_objective
            time_component_lps[h]=my_proj.lp_time
            self.TOT_time_component_lps[h]=self.TOT_time_component_lps[h]+my_proj.lp_time
            count_prior_split[h]=len(set(self.graph_node_2_agg_node[h].values()))
            if objective>0.00000001:#%self.jy_opt['epsilon']:
                self.graph_node_2_agg_node[h]=my_proj.NEW_node_2_agg_node
                did_split=True
            count_after_split[h]=len(set(self.graph_node_2_agg_node[h].values()))

        print('time_component_lps')
        print(time_component_lps)
        #print('count_prior_split')
        #print(count_prior_split)
        #print('count_after_split')
        #print(count_after_split)
        #input('don splits')
        return did_split,objective_componentLps,time_component_lps
    
    def count_size(self,supress_output=True):
        my_count_size=dict()
        if supress_output==False:
            print('Graph Sizes (in terms of number of compressed nodes) are below:  ')
        #sz_vec=[]
        for h in self.graph_names:
            my_count_size[h]=len(set(self.graph_node_2_agg_node[h].values()))
            if supress_output==False:
                print('h:   '+h+' is of size:   '+str(my_count_size[h]))
            #sz_vec.append(h)
        return my_count_size


    def ApplyCompresssion(self):
        self.my_compressor=compressor(self)
        self.graph_node_2_agg_node=self.my_compressor.NEW_graph_node_2_agg_node
        compress_lp_time=self.my_compressor.lp_time
        compress_lp_val=self.my_compressor.lp_objective
        return [compress_lp_time,compress_lp_val]
    
    def prepare_ILP_solution(self):
        
        if self.jy_opt['do_ilp']>0.5:
            my_ilp_sol=self.my_lower_bound_ILP.milp_solution
            out_sol=dict()
            for my_delta in self.all_delta:
                if my_delta.startswith('delta_timeRem'):
                    print(my_delta+"  "+str(my_ilp_sol[my_delta]))
                out_sol[my_delta]=my_ilp_sol[my_delta]

            tot_cost=0
            for my_act in self.all_actions:
                out_sol[my_act]=my_ilp_sol[my_act]
                if my_ilp_sol[my_act]>0.5:
                    print(my_act)
                    tot_cost=tot_cost+self.action_2_cost[my_act]
            print('tot_cost')
            print(tot_cost)
            for my_prim in self.all_primitive_vars:
                if my_prim in my_ilp_sol:
                    out_sol[my_prim]=my_ilp_sol[my_prim]
            self.history_dict['jy_opt']=self.jy_opt
            self.history_dict['output_ilp_solution']=out_sol

        with open(self.output_file_path, 'w') as file:
            json.dump(self.history_dict, file)

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
        incumbant_lp=-np.inf

        #self.count_size(False)
        #input('---')
        #self.my_lower_bound_LP=lower_bound_LP_milp(self,self.graph_node_2_agg_node,True,False)
        #input('ready')
        iter=0
        use_compression=self.jy_opt['use_compression']
        did_compress_call=False
        if (self.jy_opt['in_demo_mode']==True):
            input('Press enter about to start the algorithm')
        tot_lplb_time=0
        tot_proj_lp_time=0
        tot_comp_lp_time=0
        
        self.actions_ignore=self.all_actions_not_source_sink_connected
        self.all_actions_ever_seen=set([])#set(self.all_non_null_action.copy())#set()
        #self.all_actions_ever_seen=set(self.all_non_null_action.copy())#set()
        while iter<self.jy_opt['max_iterations_loop_compress_project']:
            self.time_list_outer=dict()
            iter=iter+1
            t1=time.time()

            prob_sizes_at_start=self.count_size()
            self.time_list_outer['part0']=time.time()-t1
            print('lower bound computing ')
            t1=time.time()
            self.my_lower_bound_LP=lower_bound_LP_milp(self,self.graph_node_2_agg_node,False,False)
            self.time_list_outer['part0.5']=time.time()-t1
            print('DONE lower bound computing ')
            
            for p in self.all_non_null_action:
                if self.my_lower_bound_LP.lp_primal_solution[p]>0:
                    self.all_actions_ever_seen.add(p)
            
            #self.THINK_get_p_h_mapping()
            self.actions_ignore=self.my_lower_bound_LP.new_actions_ignore.copy()
            t1=time.time()
            lblp_time=self.my_lower_bound_LP.lp_time
            new_lp_value=self.my_lower_bound_LP.lp_objective
            
            did_compress_call=False
            compress_lp_time=np.nan
            compress_lp_val=np.nan
            if new_lp_value<incumbant_lp-0.01:
                print('new_lp_value')
                print(new_lp_value)
                print('incumbant_lp')
                print(incumbant_lp)
                input('error here ')
            if use_compression==False:
                incumbant_lp=new_lp_value
            did_split=True
            proj_objective_componentLps=dict()
            proj_time_component_lps=dict()
            for h in self.graph_names:
                proj_objective_componentLps[h]=np.nan
                proj_time_component_lps[h]=np.nan
            self.time_list_outer['part1']=time.time()-t1
            t1=time.time()
            if incumbant_lp<new_lp_value-self.jy_opt['min_inc_2_compress']: #and iter>0:
                #self.count_size()
                #input('starting compression ')
                if self.jy_opt['use_classic_compress']<0.5:
                    print('Staritng COmpression ')

                    [compress_lp_time,compress_lp_val]=self.ApplyCompresssion()
                    print('DOEN Staritng COmpression ')

                else:
                    self.graph_node_2_agg_node=self.my_lower_bound_LP.NAIVE_graph_node_2_agg_node
                #self.count_size()
                did_compress_call=True
                #input('done compression ')
                incumbant_lp=new_lp_value
            #else:
            if self.jy_opt['restore_after_each_step']>0.5:
                self.split_based_init()

            self.time_list_outer['part2']=time.time()-t1
            t1=time.time()
            this_prob_sizes_mid=self.count_size()
            [did_split,proj_objective_componentLps,proj_time_component_lps]=self.apply_splitting()
                
                #if did_split==False:
                #    print('braeking do to no split')
                #    print('new_lp_value=  '+str(new_lp_value))
                #    break
            


            self.time_list_outer['part3']=time.time()-t1
            t1=time.time()

            self.history_dict['lblp_lower'].append(new_lp_value)
            self.history_dict['prob_sizes_at_start'].append(prob_sizes_at_start)
            self.history_dict['prob_sizes_mid'].append(this_prob_sizes_mid)
            self.history_dict['did_compress'].append(did_compress_call)
            self.history_dict['lp_time_compress'].append(compress_lp_time)
            self.history_dict['lp_time_project'].append(proj_time_component_lps)
            self.history_dict['lp_time_LB'].append(lblp_time)
            self.history_dict['lp_value_project'].append(proj_objective_componentLps)
            self.history_dict['lp_value_compress'].append(compress_lp_val)
            self.history_dict['sum_lp_value_project'].append(sum(proj_objective_componentLps.values()))
            self.history_dict['sum_lp_time_project'].append(sum(proj_time_component_lps.values()))
            if self.jy_opt['save_graph_each_iter']>0.5:
                self.augment_history_graphs()
            self.time_list_outer['part4']=time.time()-t1
            t1=time.time()
            print('-----')
            print('-----')
            print('-----')
            print('ITER FINISHE:  '+str(iter))
            print('new_lp_value=  '+str(new_lp_value))
            print('did_compress_call:  '+str(did_compress_call))
            print('lp project time '+str(self.history_dict['sum_lp_time_project'][-1]))
            print('lp compress time '+str(self.history_dict['lp_time_compress'][-1]))
            print('lplb time '+str(self.history_dict['lp_time_LB'][-1]))
            print('sum_lp_value_project '+str(self.history_dict['sum_lp_value_project'][-1]))
            self.count_size(False)
            print('prob_sizes_at_start')
            print(prob_sizes_at_start)
            print('this_prob_sizes_mid')
            print(this_prob_sizes_mid)
            print('-----')
            print('-----')
            print('-----')
            #input('---')
            print('self.time_list_outer')
            print(self.time_list_outer)
            #input('---')
            
            if did_compress_call==False and did_split==False:
                print('breaking do to no split')
                break
        #input('done entire call')
        if use_compression>0.5:#1>0:#did_compress_call==False and use_compression==True and iter>0:
            #input('here')
            print('Doing final Clean up operations')
            self.my_lower_bound_LP=lower_bound_LP_milp(self,self.graph_node_2_agg_node,False,False)
            prob_sizes_at_start=self.count_size()

            lblp_time=self.my_lower_bound_LP.lp_time
            new_lp_value=self.my_lower_bound_LP.lp_objective
            did_compress_call=True
            proj_objective_componentLps=dict()
            proj_time_component_lps=dict()
            compress_lp_time=0
            compress_lp_val=np.nan
            for h in self.graph_names:
                proj_objective_componentLps[h]=np.nan
                proj_time_component_lps[h]=np.nan
            if self.jy_opt['use_classic_compress_last']<0.5:
                [compress_lp_time,compress_lp_val]=self.ApplyCompresssion()
            else:
                self.graph_node_2_agg_node=self.my_lower_bound_LP.NAIVE_graph_node_2_agg_node
            
            self.history_dict['lblp_lower'].append(new_lp_value)
            self.history_dict['prob_sizes_at_start'].append(prob_sizes_at_start)
            self.history_dict['did_compress'].append(did_compress_call)
            self.history_dict['lp_time_compress'].append(compress_lp_time)
            self.history_dict['lp_time_project'].append(proj_time_component_lps)
            self.history_dict['lp_time_LB'].append(lblp_time)
            self.history_dict['lp_value_project'].append(proj_objective_componentLps)
            self.history_dict['lp_value_compress'].append(compress_lp_val)
            self.history_dict['sum_lp_value_project'].append(sum(proj_objective_componentLps.values()))
            self.history_dict['sum_lp_time_project'].append(sum(proj_time_component_lps.values()))
            if self.jy_opt['save_graph_each_iter']>0.5:
                self.augment_history_graphs()
            print('-----')
            print('-----')
            print('-----')
            print('FINAL CLEANUP FINISHE:  ')
            print('new_lp_value=  '+str(new_lp_value))
            print('did_compress_call:  '+str(did_compress_call))
            print('lp project time '+str(self.history_dict['sum_lp_time_project'][-1]))
            print('lp compress time '+str(self.history_dict['lp_time_compress'][-1]))
            print('lplb time '+str(self.history_dict['lp_time_LB'][-1]))
            print('sum_lp_value_project '+str(self.history_dict['sum_lp_value_project'][-1]))

            self.count_size(False)
            print('prob_sizes_at_start')
            print(prob_sizes_at_start)
            print('-----')
            print('-----')
            print('-----')
            
        if (self.jy_opt['in_demo_mode']==True):
            input('Press enter about to start the acutal ILP')

        print('--JUST BEFORE CALLIN ILP DONE-')
        self.count_size(False)
        print('sum(self.history_dict[lp_time_LB])')
        print(sum(self.history_dict['lp_time_LB']))
        print('nana  sum lp_time_compress')
        print(np.nansum(np.array(self.history_dict['lp_time_compress'])))
        print('sum(self.history_dict[sum_lp_time_project].values())')
        print(np.nansum(np.array(self.history_dict['sum_lp_time_project'])))
        print('TOT_time_component_lps')


        if self.jy_opt['add_all_ng_split_at_end']>0.5:
            ng_source=self.h_2_source_id['ngGraph']
            ng_sink=self.h_2_sink_id['ngGraph']
            
            all_ng_non_nodes=set(self.graph_node_2_agg_node['ngGraph'].keys())-set([ng_source,ng_sink])
            counter=0
            #print('self.graph_node_2_agg_node[ng_graph][ng_source]')
            #print(self.graph_node_2_agg_node['ngGraph'][ng_source])
            #print('self.graph_node_2_agg_node[ng_graph][ng_sink]')
            #print(self.graph_node_2_agg_node['ngGraph'][ng_sink])
            #input('--')
            for i in all_ng_non_nodes:
                self.graph_node_2_agg_node['ngGraph'][i]=str(counter)
                counter=counter+1
        
        if self.jy_opt['do_split_based_init']>0.5:
            #print('OLD STUFF')

            #self.count_size(False)
            #print('OLD STUFF')
            self.split_based_init()
            #print('NEW STUFF')

            self.count_size(False)
            #print('NEW STUFF')

            #input('--')
        self.count_size(False)
        self.history_dict['final_sizes']=self.count_size()
        self.history_dict['final_graph_node_2_agg_node']=self.graph_node_2_agg_node
        if self.jy_opt['do_ilp']>0.5:
            print('starting ILP')

            self.my_lower_bound_ILP=lower_bound_LP_milp(self,self.graph_node_2_agg_node,True,False)
            new_Ilp_value=self.my_lower_bound_ILP.milp_solution_objective_value
            self.history_dict['OUR_ilp_objective']=new_Ilp_value
            self.history_dict['OUR_MIP_Lower_Bound']=self.my_lower_bound_ILP.MIP_lower_bound
            self.history_dict['OUR_ilp_time']=self.my_lower_bound_ILP.milp_time
            if self.jy_opt['use_gurobi']>0.5:
                self.history_dict['OUR_gurobi_MILP_str']=self.my_lower_bound_ILP.gurobi_MILP_str
            print('final solution objective')
            print(new_Ilp_value)
            if self.jy_opt['run_baseline']==True:
                print('running baseline')
                if (self.jy_opt['in_demo_mode']==True):
                    input('Press enter about to start the running of the baseline ILP')
                my_base=baseline_solver(self,True,False)
                self.history_dict['BASE_ILP_sol_obj']=my_base.milp_solution_objective_value
                self.history_dict['BASE_milp_solution']=my_base.milp_solution
                self.history_dict['BASE_milp_time']=my_base.milp_time
                self.history_dict['BASE_MIP_lower_bound'] = my_base.MIP_lower_bound#model.ObjBound

            print('--ALL DONE-')
            print('self.my_lower_bound_ILP.milp_time')
            print(self.my_lower_bound_ILP.milp_time)
            print('self.my_lower_bound_ILP')
            print(new_Ilp_value)
            print('sum(self.history_dict[lp_time_LB])')
            print(sum(self.history_dict['lp_time_LB']))
            print('nana  sum lp_time_compress')
            print(np.nansum(np.array(self.history_dict['lp_time_compress'])))
            print('sum(self.history_dict[sum_lp_time_project].values())')
            print(np.nansum(np.array(self.history_dict['sum_lp_time_project'])))
            print('TOT_time_component_lps')
            print(self.TOT_time_component_lps)
        if self.jy_opt['run_baseline']==True:

            print('my_base.milp_time')
            print(my_base.milp_time)
            #self.history_dict['my_base_milp_time']=my_base.milp_time
            #self.history_dict['my_base_milp_solution']=my_base.milp_solution
            #self.history_dict['milp_solution_objective_value']=milp_solution_objective_value
        self.prepare_ILP_solution()



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
                #print('old_name')
                #print(old_name)
                #print('init_name')
                #print(init_name)
                #print('new_name')
                #print(new_name)
                self.NEW_graph_node_2_agg_node[h][i]=new_name
        self.graph_node_2_agg_node=self.NEW_graph_node_2_agg_node
    