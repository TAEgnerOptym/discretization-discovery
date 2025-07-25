import xpress as xp

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
import re
from scipy.sparse import csr_matrix
import pulp
import sys
import itertools
import random
import bisect


from pre_process.naive_pre import *
sys.path.append("exper_ideas")
from jy_active_set_lp import jy_active_set_lp
from jy_active_set_lp import jy_active_set_lp_primal_dual
from warm_start_lp import warm_start_lp
from warm_start_lp import forbidden_variables_loop
from warm_start_lp import forbidden_variables_loop_dual
from warm_start_lp import warm_start_lp_using_class
from warm_start_lp import warm_start_lp_using_class_gurobi

from solve_gurobi_lp import solve_gurobi_lp
from solve_gurobi_lp import solve_gurobi_lp_bounds
from solve_gurobi_lp import solve_gurobi_milp
from solve_gurobi_lp import solve_gurobi_milp_bounds

class projector_on_lb:


    def __init__(self,full_prob,dict_2_action_ignore):
        t1=time.time()

        self.times_lp_times=dict()
        self.full_prob=full_prob
        self.MF=full_prob
        full_input_dict=full_prob.full_input_dict
        self.actions_ignore=full_prob.actions_ignore
        self.dict_pred_gain=None
        self.dict_2_action_ignore=dict_2_action_ignore
        self.vars_names_ignore=self.actions_ignore.copy()
        
        
        self.all_delta=full_input_dict['allDelta']
        #self.all_graph_names:  names of all of the graphs
        self.all_graph_names=full_input_dict['allGraphNames']

        self.graph_node_2_agg_node=dict()
        for g in full_prob.graph_name_2_nodes:
            mykeys=full_prob.graph_name_2_nodes[g]
            tmp = {k: k for k in mykeys}

            self.graph_node_2_agg_node[g]=tmp
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
        self.graph_names=full_input_dict['allGraphNames']
        self.OPT_use_psi=False
        self.OPT_do_ilp=False
        self.times_lp_times['prior']=time.time()-t1

        self.construct_LB_or_ILP(self.OPT_use_psi,self.OPT_do_ilp)
        t1=time.time()
  
        self.filter_constraints()
        self.times_lp_times['filetering']=time.time()-t1

        if 'var_int_names_fancy_branch' in self.full_prob.D:
            big_M=2000000
            
            for var_penalty in self.full_prob.D['var_cont_names_fancy_branch']:
                self.dict_var_name_2_obj[var_penalty]=big_M
            
        
        self.call_gurobi_solver()
        t1=time.time()
        self.get_ij_poss_active()
        self.times_lp_times['allPostOpps4']=time.time()-t1
        if 1<0:
            t1=time.time()
            self.naive_compress_get_pi_by_h_node()
            self.times_lp_times['allPostOpps3']=time.time()-t1
            t1=time.time()
            self.naive_compress_make_f_2_new_f()
            self.times_lp_times['allPostOpps2']=time.time()-t1
            t1=time.time()
            self.Naive_make_i_2_new_f()
            self.times_lp_times['allPostOpps1']=time.time()-t1
        else:
            self.make_new_splits()

    def make_agg_node_2_nodes(self):
        self.agg_node_2_nodes = {
            h: {
                f: { i for i, f_val in self.graph_node_2_agg_node[h].items() if f_val == f }
                for f in set(self.graph_node_2_agg_node[h].values())
            }
            for h in self.graph_names
        }

    def make_edge_fg_2_ij_reverse(self):
        self.h_fg_2_ij = {}
        self.h_ij_2_fg = {}

        for h in self.graph_names:
            # Precompute the mapping from each edge tup_ij to its aggregated pair tup_fg.
            h_ij_2_fg_h = {
                tup_ij: (
                    self.graph_node_2_agg_node[h][tup_ij[0]],
                    self.graph_node_2_agg_node[h][tup_ij[1]]
                )
                for tup_ij in self.hij_2_P[h].keys()
            }
            self.h_ij_2_fg[h] = h_ij_2_fg_h

            # Group edges by their aggregated pair.
            edge_group = defaultdict(set)
            for tup_ij, tup_fg in h_ij_2_fg_h.items():
                edge_group[tup_fg].add(tup_ij)
            self.h_fg_2_ij[h] = dict(edge_group)


    def make_h_fg_2_p_reverse(self):
        self.h_q_2_fg=dict()
        self.h_fg_2_q=dict() #given h,fg gives the equvelence_class
        self.h_q_2_q_id=dict()
        for h in self.graph_names:
            if 1<0:
                all_fg_edges=self.h_fg_2_ij[h]
                self.h_fg_2_q[h]=dict()
                self.h_q_2_fg[h]=dict()
                self.h_q_2_q_id[h]=dict()
                count=0
                for  tup_fg in all_fg_edges:
                    my_set=set([])
                    for tup_ij in  self.h_fg_2_ij[h][tup_fg]:
                        for p in self.hij_2_P[h][tup_ij]:
                            my_set.add(p)
                    my_tup_pq=tuple(sorted(list(my_set)))
                    self.h_fg_2_q[h][tup_fg]=my_tup_pq
                    
                    if my_tup_pq not in self.h_q_2_fg[h]:
                        self.h_q_2_fg[h][my_tup_pq]=set([])
                        self.h_q_2_q_id[h][my_tup_pq]=tuple([h,count])
                        count=count+1
                    self.h_q_2_fg[h][my_tup_pq].add(tup_fg)
            else:
                all_fg_edges = self.h_fg_2_ij[h]
                self.h_fg_2_q[h] = dict()
                self.h_q_2_fg[h] = dict()
                self.h_q_2_q_id[h] = dict()
                count = 0

                for tup_fg in all_fg_edges:
                    # Collect all sets of p-values quickly
                    sets_to_union = (self.hij_2_P[h][tup_ij] for tup_ij in self.h_fg_2_ij[h][tup_fg])
                    # Use set union in bulk
                    my_set = set().union(*sets_to_union)
                    
                    my_tup_pq = tuple(sorted(my_set))
                    self.h_fg_2_q[h][tup_fg] = my_tup_pq

                    if my_tup_pq not in self.h_q_2_fg[h]:
                        self.h_q_2_fg[h][my_tup_pq] = set()
                        self.h_q_2_q_id[h][my_tup_pq] = (h, count)
                        count += 1
                    self.h_q_2_fg[h][my_tup_pq].add(tup_fg)

    def make_mappings(self):
        t1=time.time()
        self.make_agg_node_2_nodes()
        self.times_lp_times['make_mappings_1']=time.time()-t1
        t1=time.time()
        self.make_edge_fg_2_ij_reverse()
        self.times_lp_times['make_mappings_2']=time.time()-t1
        t1=time.time()
        self.make_h_fg_2_p_reverse()
        self.times_lp_times['make_mappings_3']=time.time()-t1


    def help_construct_LB_make_vars(self):
        t1=time.time()
        use_psi=self.OPT_use_psi
        do_ilp=self.OPT_do_ilp
        self.dict_var_name_2_is_binary=defaultdict(int)
        self.dict_var_name_2_is_integer=defaultdict(int)

        self.names_binary=[]
        if use_psi==True and do_ilp==True:
            for var_name in self.all_primitive_vars:
                self.dict_var_name_2_obj[var_name]=0
                self.dict_var_name_2_is_binary[var_name]=1
        self.times_lp_times['help_construct_LB_make_vars_1']=time.time()-t1
        t1=time.time()
        my_x_typr='Binary'
        if do_ilp==False or use_psi==True:
            my_x_typr='Continuous'
        for var_name in self.all_non_null_action:
            self.dict_var_name_2_obj[var_name]=self.action_2_cost[var_name]
        self.times_lp_times['help_construct_LB_make_vars_2']=time.time()-t1
        t1=time.time()
        #for 
        if do_ilp==True or use_psi==False:
            for var_name in self.all_non_null_action:
                self.dict_var_name_2_is_binary[var_name]=1
        self.times_lp_times['help_construct_LB_make_vars_3']=time.time()-t1
        t1=time.time()
        for var_name in self.all_delta:
            self.dict_var_name_2_obj[var_name]=0
        self.times_lp_times['help_construct_LB_make_vars_4']=time.time()-t1
        t1=time.time()
        for h in self.graph_names:
            for tup_fg in self.h_fg_2_ij[h]:
                
                f=tup_fg[0]
                g=tup_fg[1]
                var_name='EDGE_h='+h+'_f='+f+'_g='+g
                self.dict_var_name_2_obj[var_name]=0
                #if (self.full_prob.jy_opt['allOneBig_init']==False and self.full_prob.jy_opt['do_split_based_init']==True) and self.full_prob.jy_opt['all_vars_binary']==True:
                if self.full_prob.jy_opt['all_vars_binary']==True:
                    self.dict_var_name_2_is_integer[var_name]=1
        self.times_lp_times['help_construct_LB_make_vars_5']=time.time()-t1
        t1=time.time()
        t1 = time.time()
        all_new_entries_ignore = []
        vars_names_ignore_set = set(self.vars_names_ignore)  # For O(1) lookups
        dict_update = {}  # Collect all new entries for one bulk update
        dict_update_non_null = {}  # Collect all new entries for one bulk update

        for h in self.graph_names:
            for q in self.h_q_2_q_id[h]:
                prefix = f"fill_PQ_h={h}_q={q}_p="
                for p in q:
                    var_name = prefix + p
                    dict_update[var_name] = 0
                    
                    if p in vars_names_ignore_set:
                        all_new_entries_ignore.append(var_name)
                    if p !=self.null_action: #or (self.full_prob.jy_opt['allOneBig_init']==False and self.full_prob.jy_opt['do_split_based_init']==True):
                        dict_update_non_null[var_name] = 0
        # Single update call
        self.dict_var_name_2_obj.update(dict_update)
        if self.full_prob.jy_opt['all_vars_binary']==True:
            #print('dict_update_non_null')
            #print(dict_update_non_null)
            #input('yoyo')
            for var_name in dict_update_non_null:
                self.dict_var_name_2_is_binary[var_name]=1
        # Final time record
        self.times_lp_times['help_construct_LB_make_vars_6'] = time.time() - t1

        t1=time.time()
        self.vars_names_ignore=self.vars_names_ignore+all_new_entries_ignore
        #if self.full_prob.jy_opt['all_vars_binary']==True:
        #    for var_name in set(self.dict_var_name_2_obj)-set(self.all_delta):
        #        self.dict_var_name_2_is_binary[var_name]=1
        self.times_lp_times['help_construct_LB_make_vars_7']=time.time()-t1

    def help_construct_UB_LB_con(self):
        
        t1=time.time()
        for exog_name in self.exog_name_2_rhs:
            self.dict_con_name_2_LB[exog_name]=self.exog_name_2_rhs[exog_name]
            
        if self.OPT_use_psi==True and self.OPT_do_ilp==True:
            for con_name in self.all_integCon:
                self.dict_con_name_2_eq[con_name]=0
        for h in self.graph_names:
            this_sink=self.graph_node_2_agg_node[h][self.h_2_sink_id[h]]
            this_source=self.graph_node_2_agg_node[h][self.h_2_source_id[h]]
            nodes_use=set(self.agg_node_2_nodes[h])-set([this_sink,this_source])
            my_prefix='flow_in_out_h='+h+"_n="
            new_entries = {my_prefix + n: 0 for n in nodes_use}
            self.dict_con_name_2_eq.update(new_entries)
        for h in self.graph_names:
            
            for q in self.h_q_2_fg[h]:
                con_name='equiv_class='+h+"_q="+str(q)
                self.dict_con_name_2_eq[con_name]=0
        for h in self.graph_names:
            prefix='action_match_h='+h+"_p="
            new_entries = {prefix + p: 0 for p in self.all_non_null_action}
            self.dict_con_name_2_eq.update(new_entries)
        t2=time.time()
        
    def construct_constraints_exog(self):
        for v_con in self.delta_con_2_contrib:
            var_name=v_con[0]
            con_name=v_con[1]
            self.dict_var_con_2_lhs_exog[tuple([var_name,con_name])]=self.delta_con_2_contrib[v_con]
        
        for v_con in self.action_con_2_contrib:
            var_name=v_con[0]
            con_name=v_con[1]
            self.dict_var_con_2_lhs_exog[tuple([var_name,con_name])]=self.action_con_2_contrib[v_con]
    
    def construct_constraints_prim(self):
        for v_con in self.prim_integCon_2_contrib:
            var_name=v_con[0]
            con_name=v_con[1]
            self.dict_var_con_2_lhs_eq[tuple([var_name,con_name])]=self.prim_integCon_2_contrib[v_con]
        for v_con in self.action_integCon_2_contrib:
            var_name=v_con[0]
            con_name=v_con[1]
            self.dict_var_con_2_lhs_eq[tuple([var_name,con_name])]=self.action_integCon_2_contrib[v_con]
    
    def construct_constraints_flow_in_out(self):
        for h in self.graph_names:
            my_sink=self.graph_node_2_agg_node[h][self.h_2_sink_id[h]]
            my_source=self.graph_node_2_agg_node[h][self.h_2_source_id[h]]
            for e in self.h_fg_2_ij[h]:
                f=e[0]
                g=e[1]
                var_name='EDGE_h='+h+'_f='+f+'_g='+g
                
                if f==g: 
                    continue
                if f!=my_source:
                    con_name_in='flow_in_out_h='+h+"_n="+f
                    self.dict_var_con_2_lhs_eq[tuple([var_name,con_name_in])]=1
                if g!=my_sink:
                    con_name_out='flow_in_out_h='+h+"_n="+g
                    self.dict_var_con_2_lhs_eq[tuple([var_name,con_name_out])]=-1


    def construct_constraints_actions_match_compact(self):
    # First set: iterate over every graph h and every non-null action in all_non_null_action.
        constraints_1 = {
            (p, f"action_match_h={h}_p={p}"): -1
            for h in self.graph_names
            for p in self.all_non_null_action
        }

        # Second set: iterate over each graph h, each q in h_q_2_fg[h], and then each p in q,
        # but precompute the fixed prefix for the variable name and constraint name so that the inner loop
        # over p (which is very large) does only the minimal string concatenation.
        constraints_2 = {
            (prefix_var + p, prefix_cons + p): 1
            for h in self.graph_names
            for q in self.h_q_2_fg[h]
            for prefix_var, prefix_cons in [(f"fill_PQ_h={h}_q={q}_p=", f"action_match_h={h}_p=")]
            for p in q if p != self.null_action
        }

        # Update the existing dictionary (without removing existing entries).
        self.dict_var_con_2_lhs_eq.update(constraints_1)
        self.dict_var_con_2_lhs_eq.update(constraints_2)
        
    def construct_constraints_actions_match_flow(self):
        # Build constraints from actions (p in q) with value -1.
        
        constraints_from_actions=dict()
        for h in self.graph_names:
            for q in self.h_q_2_fg[h]:
                prefix = f"fill_PQ_h={h}_q={q}_p="
                cons = f"equiv_class={h}_q={q}"
                # For each p in q, simply concatenate the precomputed prefix with p.
                constraints_from_actions.update({(prefix + p, cons): -1 for p in q})




        # Build constraints from edges (e in self.h_q_2_fg[h][q]) with value 1.
        constraints_from_edges = {
            (f"EDGE_h={h}_f={e[0]}_g={e[1]}", f"equiv_class={h}_q={q}"): 1
            for h in self.graph_names
            for q in self.h_q_2_fg[h]
            for e in self.h_q_2_fg[h][q]
        }
        
        # Combine them with existing content, ensuring no keys are overwritten.
        new_entries = {**constraints_from_actions, **constraints_from_edges}
        self.dict_var_con_2_lhs_eq.update(new_entries)


    def construct_LB_or_ILP(self,use_psi,do_ilp):
        self.OPT_use_psi=use_psi
        self.OPT_do_ilp=do_ilp
        #t1=time.time()
        self.make_mappings()
        #self.times_lp_times['make_mappings']=time.time()-t1
        #t1=time.time()


        self.pulp_all_vars=set()
        self.dict_var_name_2_obj=dict()
        self.dict_var_name_2_is_binary=dict()
        self.dict_var_con_2_lhs_exog=dict()
        self.dict_var_con_2_lhs_eq=dict()
        self.dict_con_name_2_LB=dict()
        self.dict_con_name_2_eq=dict()
        self.help_construct_LB_make_vars()
        #self.times_lp_times['help_construct_LB_make_vars']=time.time()-t1
        #t1=time.time()
        self.help_construct_UB_LB_con()
        #self.times_lp_times['help_construct_UB_LB_con']=time.time()-t1
        t1=time.time()
        self.construct_constraints_exog()
        self.times_lp_times['construct_constraints_exog']=time.time()-t1
        t1=time.time()

        self.construct_constraints_flow_in_out()
        self.times_lp_times['construct_constraints_flow_in_out']=time.time()-t1
        t1=time.time()

        self.construct_constraints_actions_match_compact()
        self.times_lp_times['construct_constraints_actions_match_compact']=time.time()-t1
        t1=time.time()

        self.construct_constraints_actions_match_flow()
        self.times_lp_times['construct_constraints_actions_match_compact']=time.time()-t1
        t1=time.time()

        if use_psi==True:
            self.construct_constraints_prim()
        self.times_lp_times['construct_constraints_prim']=time.time()-t1

    def call_gurobi_solver(self):
        #input('in this call ')
        did_call_gur_warm=False
        GUR_CLASS_lp_prob=[]
        self.new_actions_ignore=None
        #if len(self.full_prob.history_dict['lp_time_compress'])<1 or  self.full_prob.jy_opt['use_julians_custom_lp_solver']<0.5:
       
        lb_use=dict()
        ub_use=dict()
        if self.full_prob.jy_opt['use_delta_in_lp']==True:
            lb_use=self.full_prob.delta_name_2_lb.copy()
            ub_use=self.full_prob.delta_name_2_ub.copy()

        for var in self.actions_ignore:
            ub_use[var]=0

        #input("mooose")
        out_solution=solve_gurobi_lp_bounds(self.dict_var_name_2_obj,
            self.CLEAN_dict_var_con_2_lhs_exog,
            self.CLEAN_dict_con_name_2_LB,
            self.CLEAN_dict_var_con_2_lhs_eq,
            self.CLEAN_dict_con_name_2_eq,lb_use,ub_use)

        
        self.lp_dual_solution=out_solution['dual_solution']
        self.lp_primal_solution=out_solution['primal_solution']
        self.lp_objective=out_solution['objective']
        self.times_lp_times['GUR_time_pre']=out_solution['time_pre']
        self.times_lp_times['GUR_time_opt']=out_solution['time_opt']
        self.times_lp_times['GUR_time_post']=out_solution['time_post']
        self.lp_time=out_solution['time_opt']
    def naive_compress_get_pi_by_h_node(self):
        self.Naive_h_f_2_dual=dict()
        self.Naive_h_f_2_dual_sig_fig=dict()
        self.Naive_h_val_2_id=dict()
        for h in self.graph_names:
            self.Naive_h_f_2_dual[h]=dict()
            self.Naive_h_f_2_dual_sig_fig[h]=dict()
            self.Naive_h_val_2_id[h]=dict()
            counter_h=0
            #print('self.lp_dual_solution')
            #print(self.lp_dual_solution)
            #matching_keys = [key for key in self.lp_dual_solution if key.startswith("flow_in_out_h="+h)]

            #print(matching_keys)
            #input('---')
            this_fg_sink=self.graph_node_2_agg_node[h][self.h_2_sink_id[h]]
            this_fg_source=self.graph_node_2_agg_node[h][self.h_2_source_id[h]]
            f_except_source_sink=set(self.agg_node_2_nodes[h])-set([this_fg_sink,this_fg_source])
            for f in f_except_source_sink:
                this_con_name='flow_in_out_h='+h+"_n="+f
                #this_con_name= this_con_name.replace(" ", "_")
                #this_con_name= this_con_name.replace("(", "_")
                #this_con_name= this_con_name.replace(")", "_")
                self.Naive_h_f_2_dual[h][f]=self.lp_dual_solution[this_con_name]
                new_val=round(self.Naive_h_f_2_dual[h][f],3)
                self.Naive_h_f_2_dual_sig_fig[h][f]=new_val
                if tuple([h,new_val]) not in self.Naive_h_val_2_id[h]:
                    self.Naive_h_val_2_id[h][tuple([h,new_val])]=counter_h
                    counter_h=counter_h+1
                
    def naive_compress_make_f_2_new_f(self):
        self.Naive_H_f_2_new_f=dict()
        for h in self.graph_names:
            self.Naive_H_f_2_new_f[h]=dict()
            this_fg_sink=self.graph_node_2_agg_node[h][self.h_2_sink_id[h]]
            this_fg_source=self.graph_node_2_agg_node[h][self.h_2_source_id[h]]
            self.Naive_H_f_2_new_f[h][this_fg_sink]=tuple([h,-2])
            self.Naive_H_f_2_new_f[h][this_fg_source]=tuple([h,-1])
            for f in self.Naive_h_f_2_dual_sig_fig[h]:
                my_dual_val=self.Naive_h_f_2_dual_sig_fig[h][f]
                my_dual_id=self.Naive_h_val_2_id[h][tuple([h,my_dual_val])]
                extra_term=self.full_prob.graph_node_2_agg_node[h][f]
                my_key=tuple([h,my_dual_id,extra_term])
                self.Naive_H_f_2_new_f[h][f]=my_key
    
    def Naive_make_i_2_new_f(self):
        self.NAIVE_graph_node_2_agg_node=dict()
        count_orig=dict()
        count_new=dict()
        for h in self.graph_names:
            self.NAIVE_graph_node_2_agg_node[h]=dict()
            count_orig[h]=len(set(self.graph_node_2_agg_node[h].values()))
            count_new[h]=len(set(self.Naive_H_f_2_new_f[h].values()))
            #print('[h,count_orig[h],count_new[h]]')
            #print([h,count_orig[h],count_new[h]])
            #print('---')
            for i in self.graph_node_2_agg_node[h]:
                f=self.graph_node_2_agg_node[h][i]
                
                
                if f not in self.Naive_H_f_2_new_f[h]:
                    print('not fuond')
                    input('error here ')
                #print('f')
                #print(f)
                my_new_name=str(self.Naive_H_f_2_new_f[h][f])
                my_new_name=my_new_name.replace(" ", "_")
                self.NAIVE_graph_node_2_agg_node[h][i]=my_new_name
        #print('self.NAIVE_graph_node_2_agg_node')
        #print(self.NAIVE_graph_node_2_agg_node)
        #input('---')
    def filter_constraints(self):
        self.ignore_set = set(self.full_prob.ineq_replaced_by_lb_ub)

        # Filter exogenous constraint contributions
        self.CLEAN_dict_var_con_2_lhs_exog = {
            (var, con): coeff
            for (var, con), coeff in self.dict_var_con_2_lhs_exog.items()
            if con not in self.ignore_set
        }

        # Filter lower bounds
        self.CLEAN_dict_con_name_2_LB= {
            con: val for con, val in self.dict_con_name_2_LB.items()
            if con not in self.ignore_set
        }

        # Filter equality constraint contributions
        self.CLEAN_dict_var_con_2_lhs_eq = {
            (var, con): coeff
            for (var, con), coeff in self.dict_var_con_2_lhs_eq.items()
            if con not in self.ignore_set
        }

        # Filter equality RHS values
        self.CLEAN_dict_con_name_2_eq = {
            con: val for con, val in self.dict_con_name_2_eq.items()
            if con not in self.ignore_set
        }


    def get_ij_poss_active(self):
        x=self.lp_primal_solution
        fg_2_val=defaultdict(float)
        f_out_val=set([])
        g_in_val=set([])
        both_val=set([])
        t1=time.time()
        for h in self.graph_names:
            for f, g in self.h_fg_2_ij[h]:
                if f==g:
                    continue
                var_name = f"EDGE_h={h}_f={f}_g={g}"
                val = x[var_name]
                #print('val')
                #print(val)
                fg_2_val[(f, g)] += val
                if val>0.000001:
                    f_out_val.add(f)
                    g_in_val.add(g)
                    both_val.add(f)
                    both_val.add(g)
        t_first=time.time()-t1
        t_sec=time.time()      
        self.maybe_mapping=dict()
        
        for h in self.graph_names:
            ij_list = self.h_ij_2_fg[h].keys()
            agg_node_map = self.graph_node_2_agg_node[h]
            hij_P = self.hij_2_P[h]
            #print('type')
            #print(type(ij_list))
            #print('type')
            #print(type(agg_node_map))
            same_group_ij = {ij for ij in ij_list if agg_node_map[ij[0]] == agg_node_map[ij[1]]}
            null_action_ij = {ij for ij in ij_list if self.null_action in hij_P[ij]}
            f_in_both_val = {ij for ij in ij_list if agg_node_map[ij[0]] in both_val}
            g_in_both_val = {ij for ij in ij_list if agg_node_map[ij[1]] in both_val}

            active_ij = same_group_ij | null_action_ij | f_in_both_val | g_in_both_val

            # Initialize maybe_mapping[h] as a defaultdict with default value 0
            self.maybe_mapping[h]=active_ij
            #self.maybe_mapping[h] =defaultdict(int)
            #for ij in active_ij:
            #    self.maybe_mapping[h][ij] = 1            
        t_sec=time.time()-t_sec
        print('[t_first,t_sec]')
        print([t_first,t_sec])
        #input('---')

    def apply_LA_branching(self):

         

        #make big range sets 

        #make power set; sets

        
    
        my_VRP=self.full_prob.D['my_VRP']
        D=self.full_prob.D
        print(my_VRP)
        Nc=my_VRP.num_cust
        self.dict_pred_gain=dict()
        [ng_neigh_by_cust_power,junk]=naive_get_LA_neigh(my_VRP,self.full_prob.jy_opt['LAB_MP_neigh_use_power'])
        [ng_neigh_by_cust_all,junk]=naive_get_LA_neigh(my_VRP,self.full_prob.jy_opt['LAB_MP_neigh_use_all'])
        G=set()
        for u in range(0,Nc):

            neighborhood = set(ng_neigh_by_cust_power[u]) | {u}
            for g in power_set(neighborhood):
                if len(g)>1:  # skip empty set and size 1 sets
                    G.add(frozenset(g))
            for i in range(1,len(ng_neigh_by_cust_all[u])):
                g=set(ng_neigh_by_cust_all[u][0:i]).union(set([u]))
                G.add(frozenset(g))
        G.add(frozenset(np.arange(0,Nc)))
        self.G=G

        self.Z_by_group = defaultdict(set)  # g → set of act_u_v
        all_actions=D['action2Cost'].keys()
        u_in_groups = defaultdict(set)
        v_not_in_groups = defaultdict(set)

        for g in G:
            for u in g:
                u_in_groups[u].add(g)
            for v in range(0,Nc+2):  # include depot if needed
                if v not in g:
                    v_not_in_groups[v].add(g)

        # Step 2: Build Z_by_group using set intersection
        for act in all_actions:
            _, u_str, v_str = act.split("_")
            u, v = int(u_str), int(v_str)

            relevant_groups = u_in_groups[u] & v_not_in_groups[v]
            for g in relevant_groups:
                self.Z_by_group[g].add(act)

        costs=D['action2Cost']
        self.pred_val_gain = {
            g: min(costs[act] for act in self.Z_by_group[g]) if self.Z_by_group[g] else float("inf")
            for g in G
        }

        #make auxiliary variables
        print('len(G)')
        print(len(G))
        print('G')
        for g in G:
            con_name_eq='NEW_BOUND_Branch_eq'+str(g)
            my_var_name='fancy_branching_var_'+str(g)
            self.dict_var_name_2_obj[my_var_name]=0
            self.dict_con_name_2_eq[con_name_eq]=0
            self.dict_pred_gain[my_var_name]=self.pred_val_gain[g]
            self.dict_var_con_2_lhs_eq[tuple([my_var_name,con_name_eq])]=1#self.delta_con_2_contrib[v_con]
            self.dict_var_name_2_is_integer[my_var_name]=1
            for my_act in self.Z_by_group[g]:
                self.dict_var_con_2_lhs_eq[tuple([my_act,con_name_eq])]=-1
            
    def make_new_splits(self):
        self.MF.jy_opt['threshold_split']=0.01
        #get max value
        self.NAIVE_graph_node_2_agg_node=dict()#self.graph_node_2_agg_node.copy()
        #max_val=max(self.graph_node_2_agg_node.values())
        
        for h in self.all_graph_names:
            self.NAIVE_graph_node_2_agg_node[h]=self.MF.my_lower_bound_LP.graph_node_2_agg_node[h].copy()
            start_value=0
            extra_string='rand_'+str(random.randint(0,100000000))+'_'
            i_2_dual=dict()
            compact_sink=self.MF.h_2_sink_id[h]
            compact_source=self.MF.h_2_source_id[h]
            non_source_sink=set(self.graph_node_2_agg_node[h])-set([compact_sink ,compact_source])

            for i_orig in non_source_sink:
                i=i_orig[:]
            
                #con_name_1='con_i_slack_pos_'+i
                con_name_1='flow_in_out_h='+h+"_n="+i
                if  con_name_1 not in self.lp_dual_solution:

                    print('ddd')
                    input('hold')
                i_2_dual[i_orig]=self.lp_dual_solution[con_name_1]#-self.lp_dual_solution[con_name_2]
        
            i_2_dual[compact_sink]=0
            i_2_dual[compact_source]=0
            
            count_change=0
            

            f_2_mean_val=dict()
            f_2_min_val=dict()
            f_2_max_val=dict()
            do_split_f=[]
            node_2_node_agg=self.MF.graph_node_2_agg_node[h]
            agg_node_2_node = {
                f: { i for i, f_val in node_2_node_agg.items() if f_val == f }
                for f in set(node_2_node_agg.values())
            }
 
            #agg_node_2_node=self.MF.my_lower_bound_LP.agg_node_2_nodes[h]
            #node_2_node_agg=self.MF.my_lower_bound_LP.graph_node_2_agg_node[h]
            compact_source=self.MF.h_2_source_id[h]
            compact_sink=self.MF.h_2_sink_id[h]
            this_fg_sink=node_2_node_agg[compact_sink]#self.MF.graph_node_2_agg_node[h][self.MF.h_2_sink_id[h]]
            this_fg_source=node_2_node_agg[compact_source]#self.MF.graph_node_2_agg_node[h][self.MF.h_2_source_id[h]]
            non_source_sink_agg_nodes=set(agg_node_2_node.keys())-set([this_fg_sink,this_fg_source])
            #print('i_2_dual')
            #print(i_2_dual)
            #print('h')
            #print(h)
            #print('len(agg_node_2_node)')
            #print(len(agg_node_2_node))
            #input('--')
            for f in agg_node_2_node:
                my_sum=0
                my_min=np.inf
                my_max=-np.inf
                all_terms=[]
                all_names=[]
                for i in agg_node_2_node[f]:
                    this_term=i_2_dual[i]
                    my_sum=my_sum+this_term#i_2_dual[i]
                    my_max=max([my_max,this_term])
                    my_min=min([my_min,this_term])
                    all_names.append(i)
                    all_terms.append(this_term)

                f_2_mean_val[f]=my_sum/len(agg_node_2_node[f])
                f_2_min_val[f]=my_min
                f_2_max_val[f]=my_max
                if f_2_max_val[f]-f_2_min_val[f]>self.MF.jy_opt['threshold_split']:#.0001:
                    do_split_f.append(f)
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
                    
            
            start_value=0
            num_thesh_use=self.MF.jy_opt['num_thresh_split_projector']
            all_keys_h=set()
            for f in do_split_f:
                start_value=start_value+num_thesh_use
                count_pos=0
                extra_str_f=str(random.randint(0,100000000))
                tmp_dict=dict()
                for i in agg_node_2_node[f]:
                    tmp_dict[i]=i_2_dual[i]
                [chosen, new_dict]=self.quantize_dict_to_index(tmp_dict,num_thesh_use)
                for i in agg_node_2_node[f]:
                    new_key=extra_string+'_'+extra_str_f+'_'+str(new_dict[i])
                    self.NAIVE_graph_node_2_agg_node[h][i]=new_key
                    count_change=count_change+1
                    count_pos=count_pos+1
                    #all_keys_h.add(new_key)

            tmp=len(set(self.NAIVE_graph_node_2_agg_node[h].values()))
            #print('len all keys')
            #print(tmp)
            #input('--')

    def quantize_dict_to_index(self,orig_dict, K):
    # 1) round all values to 3dp and get sorted uniques
        num_digits_round=self.MF.jy_opt['roundingDiscretization_num_digits_keep']
        levels = sorted({round(v, num_digits_round) for v in orig_dict.values()})

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
            
        else:
            chosen = levels

        chosen.sort()  # just in case

        # 3) snap each entry to the index of the nearest chosen level
        index_map = {}
        for key, val in orig_dict.items():
            r = round(val, num_digits_round)
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
def power_set(s):
    """
    Returns the power set of the input collection `s` as a list of tuples.
    The power set is defined as all possible subsets of `s`.
    
    Example:
        power_set({1, 2}) returns [(), (1,), (2,), (1, 2)]
    """
    s_list = list(s)  # Ensure the input is ordered
    return list(itertools.chain.from_iterable(
        itertools.combinations(s_list, r) for r in range(len(s_list) + 1)
    ))