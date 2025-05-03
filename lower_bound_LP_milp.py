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

from scipy.sparse import csr_matrix
import pulp
import sys
sys.path.append("exper_ideas")
from jy_active_set_lp import jy_active_set_lp
from jy_active_set_lp import jy_active_set_lp_primal_dual
from warm_start_lp import warm_start_lp
from warm_start_lp import forbidden_variables_loop
from warm_start_lp import forbidden_variables_loop_dual
from warm_start_lp import warm_start_lp_using_class

from solve_gurobi_lp import solve_gurobi_lp
from solve_gurobi_lp import solve_gurobi_milp
class lower_bound_LP_milp:


    def __init__(self,full_prob,graph_node_2_agg_node,OPT_do_ILP,OPT_use_psi):
        t1=time.time()

        self.times_lp_times=dict()
        self.full_prob=full_prob
        full_input_dict=full_prob.full_input_dict
        self.actions_ignore=full_prob.actions_ignore

        #print('self.actions_ignore=')
        #print(self.actions_ignore)
        #input('hi')
        self.dict_2_action_ignore=defaultdict(int)
        self.vars_names_ignore=self.actions_ignore.copy()
        #self.action_var_names_keep=set(self.all_actions)-set(full_prob.actions_ignore)

        self.graph_node_2_agg_node=graph_node_2_agg_node
        self.all_delta=full_input_dict['allDelta']
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
        self.graph_names=full_input_dict['allGraphNames']
        self.OPT_use_psi=OPT_use_psi
        self.OPT_do_ilp=OPT_do_ILP
        self.times_lp_times['prior']=time.time()-t1

        self.construct_LB_or_ILP(self.OPT_use_psi,self.OPT_do_ilp)

        if self.OPT_do_ilp==0:
            
            
            if self.full_prob.jy_opt['use_Xpress']==False and self.full_prob.jy_opt['use_gurobi']==False:
                input('i shouold not be here')
                self.make_LP()
                
            if self.full_prob.jy_opt['use_gurobi']>0.5:# and self.full_prob.jy_opt['use_gurobi']==False:
                self.call_gurobi_solver()
                self.naive_compress_get_pi_by_h_node()
                self.naive_compress_make_f_2_new_f()
                self.Naive_make_i_2_new_f()

            if self.full_prob.jy_opt['use_gurobi']<0.5 and self.full_prob.jy_opt['use_Xpress']==True:
                input('i dont want to be here im trying to paly gurobi')
                self.make_xpress_LP()
                if self.full_prob.jy_opt['use_classic_compress']:
                    t1=time.time()

                    self.naive_compress_get_pi_by_h_node()
                    self.naive_compress_make_f_2_new_f()
                    self.Naive_make_i_2_new_f()
                    self.times_lp_times['after_compression']=time.time()-t1
        else:
            if self.full_prob.jy_opt['use_gurobi']>0.5:
                self.call_gurobi_milp_solver()
            if self.full_prob.jy_opt['use_gurobi']<0.5 and self.full_prob.jy_opt['use_Xpress']==True:
                 self.solve_xpress_milp()
            if self.full_prob.jy_opt['use_Xpress']==False and self.full_prob.jy_opt['use_gurobi']==False:
                self.solve_milp()

        if self.full_prob.jy_opt['verbose']==True:
            total = sum(self.times_lp_times.values())

            time_percentage_LBLP = {key: (val / total if total != 0 else 0) for key, val in self.times_lp_times.items()}

            print('self.times_lp_times')
            print(self.times_lp_times)
            print('--')
            print('time_percentage_LBLP')
            print(time_percentage_LBLP)
            print('total')
            print(total)
            print('----')

            print('----')
            print('----')

            for key, val in sorted(self.times_lp_times.items(),
                       key=lambda kv: kv[1],
                       reverse=True):
                print(f"{key}: {val}")
            print('----')
            print('----')
            print('percentages')
            for key, val in sorted(time_percentage_LBLP.items(),
                       key=lambda kv: kv[1],
                       reverse=True):
                print(f"{key}: {val}")
            #print('self.DEBUG_len')
            #print(self.DEBUG_len)
            #input('look here')
    def make_agg_node_2_nodes(self):
        self.agg_node_2_nodes = {
            h: {
                f: { i for i, f_val in self.graph_node_2_agg_node[h].items() if f_val == f }
                for f in set(self.graph_node_2_agg_node[h].values())
            }
            for h in self.graph_names
        }

    def OLD_make_agg_node_2_nodes(self):
        self.agg_node_2_nodes=dict()
        for h in self.graph_names:
            self.agg_node_2_nodes[h]=dict()
            for i in self.graph_node_2_agg_node[h]:
                f=self.graph_node_2_agg_node[h][i]
                if f not in self.agg_node_2_nodes[h]:
                    self.agg_node_2_nodes[h][f]=set([])
                self.agg_node_2_nodes[h][f].add(i)
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


    def OLD_make_edge_fg_2_ij_reverse(self):
        self.h_fg_2_ij=dict()
        self.h_ij_2_fg=dict()
        for h in self.graph_names:
            self.h_fg_2_ij[h]=dict()
            self.h_ij_2_fg[h]=dict()
            
            edges_compact_h=self.hij_2_P[h].keys()
            for tup_ij in edges_compact_h:
                i=tup_ij[0]
                j=tup_ij[1]
                f=self.graph_node_2_agg_node[h][i]
                g=self.graph_node_2_agg_node[h][j]
                tup_fg=tuple([f,g])
                if tup_fg not in self.h_fg_2_ij[h]:
                    self.h_fg_2_ij[h][tup_fg]=set([])
                self.h_fg_2_ij[h][tup_fg].add(tup_ij)
                self.h_ij_2_fg[h][tup_ij]=tup_fg

                if tup_ij not in  self.hij_2_P[h]:
                    print('tup_ij')
                    print(tup_ij)
                    input('ok that not ok')
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
                #if self.full_prob.jy_opt['all_vars_binary']==True:
                #    self.dict_var_name_2_is_binary[var_name]=1
        self.times_lp_times['help_construct_LB_make_vars_5']=time.time()-t1
        t1=time.time()
        t1 = time.time()
        all_new_entries_ignore = []
        vars_names_ignore_set = set(self.vars_names_ignore)  # For O(1) lookups
        dict_update = {}  # Collect all new entries for one bulk update

        for h in self.graph_names:
            for q in self.h_q_2_q_id[h]:
                prefix = f"fill_PQ_h={h}_q={q}_p="
                for p in q:
                    var_name = prefix + p
                    dict_update[var_name] = 0
                    if p in vars_names_ignore_set:
                        all_new_entries_ignore.append(var_name)

        # Single update call
        self.dict_var_name_2_obj.update(dict_update)
        if self.full_prob.jy_opt['all_vars_binary']==True:
            for var_name in dict_update:
                self.dict_var_name_2_is_binary[var_name]=1
        # Final time record
        self.times_lp_times['help_construct_LB_make_vars_6'] = time.time() - t1

        t1=time.time()
        self.vars_names_ignore=self.vars_names_ignore+all_new_entries_ignore
        #if self.full_prob.jy_opt['all_vars_binary']==True:
        #    for var_name in set(self.dict_var_name_2_obj)-set(self.all_delta):
        #        self.dict_var_name_2_is_binary[var_name]=1
        self.times_lp_times['help_construct_LB_make_vars_7']=time.time()-t1

    def OLD_help_construct_LB_make_vars(self):
        use_psi=self.OPT_use_psi
        do_ilp=self.OPT_do_ilp
        self.dict_var_name_2_is_binary=defaultdict(int)
        self.names_binary=[]
        if use_psi==True and do_ilp==True:
            for var_name in self.all_primitive_vars:
                self.dict_var_name_2_obj[var_name]=0
                self.dict_var_name_2_is_binary[var_name]=1
        my_x_typr='Binary'
        if do_ilp==False or use_psi==True:
            my_x_typr='Continuous'
        for var_name in self.all_non_null_action:
            self.dict_var_name_2_obj[var_name]=self.action_2_cost[var_name]
        if do_ilp==True or use_psi==False:
            for var_name in self.all_non_null_action:
                self.dict_var_name_2_is_binary[var_name]=1
        for var_name in self.all_delta:
            self.dict_var_name_2_obj[var_name]=0
        for h in self.graph_names:
            for tup_fg in self.h_fg_2_ij[h]:
                
                f=tup_fg[0]
                g=tup_fg[1]
                var_name='EDGE_h='+h+'_f='+f+'_g='+g
                self.dict_var_name_2_obj[var_name]=0
        for h in self.graph_names:
            for q in self.h_q_2_q_id[h]:
                p_list=list(q)

                for p in p_list:
                    
                    var_name='fill_PQ_h='+h+'_q='+str(q)+'_p='+p

                    self.dict_var_name_2_obj[var_name]=0
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
                #print('my_source')
                #print(my_source)
                #print('my_sink')
                #print(my_sink)
                #print('self.h_2_sink_id[h]')
                #print(self.h_2_sink_id[h])
                #print('self.h_2_source_id[h]')
                #print(self.h_2_source_id[h])
                #input('---')
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
        
    def OLD_construct_constraints_actions_match_compact(self):
        for h in self.graph_names:
            for p in self.all_non_null_action:
                con_name='action_match_h='+h+"_p="+p
                self.dict_var_con_2_lhs_eq[tuple([p,con_name])]=-1
        
        for h in self.graph_names:
            for q in self.h_q_2_fg[h]:
                
                for p in q:
                    if p!=self.null_action:
                        con_name_2='action_match_h='+h+"_p="+p
                        var_name_2='fill_PQ_h='+h+'_q='+str(q)+'_p='+p
                        self.dict_var_con_2_lhs_eq[tuple([var_name_2,con_name_2])]=1
    
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


    def OLD_construct_constraints_actions_match_flow(self):
        my_count=0
        my_count_2=0
        my_counter_list=[]
        print('starting ')
        for h in self.graph_names:
            print('h')
            print(h)
            print('len(self.h_q_2_fg[h])')
            print(len(self.h_q_2_fg[h]))
            for q in self.h_q_2_fg[h]:
                con_name='equiv_class='+h+"_q="+str(q)
                for p in q:
                    var_name_2='fill_PQ_h='+h+'_q='+str(q)+'_p='+p
                    self.dict_var_con_2_lhs_eq[tuple([var_name_2,con_name])]=-1
                    my_count=my_count+1
                for e in self.h_q_2_fg[h][q]:
                    f=e[0]
                    g=e[1]

                    var_name_edge='EDGE_h='+h+'_f='+f+'_g='+g
                    self.dict_var_con_2_lhs_eq[tuple([var_name_edge,con_name])]=1
                    my_count_2=my_count_2+1
        print("my_count:  "+str(my_count))
        print("my_count_2:  "+str(my_count_2))
        input('----')
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

        #print('t_list')
        #print(t_list)
        #print(np.array(t_list)/np.sum(np.array(t_list)))
        #input('--')
    def solve_milp(self):
        dict_var_name_2_obj=self.dict_var_name_2_obj
        dict_con_name_2_LB=self.dict_con_name_2_LB
        dict_con_name_2_eq=self.dict_con_name_2_eq
        dict_var_con_2_lhs_exog=self.dict_var_con_2_lhs_exog
        dict_var_con_2_lhs_eq=self.dict_var_con_2_lhs_eq
        dict_var_name_2_is_binary=self.dict_var_name_2_is_binary
        """
        Builds and solves a MILP based on the input dictionaries.
        
        Parameters:
        dict_var_name_2_obj: dict mapping variable name (str) -> coefficient in objective.
        dict_con_name_2_LB: dict mapping inequality constraint name -> RHS lower bound for constraint (>=).
        dict_con_name_2_eq: dict mapping equality constraint name -> RHS value for the constraint.
        dict_var_con_2_lhs_exog: dict mapping (var_name, con_name) -> coefficient in the inequality constraint.
        dict_var_con_2_lhs_eq: dict mapping (var_name, con_name) -> coefficient in the equality constraint.
        dict_var_name_2_is_binary: dict mapping variable name -> 1 if binary, 0 otherwise.
        
        Returns:
        A tuple (status, objective_value, variable_values) where:
            - status: the status of the solution.
            - objective_value: the optimal value of the objective.
            - variable_values: a dict mapping variable names to their optimal values.
        """
        # Create the MILP model (for example: minimization problem)
        milp_prob = pulp.LpProblem("MILP_Problem", pulp.LpMinimize)
        
        # Create decision variables based on the input. 
        # If a variable is binary, declare it as such, otherwise as continuous (nonnegative).
        var_dict = {}
        
        for var_name, obj_coeff in dict_var_name_2_obj.items():
            if dict_var_name_2_is_binary.get(var_name, 0):
                #print('is binary')
                var_dict[var_name] = pulp.LpVariable(var_name, lowBound=0, upBound=1, cat=pulp.LpBinary)
            else:
                var_dict[var_name] = pulp.LpVariable(var_name, lowBound=0)
        
        # Define the objective function: minimize sum(objective_coefficient * variable)
        milp_prob += pulp.lpSum(dict_var_name_2_obj[var_name] * var_dict[var_name]
                                for var_name in dict_var_name_2_obj), "Objective"
        
        # --- Add inequality constraints (of the form: expression >= lower bound) ---
        ineq_expressions = {}
        for (var_name, con_name), coeff in dict_var_con_2_lhs_exog.items():
            ineq_expressions.setdefault(con_name, 0)
            ineq_expressions[con_name] += coeff * var_dict[var_name]
        
        for con_name, expr in ineq_expressions.items():
            if con_name in dict_con_name_2_LB:
                milp_prob += expr >= dict_con_name_2_LB[con_name], con_name + "_ineq"
        
        # --- Add equality constraints ---
        eq_expressions = {}
        for (var_name, con_name), coeff in dict_var_con_2_lhs_eq.items():
            eq_expressions.setdefault(con_name, 0)
            eq_expressions[con_name] += coeff * var_dict[var_name]
        
        for con_name, expr in eq_expressions.items():
            if con_name in dict_con_name_2_eq:
                milp_prob += expr == dict_con_name_2_eq[con_name], con_name + "_eq"
        
        # --- Solve the MILP ---
        start_time=time.time()

        solver = pulp.PULP_CBC_CMD(msg=True)
        milp_prob.solve(solver)
        end_time=time.time()
        self.milp_time=end_time-start_time
        self.milp_prob=milp_prob
        self.milp_solution = {var_name: var.varValue for var_name, var in var_dict.items()}
        self.milp_solution_status = pulp.LpStatus[milp_prob.status]
        self.milp_solution_objective_value = pulp.value(milp_prob.objective)

        #print('done ILP call')
        #input('done ILP call')
    def solve_xpress_milp(self):
        print('starting the setup of the  MILP call')

        t2=time.time()
        dict_var_name_2_obj=self.dict_var_name_2_obj
        dict_con_name_2_LB=self.dict_con_name_2_LB
        dict_con_name_2_eq=self.dict_con_name_2_eq
        dict_var_con_2_lhs_exog=self.dict_var_con_2_lhs_exog
        dict_var_con_2_lhs_eq=self.dict_var_con_2_lhs_eq
        dict_var_name_2_is_binary=self.dict_var_name_2_is_binary
        #import xpress as xp
        xp.init('C:/xpressmp/bin/xpauth.xpr')
        milp_prob = xp.problem("MILP_Problem")
        milp_prob.setOutputEnabled(self.full_prob.jy_opt['verbose']>0.5)

        # Create decision variables based on the input. 
        # If a variable is binary, declare it as such, otherwise as continuous (nonnegative).
        var_dict = {}

        if 1<0:
            vars_list = [xp.var(name=name, lb=0) for name in dict_var_name_2_obj]

            for var_name, obj_coeff in dict_var_name_2_obj.items():
                if dict_var_name_2_is_binary.get(var_name, 0):
                    var_dict[var_name] = milp_prob.addVariable(name=var_name, vartype=xp.binary)
                else:
                    var_dict[var_name] = milp_prob.addVariable(name=var_name, lb=0)
        else:
            vars_list = [
            xp.var(
                name=name,
                lb=0,
                vartype=(xp.binary if dict_var_name_2_is_binary.get(name, 0) else xp.continuous)
            )
            for name in dict_var_name_2_obj
            ]
            for var in vars_list:
                var_dict[var.name]=var
            milp_prob.addVariable(*vars_list) 
        # Define the objective function: minimize sum(objective_coefficient * variable)
        objective = xp.Sum(dict_var_name_2_obj[var_name] * var_dict[var_name] 
                            for var_name in dict_var_name_2_obj)
        milp_prob.setObjective(objective, sense=xp.minimize)

        # --- Add inequality constraints (of the form: expression >= lower bound) ---
        
        vdict    = var_dict
        LB       = dict_con_name_2_LB
        EQ       = dict_con_name_2_eq
        exog     = dict_var_con_2_lhs_exog
        eq_map   = dict_var_con_2_lhs_eq
        cx       = xp.constraint
        ac       = milp_prob.addConstraint

        # 2) One‐time grouping of terms by constraint name
        group_exog = defaultdict(list)
        for (var, con), coeff in exog.items():
            group_exog[con].append((vdict[var], coeff))

        group_eq = defaultdict(list)
        for (var, con), coeff in eq_map.items():
            group_eq[con].append((vdict[var], coeff))

        # 3) Build all constraint objects
        cons = []
        for con_name, terms in group_exog.items():
            # sum up coeff * var
            expr = sum(var * coeff for var, coeff in terms)
            cons.append(cx(expr >= LB[con_name], name=con_name))

        for con_name, terms in group_eq.items():
            expr = sum(var * coeff for var, coeff in terms)
            cons.append(cx(expr == EQ[con_name], name=con_name))

        # 4) Bulk‐add them in one call
        ac(*cons)
        # --- Solve the MILP ---
        self.times_lp_times['pre_XMILP']=time.time()-t2
        print('starting the final MILP call')
        start_time = time.time()
        milp_prob.solve()
        end_time = time.time()
        self.times_lp_times['XMILP']=end_time - start_time

        t3=time.time()

        self.milp_time = end_time - start_time
        self.milp_prob = milp_prob
        
        vals = milp_prob.getSolution(vars_list)
        t3=time.time()

        self.milp_solution = {
            var.name: vals[i]
            for i, var in enumerate(vars_list)
        }
        
        #self.milp_solution = {var_name: milp_prob.getSolution(var_name) for var_name in var_dict}
        self.milp_solution_status = milp_prob.getProbStatus()
        self.milp_solution_objective_value = milp_prob.getObjVal()
        self.times_lp_times['post_XMILP']=time.time()-t3

    def make_LP(self):


        
        dict_var_name_2_obj=self.dict_var_name_2_obj
        dict_var_con_2_lhs_exog=self.dict_var_con_2_lhs_exog
        dict_con_name_2_LB=self.dict_con_name_2_LB
        dict_var_con_2_lhs_eq=self.dict_var_con_2_lhs_eq
        dict_con_name_2_eq=self.dict_con_name_2_eq
        t2=time.time()
        debug_on=False
        if debug_on==True:
            dict_con_name_2_eq=dict()
            dict_var_con_2_lhs_eq=dict()
        # --- Build the LP model ---
        my_times=[]
        t1=time.time()
        lp_prob = pulp.LpProblem("MyLP", pulp.LpMinimize)
        my_times.append(time.time()-t1)
        #print('my_times[-1];  0')
        #print(my_times[-1])
        # Create decision variables (all non-negative)
        var_dict = {}
        t1=time.time()
        for var_name, coeff in dict_var_name_2_obj.items():
            var_dict[var_name] = pulp.LpVariable(var_name, lowBound=0)
        my_times.append(time.time()-t1)
        #print('my_times[-1];  1')
        #print(my_times[-1])
        t1=time.time()

        # Define the objective function (minimize sum(obj_coeff * var))
        lp_prob += pulp.lpSum(dict_var_name_2_obj[var_name] * var_dict[var_name]
                            for var_name in self.all_actions), "Objective"
        my_times.append(time.time()-t1)
        #print('my_times[-1];  2')
        #print(my_times[-1])
        # --- Add inequality constraints (>=) ---
        # Group terms for each inequality constraint.
        ineq_expressions = {}
        did_find_2=False
        #input('----')
        t1=time.time()

        for (var_name, con_name), coeff in dict_var_con_2_lhs_exog.items():
            ineq_expressions.setdefault(con_name, 0)
            ineq_expressions[con_name] += coeff * var_dict[var_name]
            if con_name=='exog_min_veh_':

                did_find_2=True
        my_times.append(time.time()-t1)
        #print('my_times[-1];  3')
        #print(my_times[-1])
        t1=time.time()
        did_find=False
        for con_name, expr in ineq_expressions.items():
            if con_name in dict_con_name_2_LB:

                if con_name=='exog_min_veh_':
                    did_find=True
                    #input('---')
                lp_prob += expr >= dict_con_name_2_LB[con_name], con_name + "_ineq"
        my_times.append(time.time()-t1)
        #print('my_times[-1];  4')
        #print(my_times[-1])
        if did_find==False:
            input('this is odd')
        # --- Add equality constraints ---
        # Group terms for each equality constraint.
        t1=time.time()
        eq_expressions = {}
        for (var_name, con_name), coeff in dict_var_con_2_lhs_eq.items():
            eq_expressions.setdefault(con_name, 0)
            eq_expressions[con_name] += coeff * var_dict[var_name]
        my_times.append(time.time()-t1)
        #print('my_times[-1]; 5')
        #print(my_times[-1])
        t1=time.time()
        # Add each equality constraint to the model.
        for con_name, expr in eq_expressions.items():
            if con_name in dict_con_name_2_eq:
                lp_prob += expr == dict_con_name_2_eq[con_name], con_name #+ "_eq"
        my_times.append(time.time()-t1)
        #print('my_times[-1]; 6')
        #print(my_times[-1])     
                #input('----')
        # --- Solve the LP ---
        # Using the default CBC solver here.
        self.times_lp_times['pre_lp_solve']=time.time()-t2
        start_time=time.time()
        #if 1>0:
        solver = pulp.PULP_CBC_CMD(msg=False)
        #if 1>0:
        #    input('here')
        #    solver = pulp.XPRESS_CMD(msg=False)
        #    input('done')

        #input('hhii')
        lp_prob.solve(solver)
        #input('hohoh')

        end_time=time.time()
        self.lp_time=end_time-start_time
        my_times.append(end_time-start_time)
        self.times_lp_times['lp_time']=end_time-start_time
        t3=time.time()
        #print('my_times[-1]; 7')
        #print(my_times[-1])     
        self.lp_prob=lp_prob
        self.lp_primal_solution=dict()
        t1=time.time()

        for var_name, var in var_dict.items():
            self.lp_primal_solution[var_name]=var.varValue
        my_times.append(time.time()-t1)
        #print('my_times[-1]; 8')
        #print(my_times[-1])     
        self.lp_status=pulp.LpStatus[lp_prob.status]
        self.lp_objective= pulp.value(lp_prob.objective)
        t1=time.time()
        self.lp_dual_solution=dict()
        for con_name, constraint in lp_prob.constraints.items():
            self.lp_dual_solution[con_name]=constraint.pi
        my_times.append(time.time()-t1)
        #print('my_times[-1]; 9')
        #print(my_times[-1])     
        self.times_lp_times['post_lp_time']=time.time()-t3
        #print(np.array(my_times)/(np.sum(np.array(my_times))))
        #input('---')
        
        total = sum(self.times_lp_times.values())
        time_percentage_LP = {key: (val / total if total != 0 else 0) for key, val in self.times_lp_times.items()}
        if self.full_prob.jy_opt['verbose']==True:
            print('self.times_lp_times')
            print(self.times_lp_times)
            print('--')
            print('time_percentage_LP')
            print(time_percentage_LP)
            print('----')
        if self.lp_status=='Infeasible':
            input('HOLD')
    def make_xpress_LP(self):

       # if 'exog_min_veh_' not in self.dict_con_name_2_LB:
       #     input('error ')
       # else:
       #     print(self.dict_con_name_2_LB['exog_min_veh_'])
       #     input('--')
        #/Users/julian/Documents/FICO\ Xpress\ Config/xpauth.xpr
        #xp.init('C:/xpressmp/bin/xpauth.xpr')
        xp.init(self.full_prob.jy_opt['xpress_file_loc'])
        t2=time.time()
        dict_var_name_2_obj = self.dict_var_name_2_obj
        dict_var_con_2_lhs_exog = self.dict_var_con_2_lhs_exog
        dict_con_name_2_LB = self.dict_con_name_2_LB
        dict_var_con_2_lhs_eq = self.dict_var_con_2_lhs_eq
        dict_con_name_2_eq = self.dict_con_name_2_eq

        debug_on = False
        if debug_on == True:
            dict_con_name_2_eq = dict()
            dict_var_con_2_lhs_eq = dict()

        # --- Build the LP model ---
        lp_prob = xp.problem("MyLP")
        lp_prob.setOutputEnabled(self.full_prob.jy_opt['verbose']>0.5)

        # Create decision variables (all non-negative) and store them in a list.
        var_dict = {}
        vars_list = []  # list of variables to add to the model
        vars_list = [xp.var(name=name, lb=0) for name in dict_var_name_2_obj]

        self.times_lp_times['pre_XP_lp_2_pt0']=time.time()-t2
        #print('BIG LEN(vars_list)')
        #print(len(vars_list))
        self.DEBUG_len=len(vars_list)
        t2=time.time()

        lp_prob.addVariable(*vars_list)   # ← note the * here!

        self.times_lp_times['pre_XP_lp_2_pt0.5']=time.time()-t2
        t2=time.time()
        var_dict = {v.name: v for v in vars_list}

        self.var_dict=var_dict
        # Define the objective function (minimize sum of coeff * variable).
        # Converting the generator to a list to be safe.
        objective = xp.Sum([dict_var_name_2_obj[var_name] * var_dict[var_name]
                            for var_name in dict_var_name_2_obj])
        lp_prob.setObjective(objective, sense=xp.minimize)
        # --- Add inequality constraints (>=) ---
        # Group terms for each inequality constraint.
        self.times_lp_times['pre_XP_lp_2_pt1']=time.time()-t2
    
        
        vdict    = var_dict
        LB       = dict_con_name_2_LB
        EQ       = dict_con_name_2_eq
        exog     = dict_var_con_2_lhs_exog
        eq_map   = dict_var_con_2_lhs_eq
        cx       = xp.constraint
        ac       = lp_prob.addConstraint

        # 2) One‐time grouping of terms by constraint name
        group_exog = defaultdict(list)
        for (var, con), coeff in exog.items():
            group_exog[con].append((vdict[var], coeff))

        group_eq = defaultdict(list)
        for (var, con), coeff in eq_map.items():
            group_eq[con].append((vdict[var], coeff))

        # 3) Build all constraint objects
        cons = []
        for con_name, terms in group_exog.items():
            # sum up coeff * var
            expr = sum(var * coeff for var, coeff in terms)
            cons.append(cx(expr >= LB[con_name], name=con_name))

        for con_name, terms in group_eq.items():
            expr = sum(var * coeff for var, coeff in terms)
            cons.append(cx(expr == EQ[con_name], name=con_name))

        # 4) Bulk‐add them in one call
        ac(*cons)
        # --- Solve the LP ---
        lp_prob.controls.defaultalg = self.full_prob.jy_opt['lplb_solver']

        self.times_lp_times['pre_XP_lp_2_pt2']=time.time()-t2
        
        if self.full_prob.jy_opt['use_julians_custom_lp_solver']<0.5:
            start_time = time.time()

            lp_prob.solve()
            end_time = time.time()
            self.lp_time = end_time - start_time

        else: #lp_prob, var_dict, zero_names
            
            #lp_prob,time_lp_1=forbidden_variables_loop(lp_prob,self.var_dict,self.actions_ignore)
            #lp_prob,time_lp_1=forbidden_variables_loop_dual(lp_prob,self.var_dict,self.actions_ignore)
            print('STARTING WARM  LP LOWER ')

            lp_prob,time_lp_1=warm_start_lp_using_class(lp_prob,self.var_dict,self.full_prob.all_actions_not_source_sink_connected,self.actions_ignore)
            print('DONE WARM  LP LOWER ')

            self.lp_time=time_lp_1
        self.times_lp_times['lp_time']=self.lp_time
        t3=time.time()
        self.lp_prob = lp_prob
        self.lp_primal_solution = dict()
        self.times_lp_times['post_XLP_1']=time.time()-t3
        t3=time.time()

        self.lp_status = lp_prob.getProbStatus()
        self.lp_objective = lp_prob.getObjVal()
        self.lp_dual_solution = dict()
        self.times_lp_times['post_XLP_2']=time.time()-t3
        t3=time.time()

        vals = lp_prob.getSolution(vars_list)
        self.times_lp_times['post_XLP_3']=time.time()-t3
        t3=time.time()

        self.lp_primal_solution = {
            var.name: vals[i]
            for i, var in enumerate(vars_list)
        }
        self.times_lp_times['post_XLP_4']=time.time()-t3
        t3=time.time()
        if 0>1:
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
        self.new_actions_ignore=[]#self.full_prob.all_actions_not_source_sink_connected.copy()
        
        for my_act in self.full_prob.all_actions_not_source_sink_connected:
            if self.lp_primal_solution[my_act]==0:
                self.new_actions_ignore.append(my_act)
        #print('len(self.new_actions_ignore)')
        #print(len(self.new_actions_ignore))
        #print('len(self.full_prob.all_actions_not_source_sink_connected)')
        #print(len(self.full_prob.all_actions_not_source_sink_connected))
        if self.lp_status == 'Infeasible':
            input('HOLD')
        self.times_lp_times['post_XLP_5']=time.time()-t3


    def call_gurobi_solver(self):
        
        out_solution=solve_gurobi_lp(self.dict_var_name_2_obj,
                   self.dict_var_con_2_lhs_exog,
                   self.dict_con_name_2_LB,
                   self.dict_var_con_2_lhs_eq,
                   self.dict_con_name_2_eq)
        self.lp_dual_solution=out_solution['dual_solution']
        self.lp_primal_solution=out_solution['primal_solution']
        self.lp_objective=out_solution['objective']
        self.times_lp_times['GUR_time_pre']=out_solution['time_pre']
        self.times_lp_times['GUR_time_opt']=out_solution['time_opt']
        self.times_lp_times['GUR_time_post']=out_solution['time_post']
        self.lp_time=out_solution['time_opt']
        self.new_actions_ignore=[]

    def call_gurobi_milp_solver(self):
        
        out_solution=solve_gurobi_milp(self.dict_var_name_2_obj,
                   self.dict_var_con_2_lhs_exog,
                   self.dict_con_name_2_LB,
                   self.dict_var_con_2_lhs_eq,
                   self.dict_con_name_2_eq,
                   self.dict_var_name_2_is_binary)
        self.milp_solution=out_solution['primal_solution']
        self.milp_solution_objective_value=out_solution['objective']
        self.times_lp_times['GUR_time_pre']=out_solution['time_pre']
        self.times_lp_times['GUR_time_opt']=out_solution['time_opt']
        self.times_lp_times['GUR_time_post']=out_solution['time_post']
        self.milp_time=out_solution['time_opt']
        self.new_actions_ignore=[]


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
                my_key=tuple([h,my_dual_id])
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

    