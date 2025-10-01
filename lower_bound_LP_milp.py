import math
import xpress as xp
import pickle
import copy
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
import heapq

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

class lower_bound_LP_milp:


    def __init__(self,full_prob,graph_node_2_agg_node,OPT_do_ILP,OPT_use_psi):
        t1=time.time()
        self.extra_var_name_priority=dict()

        self.times_lp_times=dict()
        self.full_prob=full_prob
        full_input_dict=full_prob.full_input_dict
        self.actions_ignore=full_prob.actions_ignore
        self.dict_pred_gain=None
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
        self.graphNameNode_2_cust=full_input_dict['graphNameNode_2_cust']

        #maps the names of the graphs to the id for the sink
        self.h_2_sink_id=full_input_dict['h2sinkid']
        self.graph_names=full_input_dict['allGraphNames']
        self.OPT_use_psi=OPT_use_psi
        self.OPT_do_ilp=OPT_do_ILP
        self.times_lp_times['prior']=time.time()-t1

        self.construct_LB_or_ILP(self.OPT_use_psi,self.OPT_do_ilp)
        if ( self.OPT_do_ilp==0 and self.full_prob.jy_opt['use_delta_in_lp']==False) or (self.OPT_do_ilp!=0 and self.full_prob.jy_opt['use_delta_in_milp']==False):
            self.remove_remove_delta_and_delta_con()
        t1=time.time()
        #if self.OPT_do_ilp!=0:
        #    input('about to call')
        #    print(self.full_prob.jy_opt['LAB_MP_ON']>0.5)
        #    input('--')
        if self.OPT_do_ilp!=0 and self.full_prob.jy_opt['LAB_MP_ON']>0.5:
            #input('about to start')
            DEBUG_ON=False
            if DEBUG_ON==True:
                print('DEBUG_ON')
                with open("solver_checkpoint_BEF_.pkl", "wb") as f:
                    pickle.dump(self, f)
            print('done writing ')
            #self.iterative_ilp_la()
            self.iterative_ilp_la_DIVE()
            #self.apply_LA_branching()


        self.filter_constraints()
        self.times_lp_times['filetering']=time.time()-t1

        if 'var_int_names_fancy_branch' in self.full_prob.D:
            big_M=2000000
            if self.OPT_do_ilp>0.5:
                for var_force_integ in self.full_prob.D['var_int_names_fancy_branch']:
                    self.dict_var_name_2_is_integer[var_force_integ]=1
            for var_penalty in self.full_prob.D['var_cont_names_fancy_branch']:
                self.dict_var_name_2_obj[var_penalty]=big_M
            
        if self.OPT_do_ilp==0:
            
            #if self.full_prob.jy_opt['use_delta_in_lp']==False:
            #    input('hi')
            
            if self.full_prob.jy_opt['use_Xpress']==False and self.full_prob.jy_opt['use_gurobi']==False:
                input('i shouold not be here')
                self.make_LP()
                
            if self.full_prob.jy_opt['use_gurobi']>0.5:# and self.full_prob.jy_opt['use_gurobi']==False:
                #if self.full_prob.jy_opt['think_compress'] and len(self.full_prob.all_actions_ever_seen)>0:
                #    self.THINK_aggregate_constraints_dictionary()
                #input('hihi')
                self.call_gurobi_solver()
                t1=time.time()
                self.get_ij_poss_active()
                self.times_lp_times['allPostOpps4']=time.time()-t1
                t1=time.time()
                self.naive_compress_get_pi_by_h_node()
                self.times_lp_times['allPostOpps3']=time.time()-t1
                t1=time.time()
                self.naive_compress_make_f_2_new_f()
                self.times_lp_times['allPostOpps2']=time.time()-t1
                t1=time.time()
                self.Naive_make_i_2_new_f()
                self.times_lp_times['allPostOpps1']=time.time()-t1


            if self.full_prob.jy_opt['use_gurobi']<0.5 and self.full_prob.jy_opt['use_Xpress']==True:
                #input('i dont want to be here im trying to paly gurobi')
                self.make_xpress_LP()
                if self.full_prob.jy_opt['use_classic_compress']:
                    t1=time.time()

                    self.naive_compress_get_pi_by_h_node()
                    self.naive_compress_make_f_2_new_f()
                    self.Naive_make_i_2_new_f()
                    self.times_lp_times['after_compression']=time.time()-t1
        else:

            #if self.full_prob.jy_opt['use_delta_in_lp']==False:
            #    self.remove_remove_delta_and_delta_con()
            #if self.full_prob.jy_opt['think_compress']>0.5:
                
               
            #    self.THINK_aggregate_constraints_dictionary()
            #    input('STARTING LP_PRE')

            
            if self.full_prob.jy_opt['use_gurobi']>0.5:

                #lb = self.CLEAN_dict_con_name_2_LB

                # Map constraint name -> LB, filtered to those starting with "side_ineq_use"
                #side_ineq_use_constraints = {
                #    c: lb[c]
                #    for c in lb
                #    if isinstance(c, str) and c.startswith("side_ineq_use")
                #}
                #print('PRIOR')
                #print(side_ineq_use_constraints)
                #input('--')


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
            input('look here')
        #input('hih')
        UB_USE_REMOVE=self.full_prob.jy_opt['ub_use_remove']
        if UB_USE_REMOVE<0:
            UB_USE_REMOVE=np.inf
        if  UB_USE_REMOVE<np.inf and self.full_prob.jy_opt['use_julians_custom_lp_solver']<0.5 and self.OPT_do_ilp==0:
            self.select_low_reduced_cost_actions(UB_USE_REMOVE)
        #input('done')

    def select_low_reduced_cost_actions(self,ub_current):
        
        upper_bound_valid=True
        
        #input('---')
        #if self.full_prob.jy_opt['do_ilp']==False and :
        #    ub_current=np.inf
        if self.full_prob.jy_opt['do_ilp']==False and 'ub_lp' in self.full_prob.history_dict and len(self.full_prob.history_dict['ub_lp'])>0 :
            ub_current=min([ub_current,self.full_prob.history_dict['ub_lp'][-1]])
        
        #input('---')
        eta=(ub_current-self.lp_objective)+0.001
        if eta<0:
            print('eta can not be zero')
            print('eta')
            print(eta)
            input('error here')
        print('ub_current = '+ str(ub_current)+ '  eta = '+str(eta))
        primal_solution=self.out_solution_JY['primal_solution']
        reduced_costs=self.out_solution_JY['reduced_costs']
        usedTerms=set(self.all_actions)-set(self.actions_ignore)
        pattern = re.compile(r"act_(\d+)_(\d+)")
        candidate_per_u = defaultdict(list)
        num_found_high=0
        num_found_low=0
        num_used=0
        num_already_gone=0
        num_active=0
        num_in_solution_ignore=0
        terms_remove=[]
        use_path_based_RC=1
        if use_path_based_RC>0.5:
            #if hasattr(self, "path_rc_pre_done")==False:
            #    self.setup_pre_path_based_rc()
            self.compute_path_based_rc()
        for act in self.all_non_null_action:
            #if act in usedTerms:
            #    num_used=num_used+1
            #    continue
            if act in self.full_prob.delta_name_2_ub and self.full_prob.delta_name_2_ub[act]<0.001:
                num_already_gone=num_already_gone+1
                continue
            if act in self.full_prob.all_actions_inclumbent:# or primal_solution[act]>0.001:
                num_in_solution_ignore+=1
                continue
            if primal_solution[act]>0.001:
                num_active+=1
                continue

            match = pattern.fullmatch(act)
            if not match:
                input('error here')
                continue

            u, v = match.groups()
            rc = reduced_costs.get(act, float('inf'))
            
            red_val_sum=0
            tmp_val=dict()
            for h in self.graph_names:
                red_val=np.inf
                for v_name in self.mapping_h_p_to_vars_use[h][act]:
                    red_val=min([red_val,reduced_costs[v_name]])
                red_val_edge=np.inf
                for v_name_2 in self.mapping_h_p_to_fg_vars_use[h][act]:
                    red_val_edge=min([red_val_edge,reduced_costs[v_name_2]])

                tmp_val[h]=[red_val,red_val_edge]
                red_val_sum=red_val_sum+red_val+red_val_edge

            #if rc<eta  and rc+red_val_sum>eta:
            #    print('rc,red_val_sum')
            #    print([rc,red_val_sum])
            #    input('look here ')
            rc=rc+red_val_sum
            if use_path_based_RC>0.5:
                if rc-0.001>self.path_based_rc[act]:
                    input('wrong not ok')
                #if rc<eta and self.path_based_rc[act]>eta:
                #    print('rc')
                #    print(rc)
                #    print('self.path_based_rc[act]')
                #    print(self.path_based_rc[act])
                #    print('eta')
                #    print(eta)
                #    input('OK INTERESTING ')
                rc=self.path_based_rc[act]
            if rc>eta:
                #print('rc')
                #p#rint(rc)
                terms_remove.append(act)
                num_found_high=num_found_high+1
                #input('hold')
            else:
                num_found_low=num_found_low+1
            if rc >= eta:
                continue

            # Store (rc, act) so we can heapq.nsmallest later
            candidate_per_u[u].append((rc, act))

        # Now select the 2 smallest per u
        #result = {
        #    u: [act for _, act in heapq.nsmallest(2, entries)]
        #    for u, entries in candidate_per_u.items()
        #}
        result = {
            u: heapq.nsmallest(2, entries)
            for u, entries in candidate_per_u.items()
        }
        self.terms_remove_this_round=terms_remove
        print('num_found_high,num_found_low,num_used,num_already_gone,num_in_solution_ignore,num_active')
        print([num_found_high,num_found_low,num_used,num_already_gone,num_in_solution_ignore,num_active])
        #print(num_found_high)
        #print('num_found_low')
        #print(num_found_low)
        #print('num_used')
        #print(num_used)
        #print('num_already_gone')
        #print(num_already_gone)
        #print('num_in_solution_ignore')
        #print(num_in_solution_ignore)
        #print('num_active')
        #print(num_active)
        #print(result)
        #print('result')
       # input('--')

        if upper_bound_valid==True:
            for p in terms_remove:
                self.full_prob.delta_name_2_ub[p]=0
           # given set terms_to_remove. got through the dictionary hijp  which we index by ij and remove any terms for which hijp[ij] for which p in hijp[ij]
            allowed_terms=set(self.all_actions)-set(terms_remove)
            allowed_terms.add('null_action')
            #print('removing')
            t_list=[]

            t1=time.time()
            for h in self.graph_names:
                hijp=self.full_prob.full_input_dict['hij2P'][h]
                for h in self.graph_names:
                    hijp = self.full_prob.full_input_dict['hij2P'][h]
                    self.full_prob.full_input_dict['hij2P'][h] = {
                        ij: p_list for ij, p_list in hijp.items()
                        #if p_list[0] not in terms_remove
                        if p_list[0] in allowed_terms#not in terms_remove
                    }
                #for ij in list(hijp.keys()):
                #    p = hijp[ij][0]
                #    if p in terms_remove:
                #        del hijp[ij]
            t_list.append(time.time()-t1)
            t1=time.time()
            cons_to_grab = {
                con for (action, con) in self.action_con_2_contrib
                if  (con.startswith("cap_uv_") or con.startswith("time_uv_")) and action in terms_remove
            }
            t_list.append(time.time()-t1)
            t1=time.time()
            
            self.action_con_2_contrib = {
                (action, con): val
                for (action, con), val in self.action_con_2_contrib.items()
                #if action  not in terms_remove
                if action  in allowed_terms#not in terms_remove
            }

            t_list.append(time.time()-t1)
            t1=time.time()
            self.delta_con_2_contrib = {
                (delta_term, con): val
                for (delta_term, con), val in self.delta_con_2_contrib.items()
                if con not in cons_to_grab
            }
            t_list.append(time.time()-t1)
            t1=time.time()
            self.exog_name_2_rhs = {
                (con): val
                for (con), val in self.exog_name_2_rhs.items()
                if con not in cons_to_grab
            }
            t_list.append(time.time()-t1)
            t1=time.time()
            #print('t_list')
            #print(t_list)
            #print('done removing')
        return result        

    def remove_remove_delta_and_delta_con(self):
        print('rmoving')
        delta_vars = set()
        affected_cons = set()

        for (var_name, con_name), coeff in self.dict_var_con_2_lhs_exog.items():
            if "delta" in var_name and coeff != 0:
                delta_vars.add(var_name)
                affected_cons.add(con_name)
                #if con_name=='time_uv_0_1':
                #    print((var_name, con_name))
                #    input('added' )
        #print('affected_cons')
        #print(affected_cons)
        #print('var_name')
        #input('---')
        # Optional: show what you found
        #print("Delta variables:", delta_vars)
        #print("Constraints affected:", affected_cons)

        # Step 2: Remove delta variables from all relevant dictionaries

        dicts_to_clean = [
            self.dict_var_name_2_obj,
            #self.dict_var_con_2_lhs_exog,
            #self.dict_var_con_2_lhs_eq,
            self.dict_var_name_2_is_binary,
            #self.dict_con_name_2_LB
            #self.delta_name_2_lb,
            #self.delta_name_2_ub,
        ]

        for d in dicts_to_clean:
            for var in delta_vars:
                d.pop(var, None)

        dicts_to_clean = [
            #self.dict_var_name_2_obj,
            #self.dict_var_con_2_lhs_exog,
            #self.dict_var_con_2_lhs_eq,
            #self.dict_var_name_2_is_binary,
            self.dict_con_name_2_LB
            #self.delta_name_2_lb,
            #self.delta_name_2_ub,
        ]

        for d in dicts_to_clean:
            for con in affected_cons:
                d.pop(con, None)
        # Remove entries from dicts with (var_name, con_name) keys
        dicts_with_tuple_keys = [
            self.dict_var_con_2_lhs_exog,
        ]

        for d in dicts_with_tuple_keys:
            keys_to_remove = [k for k in d if k[1] in affected_cons]
            for k in keys_to_remove:
                del d[k]
            

        # Remove affected constraints from constraint-bound dictionaries

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
        
        self.mapping_h_p_to_fg_vars_use=dict()

    
        t1=time.time()
        
        for h in self.graph_names:
            self.mapping_h_p_to_fg_vars_use[h]=defaultdict(list)

            for tup_fg in self.h_fg_2_ij[h]:
                
                f=tup_fg[0]
                g=tup_fg[1]
                var_name='EDGE_h='+h+'_f='+f+'_g='+g
                self.dict_var_name_2_obj[var_name]=0
                
                #if (self.full_prob.jy_opt['allOneBig_init']==False and self.full_prob.jy_opt['do_split_based_init']==True) and self.full_prob.jy_opt['all_vars_binary']==True:
                if self.full_prob.jy_opt['all_vars_binary']==True:
                    self.dict_var_name_2_is_integer[var_name]=1
                q=self.h_fg_2_q[h][tup_fg]
                for p in q:
                    self.mapping_h_p_to_fg_vars_use[h][p].append(var_name)
        self.times_lp_times['help_construct_LB_make_vars_5']=time.time()-t1
        t1=time.time()
        t1 = time.time()
        all_new_entries_ignore = []
        vars_names_ignore_set = set(self.vars_names_ignore)  # For O(1) lookups
        dict_update = {}  # Collect all new entries for one bulk update
        dict_update_non_null = {}  # Collect all new entries for one bulk update
        self.mapping_h_p_to_vars_use=dict()
        for h in self.graph_names:
            self.mapping_h_p_to_vars_use[h]=defaultdict(list)
            for q in self.h_q_2_q_id[h]:
                prefix = f"fill_PQ_h={h}_q={q}_p="
                for p in q:
                    var_name = prefix + p
                    self.mapping_h_p_to_vars_use[h][p].append(var_name)

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
        self.MIP_lower_bound = milp_prob.getAttrib('bestbound')

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
        #input('in this call ')
        did_call_gur_warm=False
        GUR_CLASS_lp_prob=[]
        self.new_actions_ignore=None
        #if len(self.full_prob.history_dict['lp_time_compress'])<1 or  self.full_prob.jy_opt['use_julians_custom_lp_solver']<0.5:
        if  self.full_prob.jy_opt['use_julians_custom_lp_solver']<0.5:

            if 1<0:
                input('i dont want to be here since i am usign bounds')

                out_solution=solve_gurobi_lp(self.dict_var_name_2_obj,
                        self.dict_var_con_2_lhs_exog,
                        self.dict_con_name_2_LB,
                        self.dict_var_con_2_lhs_eq,
                        self.dict_con_name_2_eq)
            else:
                #input('in this call 2')

                delta_name_2_lb=dict()
                delta_name_2_ub=dict()
                if self.full_prob.jy_opt['use_delta_in_lp']==True:
                    delta_name_2_lb=self.full_prob.delta_name_2_lb
                    delta_name_2_ub=self.full_prob.delta_name_2_ub
                #input("mooose")
                out_solution=solve_gurobi_lp_bounds(self.dict_var_name_2_obj,
                    self.CLEAN_dict_var_con_2_lhs_exog,
                    self.CLEAN_dict_con_name_2_LB,
                    self.CLEAN_dict_var_con_2_lhs_eq,
                    self.CLEAN_dict_con_name_2_eq,delta_name_2_lb,delta_name_2_ub)

                debug_on=False
                if debug_on:
                    out_solution_2=solve_gurobi_lp(self.dict_var_name_2_obj,
                            self.dict_var_con_2_lhs_exog,
                            self.dict_con_name_2_LB,
                            self.dict_var_con_2_lhs_eq,
                            self.dict_con_name_2_eq)
                    if abs(out_solution['objective']-out_solution_2['objective'])>0.001:
                        print('v1')
                        print(out_solution['objective'])
                        print('v2')
                        print(out_solution_2['objective'])
                        input('errror ')
                #self.delta_name_2_ub=full_input_dict['delta_name_2_ub']
                #self.delta_name_2_lb=full_input_dict['delta_name_2_ub']
                #self.ineq_replaced_by_lb_ub=full_input_dict['ineq_replaced_by_lb_ub']

            self.lp_dual_solution=out_solution['dual_solution']
            self.lp_primal_solution=out_solution['primal_solution']
            self.lp_objective=out_solution['objective']
            self.times_lp_times['GUR_time_pre']=out_solution['time_pre']
            self.times_lp_times['GUR_time_opt']=out_solution['time_opt']
            self.times_lp_times['GUR_time_post']=out_solution['time_post']
            self.lp_time=out_solution['time_opt']
            self.out_solution_JY=out_solution
        else:
            #
            #self.actions_ignore_2=set(self.all_actions)-self.full_prob.all_actions_ever_seen
            #self.actions_ignore_2=self.actions_ignore_2.union(self.full_prob.all_actions_not_source_sink_connected)
            did_call_gur_warm=True
            GUR_CLASS_lp_prob,time_lp_1=warm_start_lp_using_class_gurobi(self.dict_var_name_2_obj,
                    self.dict_var_con_2_lhs_exog,
                    self.dict_con_name_2_LB,
                    self.dict_var_con_2_lhs_eq,
                    self.dict_con_name_2_eq,self.full_prob.all_actions_not_source_sink_connected,self.actions_ignore,self)
            self.lp_primal_solution=GUR_CLASS_lp_prob.lp_primal_solution
            self.lp_objective=GUR_CLASS_lp_prob.lp_objective
            self.lp_dual_solution=GUR_CLASS_lp_prob.lp_dual_solution
            self.lp_time=time_lp_1
            self.new_actions_ignore=list(GUR_CLASS_lp_prob.forbidden_var_names)
            #input('hi im here')
            if len(self.full_prob.history_dict['lp_time_LB'])<1:
                print('GUR_CLASS_lp_prob.hist')
                print(GUR_CLASS_lp_prob.hist)
                #print('self.actions_ignore')
                #print(self.actions_ignore)
                #input('myHist Here')
        if self.new_actions_ignore==None:
            self.new_actions_ignore=[]
            for my_act in self.full_prob.all_actions_not_source_sink_connected:
                if self.lp_primal_solution[my_act]==0:
                    self.new_actions_ignore.append(my_act)
        

        #print('set(self.new_actions_ignore)-set(self.actions_ignore)')
        #print(set(self.new_actions_ignore)-set(self.actions_ignore))
        #print('set(self.actions_ignore)-set(self.new_actions_ignore)')
        #print(set(self.actions_ignore)-set(self.new_actions_ignore))
        #print('len(self.new_actions_ignore)')
        #print(len(self.new_actions_ignore))
        #print('len(self.actions_ignore)')
        #print(len(self.actions_ignore))
        #input('---')
    def call_gurobi_milp_solver(self,use_interior=False):
        out_solution=[]

        delta_name_2_lb=dict()
        delta_name_2_ub=dict()
        #print('self.full_prob.jy_opt[use_delta_in_milp]')
        #print(self.full_prob.jy_opt['use_delta_in_milp'])
        #input('-***--')
        #if self.full_prob.jy_opt['use_delta_in_milp']>0.5:
        delta_name_2_lb=self.full_prob.delta_name_2_lb
        delta_name_2_ub=self.full_prob.delta_name_2_ub
        #else:
        #    print('self.full_prob.jy_opt[use_delta_in_milp]')
        #    print(self.full_prob.jy_opt['use_delta_in_milp'])
        #    input('why')
        #fancy_fixed_keys = [
        #    k for k in self.full_prob.delta_name_2_lb
        #    if k.startswith("fancy")
        #    and self.full_prob.delta_name_2_lb[k] == self.full_prob.delta_name_2_ub.get(k)
        #]

        #fancy_fixed_keysAGAIN = [
        #    k for k in delta_name_2_lb
        #    if k.startswith("fancy")
        #    and delta_name_2_lb[k] == delta_name_2_ub.get(k)
        #]
        #print('fancy_fixed_keysAGAIN')
        #print(fancy_fixed_keysAGAIN)
        #print('fancy_fixed_keys')
        #print(fancy_fixed_keys)
        #print('use_delta_in_milp')
        #print(self.full_prob.jy_opt['use_delta_in_milp'])
        #input('new')

        #lb = self.CLEAN_dict_con_name_2_LB

        # Map constraint name -> LB, filtered to those starting with "side_ineq_use"
        #side_ineq_use_constraints = {
        #    c: lb[c]
        #    for c in lb
        #    if isinstance(c, str) and c.startswith("side_ineq_use")
        #}
#        print('side_ineq_use_constraints')
#        print(side_ineq_use_constraints)
#
        initial_sol_actions=dict()
        if len(self.full_prob.jy_opt['Actions_of_given_sol'])>0.5:
            
            for var in self.dict_var_name_2_obj:
                if var.startswith('act')==True:
                    initial_sol_actions[var]=0    
            for var in self.full_prob.jy_opt['Actions_of_given_sol']:
                initial_sol_actions[var]=1
            
        #input('JUST PRIOR')
        out_solution=solve_gurobi_milp_bounds(self.dict_var_name_2_obj,
            self.CLEAN_dict_var_con_2_lhs_exog,
            self.CLEAN_dict_con_name_2_LB,
            self.CLEAN_dict_var_con_2_lhs_eq,
            self.CLEAN_dict_con_name_2_eq,delta_name_2_lb,delta_name_2_ub,
            self.dict_var_name_2_is_binary,self.dict_var_name_2_is_integer,self.full_prob.jy_opt['max_ILP_time'],use_interior=use_interior,extra_var_name_priority=self.extra_var_name_priority,init_sol=initial_sol_actions)


        self.gurobi_MILP_str=out_solution['gurobi_log_string']
        self.milp_solution=out_solution['primal_solution']
        self.milp_solution_objective_value=out_solution['objective']
        self.times_lp_times['GUR_time_pre']=out_solution['time_pre']
        self.times_lp_times['GUR_time_opt']=out_solution['time_opt']
        self.times_lp_times['GUR_time_post']=out_solution['time_post']
        self.milp_time=out_solution['time_opt']
        self.MIP_lower_bound=out_solution['MIP_lower_bound']
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
                new_val=round(self.Naive_h_f_2_dual[h][f],self.full_prob.jy_opt['roundingDiscretization_num_digits_keep'])
                self.Naive_h_f_2_dual_sig_fig[h][f]=new_val
                if tuple([h,new_val]) not in self.Naive_h_val_2_id[h]:
                    self.Naive_h_val_2_id[h][tuple([h,new_val])]=counter_h
                    counter_h=counter_h+1
                
    def naive_compress_make_f_2_new_f(self):
        self.Naive_H_f_2_new_f=dict()
        #self.lp_dual_solution_agg=dict()#self.lp_dual_solution.copy()
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
                
            #self.make_agg_node_2_nodes()
            #self.make_agg_node_2_nodes()
            #con_name_in_2='flow_in_out_h='+h+"_n=-1"
            #self.lp_dual_solution[con_name_in_2]=0
            #con_name_out_2='flow_in_out_h='+h+"_n=-2

    def Naive_make_i_2_new_f(self):
        self.NAIVE_graph_node_2_agg_node=dict()
        count_orig=dict()
        count_new=dict()
        self.lp_dual_solution_agg=dict()
        for h in self.graph_names:
            self.NAIVE_graph_node_2_agg_node[h]=dict()
            count_orig[h]=len(set(self.graph_node_2_agg_node[h].values()))
            count_new[h]=len(set(self.Naive_H_f_2_new_f[h].values()))
            #print('[h,count_orig[h],count_new[h]]')
            #print([h,count_orig[h],count_new[h]])
            #print('---')
            #agg_source=self.graph_node_2_agg_node[h][self.h_2_sink_id[h]]
            #agg_sink=self.graph_node_2_agg_node[h][self.h_2_source_id[h]]
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

                #f f!=self. and f!=self.h_2_source_id[h]:
                if 1>0:
                    if i ==self.h_2_source_id[h] or i==self.h_2_sink_id[h]:
                        continue
                    #print('i')
                    #print(i)
                    #print('f')
                    #print(f)
                    #print('h')
                    #print(h)
                    #print('self.h_2_sink_id[h]')
                    #print(self.h_2_sink_id[h])
                    #print('self.h_2_source_id[h]')
                    #print(self.h_2_source_id[h])
                    
                    this_con_name='flow_in_out_h='+h+"_n="+my_new_name
                    my_dual_val=self.Naive_h_f_2_dual_sig_fig[h][f]
                    self.lp_dual_solution_agg[this_con_name]=my_dual_val
        
        #print('self.lp_dual_solution_agg')
        #print(self.lp_dual_solution_agg)
        #print('self.lp_dual_solution_agg')
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

    def THINK_aggregate_constraints_dictionary(self):
        input('DONT EXECUTE ME')
        cons_remove=[]
        con_merge=[]
        LP_SOL_LAST=self.full_prob.my_lower_bound_LP.lp_primal_solution
        act_2_finest=dict(set([]))
        finest_2_act=dict()
        #source_id=self.full_prob.jy_opt['']
        special_cons_keep=dict()
        special_cons_merge=dict()
        special_cons_keep_all=[]
        special_cons_merge_all=[]
        all_con_names_remove=[]
        all_con_names_remove_2_new_con_name=dict()
        REVERSE_all_con_names_remove_2_new_con_name=dict()
        Nc=self.full_prob.jy_opt['num_cust_use']
        one_big_merge=False
        if self.full_prob.jy_opt['think_compress']>1.5:
            one_big_merge=True
        for u in range(0,Nc+1):
            special_cons_merge[u]=[]
            special_cons_keep[u]=[]
            for v in range(0,Nc+2):
                act_name='act_'+str(u)+'_'+str(v)
                if act_name in self.full_prob.all_non_null_action:
                    if act_name in self.full_prob.all_actions_ever_seen:#LP_SOL_LAST[act_name]>0:
                        special_cons_keep[u].append(act_name)
                        special_cons_keep_all.append(act_name)
                    else:
                        special_cons_merge[u].append(act_name)
                        for h in self.graph_names:
                            con_name = 'action_match_h='+h+'_p='+act_name
                            all_con_names_remove.append(con_name)
                            new_name='NEW_action_match_h='+h+'u='+str(u)
                            if one_big_merge==True:
                                new_name='NEW_action_match_h='+h+'u=ALL'
                            all_con_names_remove_2_new_con_name[con_name]=new_name
                            if new_name not in REVERSE_all_con_names_remove_2_new_con_name:
                                REVERSE_all_con_names_remove_2_new_con_name[new_name]=[]
                            REVERSE_all_con_names_remove_2_new_con_name[new_name].append(con_name)
        NEW_dict_var_con_2_lhs_eq=dict()
        for (var, con), coeff in self.dict_var_con_2_lhs_eq.items():
            if con in all_con_names_remove_2_new_con_name:
                new_con = all_con_names_remove_2_new_con_name[con]
                new_key = (var, new_con)
            else:
                new_key = (var, con)
            
            NEW_dict_var_con_2_lhs_eq[new_key] = coeff
    
        renamed_cons = set(all_con_names_remove_2_new_con_name)
        new_cons_set = set(all_con_names_remove_2_new_con_name.values())

        # Build new dict in one go
        NEW_con_name_2_eq = {
            con: val for con, val in self.dict_con_name_2_eq.items() if con not in renamed_cons
        }
        print('len(NEW_con_name_2_eq)')
        print(len(NEW_con_name_2_eq))
        print('len(dict_con_name_2_eq)')
        print(len(self.dict_con_name_2_eq))
        # Add zero entries for each new constraint name
        NEW_con_name_2_eq.update({con: 0 for con in new_cons_set})
        self.dict_con_name_2_eq=NEW_con_name_2_eq
        self.dict_var_con_2_lhs_eq=NEW_dict_var_con_2_lhs_eq
        print('len(self.full_prob.all_actions_ever_seen)')
        print(len(self.full_prob.all_actions_ever_seen))
        print('len(self.full_prob.all_non_null_action)')
        print(len(self.full_prob.all_non_null_action))
        print('DOING THIS REMOVAL OPERATION')
        print('DOING THIS REMOVAL OPERATION')
        print('DOING THIS REMOVAL OPERATION')
        print('DOING THIS REMOVAL OPERATION')
        print('DOING THIS REMOVAL OPERATION')
        print('DOING THIS REMOVAL OPERATION')
        print('DOING THIS REMOVAL OPERATION')
        print('DOING THIS REMOVAL OPERATION')
        print('DOING THIS REMOVAL OPERATION')
        print('DOING THIS REMOVAL OPERATION')
        print('DOING THIS REMOVAL OPERATION')
        #input('---')

    def OLD_get_ij_poss_active(self):
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
                if val>0.001:
                    f_out_val.add(f)
                    g_in_val.add(g)
                    both_val.add(f)
                    both_val.add(g)
        

        t_first=time.time()-t1
        t_sec=time.time()      
        self.maybe_mapping=dict()
        #can_same_link=dict()
#
#        for h in self.graph_names:
#            can_same_link[h]=dict()
#            for i in self.graph_node_2_agg_node[h]:
#                can_same_link[h][i]=1
        for h in self.graph_names:
            self.maybe_mapping[h]=defaultdict(float)
            for ij in self.h_ij_2_fg[h].keys():
                i=ij[0]
                j=ij[1]
                f=self.graph_node_2_agg_node[h][i]
                g=self.graph_node_2_agg_node[h][j]
                is_same=self.null_action in self.hij_2_P[h][ij]
                
                #if  is_same or f==g or fg_2_val[tuple([f,g])]>0 or f in both_val or g in both_val:#or f_out_val[f]>0.0001 or g_in_val[g]>.00001):# or is_same:# or is_same:
                if   f==g or is_same or fg_2_val[tuple([f,g])]>0 or f in both_val or g in both_val:#or f_out_val[f]>0.0001 or g_in_val[g]>.00001):# or is_same:# or is_same:
                #if   f==g or is_same  or f in both_val or g in both_val:#or f_out_val[f]>0.0001 or g_in_val[g]>.00001):# or is_same:# or is_same:
                    self.maybe_mapping[h][ij]=1
                #else:
                #    self.maybe_mapping[h][ij]=0
                    #if is_same and can_same_link[h][i]>0:
                    #    can_same_link[h][i]=can_same_link[h][i]-1
                    #    self.maybe_mapping[h][ij]=1
        t_sec=time.time()-t_sec
        #print('[t_first,t_sec]')
        #print([t_first,t_sec])
        #input('---')


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
        #print('[t_first,t_sec]')
        #print([t_first,t_sec])
        #input('---')

    def apply_LA_branching(self):

         

        #make big range sets 

        #make power set; sets

        
    
        my_VRP=self.full_prob.D['my_VRP']
        D=self.full_prob.D
        print(my_VRP)
        Nc=my_VRP.num_cust
        self.dict_pred_gain=dict()
        num_Keep=self.full_prob.jy_opt['LAB_MP_num_ineq_use']

        [ng_neigh_by_cust_power,junk]=naive_get_LA_neigh(my_VRP,self.full_prob.jy_opt['LAB_MP_neigh_use_power'])
        #[ng_neigh_by_cust_all,junk]=naive_get_LA_neigh(my_VRP,self.full_prob.jy_opt['LAB_MP_neigh_use_all'])
        G=set()
        for u in range(0,Nc):

            neighborhood = set(ng_neigh_by_cust_power[u]) | {u}
            for g in power_set(neighborhood):
                if len(g)>1:  # skip empty set and size 1 sets
                    G.add(frozenset(g))
            #for i in range(1,len(ng_neigh_by_cust_all[u])):
            #    g=set(ng_neigh_by_cust_all[u][0:i]).union(set([u]))
            #    G.add(frozenset(g))
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
        num_add=0
        for act in all_actions:
            if act in self.full_prob.delta_name_2_ub and self.full_prob.delta_name_2_ub[act]<0.001:
                #num_already_gone=num_already_gone+1
                continue
            num_add=num_add+1
            _, u_str, v_str = act.split("_")
            u, v = int(u_str), int(v_str)

            relevant_groups = u_in_groups[u] & v_not_in_groups[v]
            for g in relevant_groups:
                self.Z_by_group[g].add(act)

        #print('num_add')
        #print(num_add)
        #input('---')
        costs=D['action2Cost']
        self.cost_val = {
            g: min(costs[act] for act in self.Z_by_group[g]) if self.Z_by_group[g] else float("inf")
            for g in G
        }
        primal_sol_lp=self.full_prob.current_LP_solution
        amount_inside = {
            g: (sum(primal_sol_lp[act] for act in self.Z_by_group[g]))
            for g in G
        }
        if min(amount_inside.values())<.999:
            print('min(amount_inside.values<.9999)')
            print(min(amount_inside.values))
            input('error here')

        frac_amount = {
            g: (
                min(
                    amount_inside[g] - np.floor(amount_inside[g]),
                    np.ceil(amount_inside[g]) - amount_inside[g]
                )
                if amount_inside[g] > 1
                else 1 - amount_inside[g]
            )
            for g in G
        }
        self.pred_val_gain={
            g: self.cost_val[g]*frac_amount[g]/(amount_inside[g])
            for g in G
        }
        pred_val_gain=self.pred_val_gain
        G = {
            g for g in G
            if pred_val_gain[g] > 0.0
            #and amount_inside[g] < 1.999
            and frac_amount[g] > 0.01
            #and len(g)>=2
        }

        G = heapq.nlargest(
            num_Keep,
            G,
            key=lambda g: pred_val_gain[g]
        )
        for g in G:
            print('self.cost_val[g]*frac_amount[g]/amount_inside[g]. '+str( self.cost_val[g]*frac_amount[g]/amount_inside[g]))
            print('[self.cost_val[g],self.frac_amount[g],self.amount_inside[g]]')
            print([self.cost_val[g],frac_amount[g],amount_inside[g]])
            print('g')
            print(g)
            print('---')
        #make auxiliary variables
        #print('len(G)')
        #print(len(G))
        #print('G')
        #input('--')
        #if 1>0:
        #    print('turning off binary and integer for test')
        #    self.dict_var_name_2_is_integer=dict()
        #    self.dict_var_name_2_is_binary=dict()
        #    input('---')
        
        for g in G:
            con_name_eq='NEW_BOUND_Branch_eq'+str(g)
            my_var_name='fancy_branching_var_'+str(g)
            self.dict_var_name_2_obj[my_var_name]=0
            self.dict_con_name_2_eq[con_name_eq]=0
            self.dict_pred_gain[my_var_name]=10000+self.pred_val_gain[g]
            self.dict_var_con_2_lhs_eq[tuple([my_var_name,con_name_eq])]=1#self.delta_con_2_contrib[v_con]
            self.dict_var_name_2_is_integer[my_var_name]=1
            self.full_prob.delta_name_2_lb[my_var_name]=1
            self.full_prob.delta_name_2_ub[my_var_name]=np.inf
            for my_act in self.Z_by_group[g]:
                self.dict_var_con_2_lhs_eq[tuple([my_act,con_name_eq])]=-1
            


    def iterative_ilp_la(self):
        self.full_prob.history_dict['iter_2_act_list']=[]
        maxVarsAdd=self.full_prob.jy_opt['maxVarsAdd_in_ITER'] 

        max_iters=200
        self.num_vars_added=0
        self.num_cons_add=0
        time_internal_list=[]
        OLD_dict_var_name_2_is_integer=self.dict_var_name_2_is_integer.copy()
        OLD_dict_var_name_2_is_binary=self.dict_var_name_2_is_binary.copy()
        self.dict_var_name_2_is_integer=dict()
        self.dict_var_name_2_is_binary=dict()
        G=self.full_prob.G
        Z_by_group=self.full_prob.Z_by_group
        costs=self.full_prob.D['action2Cost']
        self.cost_val = {
            g: min(costs[act] for act in Z_by_group[g]) if Z_by_group[g] else float("inf")
            for g in G
        }
        self.Gall=set([])
        LP_HIST_INTERNAL=[]
        sizes_hist=[]
        num_bin_hist=[]
        DEBUG_ON=False
        num_keep_round=1
        UB_USE_REMOVE=self.full_prob.jy_opt['ub_use_remove']
        if UB_USE_REMOVE<0:
            UB_USE_REMOVE=np.inf
        use_lowe_bound_objective=False
        if use_lowe_bound_objective==True:
            self.add_constraint_on_lb(self.full_prob.my_lower_bound_LP.lp_objective)

        for iter_step in range(0,max_iters):
            self.filter_constraints()
            if DEBUG_ON==True:
                print('DEBUG_ON')
                with open("solver_checkpoint_"+str(iter_step)+"_.pkl", "wb") as f:
                    pickle.dump(self, f)
                print('done writing internal')
            #print('iter')
            #print(iter_step)
            #print('dict_var_name_2_is_binary')
            #print(self.dict_var_name_2_is_binary)
            print('LP_HIST_INTERNAL')
            print(LP_HIST_INTERNAL)
            print('sizes_hist')
            print(sizes_hist)
            print('num_bin_hist')
            print(num_bin_hist)
            print('time_internal_list')
            print(time_internal_list)
            print('--')
            if iter_step==0:
                self.milp_solution=self.full_prob.my_lower_bound_LP.lp_primal_solution
                self.milp_solution_objective_value=self.full_prob.my_lower_bound_LP.lp_objective
                time_internal_list.append(0)
            else:
                self.call_gurobi_milp_solver()
                #self.call_gurobi_milp_solver(True)
                if use_lowe_bound_objective==True:
                    self.dict_con_name_2_LB["constr_lb_obj"]=self.milp_solution_objective_value-0.0001
                time_internal_list.append(self.milp_time)
            if 1>0:
                act_vars = {
                    varname: value
                    for varname, value in self.milp_solution.items()
                    if varname.startswith("act") and value != 0
                }
                self.full_prob.history_dict['iter_2_act_list'].append(act_vars)
            
            fancy_dict = {
                k: v for k, v in self.milp_solution.items()
                if k.startswith("fancy")
            }
            #print('fancy_dict')
            #print(fancy_dict)
            #input('--')
            LP_HIST_INTERNAL.append(self.milp_solution_objective_value)
            num_bin_hist.append(len(self.dict_var_name_2_is_binary))
            print('UB_USE_REMOVE-self.milp_solution_objective_value')
            print(UB_USE_REMOVE-self.milp_solution_objective_value)
            print('UB_USE_REMOVE-self.milp_solution_objective_value')
            if UB_USE_REMOVE-self.milp_solution_objective_value<0.05:
                print('breaking due to lack of a gap')
                break
            fractional_acts = [
                act for act in self.action_2_cost
                if not (abs((val := self.milp_solution.get(act, 0.0))) <= 0.001 or abs(val - 1.0) <= 0.001)
            ]
            if not fractional_acts:
                print('breaking due to integer')
                break

            #[did_find_separ,new_exog_terms,new_action_contrib]=self.full_prob.separate_zero_val_terms(self.milp_solution)
            #if did_find_separ:
                #input('found one ')
            #    for con_name in new_exog_terms:
            ##        self.dict_con_name_2_LB[con_name]=new_exog_terms[con_name]
            #    for (act,con_name) in new_action_contrib:
            #        self.dict_var_con_2_lhs_exog[(act,con_name)]=new_action_contrib[(act,con_name)]


            if DEBUG_ON==True:
                print('DEBUG_ON')
                with open("solver_checkpoint_AFTR_"+str(iter_step)+"_.pkl", "wb") as f:
                    pickle.dump(self, f)
            #input('opt step')
            
            [GNew,GLowerNeeded,G,pred_val_gain,frac_amount,amount_inside,amount_inside_internal]=self.identify_separation(self.milp_solution,num_keep_round)
            if iter_step==0 and len(GLowerNeeded)>0.5:
                input('this means taht something was not added properly')
            sizes_hist.append([len(GNew),len(GLowerNeeded)])
            did_add=False
            if len(GLowerNeeded)>0:
                #input('found one')
                for g in GLowerNeeded:
                    print('g LOWER')
                    print(g)
                    print('[pred_val_gain[g],frac_amount[g],amount_inside[g]]')
                    print([pred_val_gain[g],frac_amount[g],amount_inside[g]])
                    print('----')
                    #self.cuttingPlane_add_bound(g,amount_inside[g])
                    self.cuttingPlane_add_bound_internal(g,amount_inside_internal[g])
                    #cuttingPlane_add_bound_internal
                did_add=True
            else:
                if self.num_vars_added<maxVarsAdd:
                    for g in GNew:
                        #self.cuttingPlane_add_integer(g)
                        print('g UPPER')
                        print(g)
                        print('[pred_val_gain[g],frac_amount[g],amount_inside[g]]')
                        print([pred_val_gain[g],frac_amount[g],amount_inside[g]])
                        print('----')

                        #self.cuttingPlane_add_bound(g,amount_inside[g])
                        #self.cuttingPlane_add_integer(g)
                        #self.cuttingPlane_add_integer_internal(g)
                        self.cuttingPlane_add_bound_internal(g,amount_inside_internal[g])

                        did_add=True
            if did_add==False:
                print('breaking due to lack of addition')
                break
        #self.add_constraint_on_lb(self.milp_solution_objective_value)
        print('FINAL LP_HIST_INTERNAL')
        print(LP_HIST_INTERNAL)
        print('sizes_hist')
        print(sizes_hist)
        print('num_bin_hist')
        print(num_bin_hist)
        print('time_internal_list')
        print(time_internal_list)

        self.full_prob.history_dict['ITER_ILP_LA_TIME']=time_internal_list
        self.full_prob.history_dict['ITER_ILP_LA_sizes_hist']=sizes_hist
        self.full_prob.history_dict['ITER_ILP_LA_num_bin_hist']=num_bin_hist
        self.full_prob.history_dict['ITER_ILP_LA_LP_HIST_INTERNAL']=LP_HIST_INTERNAL
        print('--')
        for i in OLD_dict_var_name_2_is_integer:
            self.dict_var_name_2_is_integer[i]=OLD_dict_var_name_2_is_integer[i]
        for i in OLD_dict_var_name_2_is_binary:
            self.dict_var_name_2_is_binary[i]=OLD_dict_var_name_2_is_binary[i]


    def add_constraint_on_lb(self,val_lb):

        con_name="constr_lb_obj"
        self.dict_con_name_2_LB[con_name]=val_lb-0.0001
        my_acts=set(self.action_2_cost.keys())-set(self.full_prob.delta_name_2_ub.keys())
        for act in my_acts:
            self.dict_var_con_2_lhs_exog[tuple([act,con_name])]=self.action_2_cost[act]

    def cuttingPlane_add_bound(self,g,thresh):
        if g in self.Gall:
            input('error here')
        Z_by_group=self.full_prob.Z_by_group
        low_thresh=np.floor(thresh)
        high_thresh=np.ceil(thresh)

        print('[low_thresh,high_thresh,thresh]')
        print([low_thresh,high_thresh,thresh])

        if low_thresh==0:
            print('in low thresh')
            con_name_ineq='NewLB_'+str(g)+'_'+str(self.num_cons_add)
            self.dict_con_name_2_LB[con_name_ineq]=1

            for act in Z_by_group[g]:
                self.dict_var_con_2_lhs_exog[tuple([act,con_name_ineq])]=1
            self.num_cons_add=self.num_cons_add+1
        else:
            print('in high thresh')
            
            my_var_name='fancy_branching_var_'+str(g)+'_'+str(self.num_vars_added)
            con_name_low='NEW_BOUND_Branch_eq_low'+str(g)+str(self.num_cons_add)
            con_name_high='NEW_BOUND_Branch_eq_high'+str(g)+str(self.num_cons_add+1)
            LG=len(g)
            self.dict_var_name_2_obj[my_var_name]=0
            self.dict_con_name_2_LB[con_name_low]=-LG
            self.dict_con_name_2_LB[con_name_high]=high_thresh
            self.dict_var_con_2_lhs_exog[tuple([my_var_name,con_name_low])]=-LG+low_thresh
            self.dict_var_con_2_lhs_exog[tuple([my_var_name,con_name_high])]=high_thresh-1
            self.dict_var_name_2_is_binary[my_var_name]=1
            for act in Z_by_group[g]:
                self.dict_var_con_2_lhs_exog[tuple([act,con_name_low])]=-1
                self.dict_var_con_2_lhs_exog[tuple([act,con_name_high])]=1
            self.num_cons_add=self.num_cons_add+2
            self.num_vars_added=self.num_vars_added+1

    def cuttingPlane_add_bound_internal(self,g,thresh):
        if g in self.Gall:
            input('error here')
        Z_by_group_inside=self.full_prob.Z_by_group_inside
        low_thresh=np.floor(thresh)
        high_thresh=np.ceil(thresh)

        print('[low_thresh,high_thresh,thresh]')
        print([low_thresh,high_thresh,thresh])
        print('g')
        print(g)
        if high_thresh==len(g):
            print('in low thresh')
            con_name_ineq='NewLB_'+str(g)+'_'+str(self.num_cons_add)
           # if con_name_ineq in  self.dict_con_name_2_LB:
           #     print('already here')
           #     input('ok this no sense')
            self.dict_con_name_2_LB[con_name_ineq]=-len(g)+0.9999
            #print('g')
            #print(g)
            #print('con_name_ineq')
            #print(con_name_ineq)
            #print('self.dict_con_name_2_LB[con_name_ineq]')
            #print(self.dict_con_name_2_LB[con_name_ineq])
            cur_val=0
            for act in Z_by_group_inside[g]:
                self.dict_var_con_2_lhs_exog[tuple([act,con_name_ineq])]=-1
                cur_val-=self.milp_solution[act]
             #   print('act '+act)
             #   print(self.milp_solution[act])
            #print('cur_val')
            #print(cur_val)
            if cur_val>=self.dict_con_name_2_LB[con_name_ineq]:
                input('VERY WRONG')
            self.num_cons_add=self.num_cons_add+1
            #input('LOOK HERE')
        else:
            print('in high thresh')
            
            
            #input('this is not wrong but I dont think I want to be here')
            my_var_name='fancy_branching_var_'+str(g)+'_'+str(self.num_vars_added)
            con_name_low='NEW_BOUND_Branch_eq_low'+str(g)+str(self.num_cons_add)
            con_name_high='NEW_BOUND_Branch_eq_high'+str(g)+str(self.num_cons_add+1)
            LG=len(g)
            self.dict_var_name_2_obj[my_var_name]=0
            self.dict_con_name_2_LB[con_name_low]=-LG
            self.dict_con_name_2_LB[con_name_high]=high_thresh
            self.dict_var_con_2_lhs_exog[tuple([my_var_name,con_name_low])]=-LG+low_thresh
            self.dict_var_con_2_lhs_exog[tuple([my_var_name,con_name_high])]=high_thresh
            self.dict_var_name_2_is_binary[my_var_name]=1
            self.extra_var_name_priority[my_var_name]=1000-len(self.extra_var_name_priority)
            #print('self.dict_var_con_2_lhs_exog[tuple([my_var_name,con_name_low])]')
            #print(self.dict_var_con_2_lhs_exog[tuple([my_var_name,con_name_low])])
            #print('self.dict_var_con_2_lhs_exog[tuple([my_var_name,con_name_high])]')
            #print(self.dict_var_con_2_lhs_exog[tuple([my_var_name,con_name_high])])
            for act in Z_by_group_inside[g]:
                self.dict_var_con_2_lhs_exog[tuple([act,con_name_low])]=-1
                self.dict_var_con_2_lhs_exog[tuple([act,con_name_high])]=1
                #print('act '+act)
            self.num_cons_add=self.num_cons_add+2
            self.num_vars_added=self.num_vars_added+1


    def cuttingPlane_add_integer(self,g):
        if g in self.Gall:
            input('error here')
        Z_by_group=self.full_prob.Z_by_group
        con_name_eq='NEW_BOUND_Branch_eq'+str(g)
        my_var_name='fancy_branching_var_'+str(g)
        self.dict_var_name_2_obj[my_var_name]=0
        self.dict_con_name_2_eq[con_name_eq]=0
        self.dict_var_con_2_lhs_eq[tuple([my_var_name,con_name_eq])]=1#self.delta_con_2_contrib[v_con]
        self.dict_var_name_2_is_integer[my_var_name]=1
        self.full_prob.delta_name_2_lb[my_var_name]=1
        self.full_prob.delta_name_2_ub[my_var_name]=np.inf
        for my_act in Z_by_group[g]:
            self.dict_var_con_2_lhs_eq[tuple([my_act,con_name_eq])]=-1
        self.num_vars_added=self.num_vars_added+1

    def cuttingPlane_add_integer_internal(self,g):
        if g in self.Gall:
            input('error here')
        Z_by_group=self.full_prob.Z_by_group_inside
        con_name_eq='NEW_BOUND_Branch_eq'+str(g)
        my_var_name='fancy_branching_var_'+str(g)
        self.dict_var_name_2_obj[my_var_name]=0
        self.dict_con_name_2_eq[con_name_eq]=0
        self.dict_var_con_2_lhs_eq[tuple([my_var_name,con_name_eq])]=1#self.delta_con_2_contrib[v_con]
        self.dict_var_name_2_is_integer[my_var_name]=1
        self.full_prob.delta_name_2_lb[my_var_name]=0
        self.full_prob.delta_name_2_ub[my_var_name]=len(g)-1
        self.extra_var_name_priority[my_var_name]=1000-len(self.extra_var_name_priority)

        for my_act in Z_by_group[g]:
            self.dict_var_con_2_lhs_eq[tuple([my_act,con_name_eq])]=-1
        self.num_vars_added=self.num_vars_added+1

    def identify_separation(self,primal_sol_lp,num_keep):
        Nc=self.full_prob.D['my_VRP'].num_cust
        costs=self.full_prob.D['action2Cost']
        Z_by_group=self.full_prob.Z_by_group
        Z_by_group_internal=self.full_prob.Z_by_group_inside
        G=self.full_prob.G
        delta_keys = set(self.full_prob.delta_name_2_ub.keys())

        for g in Z_by_group:
            Z_by_group[g] = Z_by_group[g] - delta_keys
            Z_by_group_internal[g] = Z_by_group_internal[g] - delta_keys
        #self.cost_val = {
        #    g: min(costs[act] for act in Z_by_group[g]) if Z_by_group[g] else float("inf")
        #    for g in G
        #}
        
        empty_groups = [g for g in G if not Z_by_group.get(g)]

        if len(empty_groups)>0:
            print('Z_by_group')
            print(Z_by_group)
            print('empty_groups')
            print(empty_groups)
            for g in G:
                if len(empty_groups)<0.5:
                    input('wrong')
            print('hihih')
            for g in empty_groups:
                print('g')
                print(g)
                print('len(g)')
                print(len(g))
            input('---')
        self.cost_val = {
            g: min(costs[act] for act in Z_by_group[g])
            for g in G
        }
        tol_low, tol_high = 0.999, 1.001
        sum_by_u, sum_by_v = defaultdict(float), defaultdict(float)

        if "null_action" in self.all_non_null_action:
            input('error here')
        # Accumulate
        for act in self.all_non_null_action:
            _, u, v = act.split("_")
            u, v = int(u), int(v)
            val = primal_sol_lp.get(act, 0.0)
            if u < Nc: sum_by_u[u] += val
            if v < Nc: sum_by_v[v] += val
       
        print('passing this set options')


        for act in self.action_2_cost:
            if act in primal_sol_lp and act in self.full_prob.delta_name_2_ub and primal_sol_lp[act]>0.0001:
                print('act')
                print(act)
                print('self.full_prob.delta_name_2_ub[act]')
                print(self.full_prob.delta_name_2_ub[act])
                print('primal_sol_lp[act]')
                print(primal_sol_lp[act])
                input('wrong')
        amount_inside = {
            g: (sum(primal_sol_lp[act] for act in Z_by_group[g]))
            for g in G
        }

        amount_inside_internal = {
            g: (sum(primal_sol_lp[act] for act in Z_by_group_internal[g]))
            for g in G
        }
        
        frac_amount = {
            g: (
                min(
                    amount_inside[g] - np.floor(amount_inside[g]),
                    np.ceil(amount_inside[g]) - amount_inside[g]
                )
                if amount_inside[g] > 1
                else 1 - amount_inside[g]
            )
            for g in G
        }

        #frac_amount = {
        #    g: (
        #        min(
        #            np.inf,
        #            np.ceil(amount_inside[g]) - amount_inside[g]
        #        )
        #        if amount_inside[g] > 1
        #        else 1 - amount_inside[g]
        #    )
        #    for g in G
        #}


        
        self.pred_val_gain={
            g: self.cost_val[g]*frac_amount[g]/(amount_inside[g])
            for g in G
        }
        if 1<0:
            self.pred_val_gain = {
            g: self.cost_val[g] #/ amount_inside[g])
            if (frac_amount.get(g, 0.0) > 1e-2)
            else 0.0
            for g in G
        }
    
        pred_val_gain=self.pred_val_gain
        GNewOrig = {
            g for g in G
            if pred_val_gain[g] > 0.0 and amount_inside[g] > 1.001 and frac_amount[g] > 0.01
        }
        GlowerOrig = {
            g for g in G
            if pred_val_gain[g] > 0.0
            and amount_inside[g] < .999
            and frac_amount[g] > 0.01
        }
        GNew = heapq.nlargest(
            num_keep,
            GNewOrig,
            key=lambda g: pred_val_gain[g]
        )
        GLower = heapq.nlargest(
            num_keep,
            GlowerOrig,
            key=lambda g: pred_val_gain[g]
        )
        

        
        return [GNew,GLower,G,pred_val_gain,frac_amount,amount_inside,amount_inside_internal]



    def iterative_ilp_la_DIVE(self):
        self.full_prob.history_dict['iter_2_act_list']=[]
        maxVarsAdd=self.full_prob.jy_opt['maxVarsAdd_in_ITER'] 

        max_iters=200
        UB_USE_REMOVE=self.full_prob.jy_opt['ub_use_remove']

        self.num_vars_added=0
        self.num_cons_add=0
        time_internal_list=[]
        OLD_dict_var_name_2_is_integer=self.dict_var_name_2_is_integer.copy()
        OLD_dict_var_name_2_is_binary=self.dict_var_name_2_is_binary.copy()
        self.dict_var_name_2_is_integer=dict()
        self.dict_var_name_2_is_binary=dict()
        G=self.full_prob.G
        Z_by_group=self.full_prob.Z_by_group
        costs=self.full_prob.D['action2Cost']
        self.cost_val = {
            g: min(costs[act] for act in Z_by_group[g]) if Z_by_group[g] else float("inf")
            for g in G
        }
        self.Gall=set([])
        LP_HIST_INTERNAL=[]
        sizes_hist=[]
        num_bin_hist=[]
        num_keep_round=1
        use_lowe_bound_objective=False
        thus_far_always_one_branch=True
        terms_no_change=set()
        break_time=np.inf
        #if use_lowe_bound_objective==True:
        #    self.add_constraint_on_lb(self.full_prob.my_lower_bound_LP.lp_objective)
        for iter_step in range(0,max_iters):
            self.filter_constraints()
           
            print('LP_HIST_INTERNAL')
            print(LP_HIST_INTERNAL)
            print('sizes_hist')
            print(sizes_hist)
            print('num_bin_hist')
            print(num_bin_hist)
            print('time_internal_list')
            print(time_internal_list)
            print('--')
            self.milp_solution=None
            if iter_step==0:
                self.milp_solution=self.full_prob.my_lower_bound_LP.lp_primal_solution
                self.milp_solution_objective_value=self.full_prob.my_lower_bound_LP.lp_objective
                time_internal_list.append(0)
            else:
                self.milp_solution=None
                my_var_flip = set(self.extra_var_name_priority.keys()) & set(self.dict_var_name_2_is_binary.keys())
                if len(my_var_flip)>1:
                    input('error here')
                
                if len(my_var_flip)==1:
                    my_var_flip=list(my_var_flip)
                    my_var_flip=my_var_flip[0]
                    del self.dict_var_name_2_is_binary[my_var_flip]
                    #print('my_var_flip')
                    #print(my_var_flip)
                    #input('in here')
                    t_lp_over=time.time()

                    try:
                        self.full_prob.delta_name_2_lb[my_var_flip]=0
                        self.full_prob.delta_name_2_ub[my_var_flip]=0
                        self.call_gurobi_solver()
                        sol1 = copy.deepcopy(self.lp_primal_solution)
                        obj_1=self.lp_objective
                        time_1=0
                    except Exception as e:
                        print(f"Error while solving with Gurobi: {e}")
                        obj_1 = np.inf
                        sol1 = None
                        time_1=0

                    try:
                        print('my_var_flip')
                        print(my_var_flip)
                        self.full_prob.delta_name_2_lb[my_var_flip]=1
                        self.full_prob.delta_name_2_ub[my_var_flip]=1
                        self.call_gurobi_solver()
                        obj_2 = self.lp_objective
                        sol2 = copy.deepcopy(self.lp_primal_solution)
                        time_2=0
                    except Exception as e:
                        print(f"Error while solving with Gurobi: {e}")
                        obj_2 = np.inf
                        sol2 = None
                        time_2=0
                    print('obj_1,obj_2')
                    print(obj_1,obj_2)
                    print('obj_1,obj_2')
                    #input('---')
                    if obj_1<obj_2:
                        self.milp_solution=sol1
                        self.milp_solution_objective_value=obj_1
                        self.full_prob.delta_name_2_lb[my_var_flip]=0
                        self.full_prob.delta_name_2_ub[my_var_flip]=0
                        #self.milp_time=time_1+time_2
                        if obj_2>UB_USE_REMOVE and thus_far_always_one_branch==True:
                            terms_no_change.add(my_var_flip)
                        else:
                            thus_far_always_one_branch=False
                            print('turning off')
                            break_time=min([break_time,iter_step])
                    else:
                        self.milp_solution=sol2
                        self.milp_solution_objective_value=obj_2
                        if obj_1>UB_USE_REMOVE and thus_far_always_one_branch==True:
                            terms_no_change.add(my_var_flip)
                        else:
                            thus_far_always_one_branch=False
                            print('turning off 2')
                            break_time=min([break_time,iter_step])
                    t_lp_over=time.time()-t_lp_over

                    if 0.5<self.full_prob.jy_opt['use_diving_ineq']:
                        if obj_1>UB_USE_REMOVE and obj_2<=UB_USE_REMOVE:
                            self.make_dive_ineq(0,my_var_flip)


                        if obj_2>UB_USE_REMOVE and obj_1<=UB_USE_REMOVE:
                            self.make_dive_ineq(1,my_var_flip)

                        if obj_1>UB_USE_REMOVE and obj_2>UB_USE_REMOVE:
                            self.make_dive_ineq(2,my_var_flip)
                        #input('making dive ineq')
                    time_internal_list.append(t_lp_over)

                else:
                    #input('in the place after a cutting added ')
                    self.call_gurobi_solver()
                    self.milp_solution=self.lp_primal_solution
                    self.milp_solution_objective_value=self.lp_objective
                    
                    time_internal_list.append(self.lp_time)
            if 1>0:
                act_vars = {
                    varname: value
                    for varname, value in self.milp_solution.items()
                    if varname.startswith("act") and value != 0
                }
                self.full_prob.history_dict['iter_2_act_list'].append(act_vars)
            
            LP_HIST_INTERNAL.append(self.milp_solution_objective_value)
            num_bin_hist.append(len(self.dict_var_name_2_is_binary))
            print('UB_USE_REMOVE-self.milp_solution_objective_value')
            print(UB_USE_REMOVE-self.milp_solution_objective_value)
            print('UB_USE_REMOVE-self.milp_solution_objective_value')
            if UB_USE_REMOVE-self.milp_solution_objective_value<0.05:
                print('breaking due to lack of a gap')
                break
            fractional_acts = [
                act for act in self.action_2_cost
                if not (abs((val := self.milp_solution.get(act, 0.0))) <= 0.001 or abs(val - 1.0) <= 0.001)
            ]
            if not fractional_acts:
                print('breaking due to integer')
                break


            #input('opt step')
            
            [GNew,GLowerNeeded,G,pred_val_gain,frac_amount,amount_inside,amount_inside_internal]=self.identify_separation(self.milp_solution,num_keep_round)
            if iter_step==0 and len(GLowerNeeded)>0.5 and self.full_prob.jy_opt['use_ineq']>0.5:
                input('this means taht something was not added properly')
            sizes_hist.append([len(GNew),len(GLowerNeeded)])
            did_add=False
            if len(GLowerNeeded)>0:
                #input('found one')
                for g in GLowerNeeded:
                    print('g LOWER')
                    print(g)
                    print('[pred_val_gain[g],frac_amount[g],amount_inside[g]]')
                    print([pred_val_gain[g],frac_amount[g],amount_inside[g]])
                    print('----')
                    #self.cuttingPlane_add_bound(g,amount_inside[g])
                    self.cuttingPlane_add_bound_internal(g,amount_inside_internal[g])
                    #cuttingPlane_add_bound_internal
                    #input('this is maybe the issue paues to look')

                did_add=True
            else:
                if self.num_vars_added<maxVarsAdd:
                    for g in GNew:
                        #self.cuttingPlane_add_integer(g)
                        print('g UPPER')
                        print(g)
                        print('[pred_val_gain[g],frac_amount[g],amount_inside[g]]')
                        print([pred_val_gain[g],frac_amount[g],amount_inside[g]])
                        print('----')

                        #self.cuttingPlane_add_bound(g,amount_inside[g])
                        #self.cuttingPlane_add_integer(g)
                        #self.cuttingPlane_add_integer_internal(g)
                        self.cuttingPlane_add_bound_internal(g,amount_inside_internal[g])
                        did_add=True
            

            

            if did_add==False:
                print('breaking due to lack of addition')
                break
        #self.add_constraint_on_lb(self.milp_solution_objective_value)
        print('FINAL LP_HIST_INTERNAL')
        print(LP_HIST_INTERNAL)
        print('sizes_hist')
        print(sizes_hist)
        print('num_bin_hist')
        print(num_bin_hist)
        print('time_internal_list')
        print(time_internal_list)

        self.full_prob.history_dict['ITER_ILP_LA_TIME']=time_internal_list
        self.full_prob.history_dict['ITER_ILP_LA_sizes_hist']=sizes_hist
        self.full_prob.history_dict['ITER_ILP_LA_num_bin_hist']=num_bin_hist
        self.full_prob.history_dict['ITER_ILP_LA_LP_HIST_INTERNAL']=LP_HIST_INTERNAL
        print('--')
        print('self.extra_var_name_priority')
        print(self.extra_var_name_priority)
        print('terms_no_change')
        print(terms_no_change)
        print('break_time')
        print(break_time)
        print('----')
        for i in self.extra_var_name_priority:
            if i not in terms_no_change:
                self.dict_var_name_2_is_binary[i]=1
                self.full_prob.delta_name_2_lb[i]=0
                self.full_prob.delta_name_2_ub[i]=1
            #else:
            #    print('LOOK LOOOK')
            #    print('LOOK LOOOK')
            #    print('LOOK LOOOK')
            #    print('LOOK LOOOK')
            #    print('LOOK LOOOK')
            #    print('i')
            #    print(i)
            #    print('self.full_prob.delta_name_2_lb[i]')
            #    print(self.full_prob.delta_name_2_lb[i])
            #    print('self.full_prob.delta_name_2_ub[i]')
            #    print(self.full_prob.delta_name_2_ub[i])
            #    print('------')
        for i in OLD_dict_var_name_2_is_integer:
            self.dict_var_name_2_is_integer[i]=OLD_dict_var_name_2_is_integer[i]
        for i in OLD_dict_var_name_2_is_binary:
            self.dict_var_name_2_is_binary[i]=OLD_dict_var_name_2_is_binary[i]
        print('extra_var_name_priority')
        print(self.extra_var_name_priority)
        print('extra_var_name_priority')
        #fancy_fixed_keys = [
        #    k for k in self.full_prob.delta_name_2_lb
        #    if k.startswith("fancy")
        #    and self.full_prob.delta_name_2_lb[k] == self.full_prob.delta_name_2_ub.get(k)
        #]
        #print(fancy_fixed_keys)
        #input('fancy_fixed_keys IN DIVE')

        #print('HOLD')
        #b = self.dict_con_name_2_LB

        # Map constraint name -> LB, filtered to those starting with "side_ineq_use"
        #side_ineq_use_constraints = {
        #    c: lb[c]
        #    for c in lb
        #    if isinstance(c, str) and c.startswith("side_ineq_use")
        #}

        # Get (var, con) pairs (works if function returns a dict or an iterable of tuples)
        #var_con_pairs = self.dict_var_con_2_lhs_exog
        #iter_pairs = self.dict_var_con_2_lhs_exog#var_con_pairs.keys() if isinstance(var_con_pairs, dict) else var_con_pairs

        # Map (var, con) -> 1 for constraints with names starting with "side_ineq_use"
        #side_ineq_use_constraints_2 = {
        #    (v, c): iter_pairs[(v,c)]
        #    for (v, c) in iter_pairs
        #    if isinstance(c, str) and c.startswith("side_ineq_use")
        #}

        #print('side_ineq_use_constraints_2')
        #print(side_ineq_use_constraints_2)
        #print('side_ineq_use_constraints')
        #print(side_ineq_use_constraints)
        #input('-INSDIE OF THE DIVING--')

    def make_dive_ineq(self,side_invalid,var_new):
        my_constr_name='side_ineq_use'+str(len(self.extra_var_name_priority))+'_'+str(np.ceil(np.random.rand()*10000))+'_'+str(side_invalid)
        
        my_rhs=0
        ext_vars=set(self.extra_var_name_priority)-set([var_new])
        print('ext_vars')
        print(ext_vars)
        print('var_new')
        print(var_new)
        print('--')
        for var_name in ext_vars:
            is_ub=1
            my_tup=tuple([var_name,my_constr_name])
            contrib_val=1
            if self.full_prob.delta_name_2_lb[var_name]==1:
                is_ub=0
                my_rhs=my_rhs-1
                contrib_val=-1
            self.full_prob.action_con_2_contrib[my_tup]=contrib_val
            self.dict_var_con_2_lhs_exog[my_tup]=contrib_val
        my_tup_2=tuple([var_new,my_constr_name])
        #print('----')
        print('my_rhs orig ')
        print(my_rhs)
        #print('side_invalid')
        #print(side_invalid)
        #print('--')
        if side_invalid==0: #this means taht this must be 1
            #-(1-x_k)+\sum_{i<k}[ub=0](x_i)+\sum_{i<K}[lb=1](1-x_i) >=0
            #x_k+\sum_{i<k}[ub=0]x_i+\sum_{i<k}[lb=1](-x_i)>=1-\sum_{i<k}[lb=1]
            my_rhs=my_rhs+1
            self.full_prob.action_con_2_contrib[my_tup_2]=1
            self.dict_var_con_2_lhs_exog[my_tup_2]=1
            print('side 0')
        if side_invalid==1: #must be zero
            #-x_k+\sum_{i<k}[ub=0](x_i)+\sum_{i<K}[lb=1](1-x_i) >=0
            #-x_k+\sum_{i<k}[ub=0]x_i+\sum_{i<k}[lb=1](-x_i)>=-\sum_{i<k}[lb=1]
            self.full_prob.action_con_2_contrib[my_tup_2]=-1
            
            self.dict_var_con_2_lhs_exog[my_tup_2]=-1
            print('my_tup_2')
            print(my_tup_2)
            print('self.dict_var_con_2_lhs_exog[my_tup_2]')
            print(self.dict_var_con_2_lhs_exog[my_tup_2])
            print('side 1')

        if side_invalid==2:
            #\sum_{i<k}[ub=0](x_i)+\sum_{i<K}[lb=1](1-x_i) >=1
            #\sum_{i<k}[ub=0]x_i+\sum_{i<k}[lb=1](-x_i)>=1-\sum_{i<k}[lb=1]
            my_rhs=my_rhs+1
            print('side 2')
        
        self.full_prob.exog_name_2_rhs[my_constr_name]=my_rhs
        self.dict_con_name_2_LB[my_constr_name]=my_rhs

        #print('*****')
        #print('my_rhs')
        #print(my_rhs)
        #input('----')








    def compute_path_based_rc(self) -> None:
        """
        Computes self.path_based_rc[act] = reduced_cost(act) + sum_h best_by_h[h][act],
        where best_by_h[h][act] = min_{(i,j): hij_2_P[h][(i,j)]=act} [ d_src[i] + d_sink[j] + w_ij ],
        with w_ij = graph_act_2_weight[h][act] on edge (i,j).
        Assumes no negative cycles.
        """

        # --- Guards ----------------------------------------------------------------
        if self.null_action in self.all_non_null_action:
            raise ValueError("null_action must not be in all_non_null_action")

        # --- 1) Base reduced costs per action -------------------------------------
        act_2_red_cost = {}
        for act in self.all_non_null_action:
            if act in self.full_prob.delta_name_2_ub and self.full_prob.delta_name_2_ub[act] < 1e-3:
                act_2_red_cost[act] = math.inf
                #print('act')
                #print(act)
                #input('found one ')
            else:
                act_2_red_cost[act] = self.action_2_cost[act]

        # Subtract dual contributions: (act, con) -> coeff
        # NOTE: self.action_con_2_contrib is a dict, not a function.
        for (act, con), coeff in self.action_con_2_contrib.items():
            # If act is infinite (tight UB), keep it infinite.
            if act_2_red_cost.get(act, math.inf) is not math.inf:
                act_2_red_cost[act] -= self.lp_dual_solution[con] * coeff

        # --- 2) Build per-graph action weights ------------------------------------
        # For each h: weight for 'null_action' is 0; non-null actions map from duals.
        BIGPOS = 9.9999999999e11
        EPS = 1e-5
        EPS_MULT_END=0#2000
        graph_act_2_weight = {}
        for h in self.graph_name_2_nodes:
            w = {self.null_action: 0.0}
            graph_act_2_weight[h] = w

        for act in self.all_non_null_action:
            # If reduced cost is inf (disallowed/very tight), give huge edge weights.
            if act_2_red_cost[act] == math.inf:
                for h in self.graph_name_2_nodes:
                    graph_act_2_weight[h][act] = BIGPOS
                #input('found me here')
            else:
                for h in self.graph_name_2_nodes:
                    con = f"action_match_h={h}_p={act}"
                    # Add the dual once to the action's reduced cost, and set per-graph weight.
                    act_2_red_cost[act] += self.lp_dual_solution[con]
                    # Edge weight used in shortest-paths (slightly nudged with EPS).
                    graph_act_2_weight[h][act] = -self.lp_dual_solution[con] + EPS

        # --- 3) For each graph h, compute best edge-based marginal per action ------
        # best_by_h[h][act] = min over edges labeled act of  d_src[i] + d_sink[j] + w_ij
        best_by_h = {}

        for h in self.graph_name_2_nodes:
            mapping = self.hij_2_P[h]                 # dict: (i,j) -> act
            act_w = graph_act_2_weight[h]             # dict: act -> weight

            # Build weighted DiGraph quickly
            # Edges: (i, j, weight=act_w[mapping[(i,j)]])
            weighted_edges = []
            for (i, j), act_L in mapping.items():
                act=act_L[0]

                w = act_w.get(act)
                if w is not None:
                    weighted_edges.append((i, j, w))
                else:
                    input('error here')
            G = nx.DiGraph()
            G.add_weighted_edges_from(weighted_edges)

            source = self.full_prob.h_2_source_id[h]
            sink = self.full_prob.h_2_sink_id[h]

            # Distances FROM source (Bellman–Ford, supports negative edges)
            try:
                d_src = nx.single_source_bellman_ford_path_length(G, source, weight="weight")
            except nx.NetworkXUnbounded:
                raise ValueError(f"Negative-weight cycle reachable from source in graph h={h}")

            # Distances TO sink: run from sink on the reversed graph
            Grev = G.reverse(copy=False)
            try:
                d_to_sink_rev = nx.single_source_bellman_ford_path_length(Grev, sink, weight="weight")
            except nx.NetworkXUnbounded:
                raise ValueError(f"Negative-weight cycle can reach sink in graph h={h} (on reversed graph)")

            # Edge marginal cost: d_src[i] + d_sink[j] + w_ij
            edge_min_marg = {}
            for (i, j), act in mapping.items():
                if not G.has_edge(i, j):
                    continue  # edge weight wasn't available
                w_ij = G[i][j]["weight"]
                v1=d_src.get(i, math.inf)
                v2=d_to_sink_rev.get(j, math.inf)
                v3=w_ij
                
                val = v1+v2+v3 #+ d_to_sink_rev.get(j, math.inf) + w_ij
                #if val==np.inf:
                #    print('[i,j]')
                #    print([i,j])
                #    print('[v1,v2,v3]')
                #    print([v1,v2,v3])
                #    input('---')
                edge_min_marg[(i, j)] = val

            # Aggregate min per action
            best = defaultdict(lambda: math.inf)
            for (i, j), actL in mapping.items():
                act=actL[0]

                v = edge_min_marg.get((i, j))
                if v==None:
                    input('error here 2')
                if  v < best[act]:
                    best[act] = v
                #print('best[act]')
                #print(best[act])
                #print('act')
                #print(act)
                ##print(v)
                #input('---')
            best_by_h[h] = best

        # --- 4) Final path-based reduced costs per action --------------------------
        self.path_based_rc = {}
        for act in self.all_non_null_action:
            total = act_2_red_cost[act]
            if total == math.inf:
                self.path_based_rc[act] = math.inf
                continue
            # Sum the best contribution across all h (if none, it stays +inf and total -> +inf)
            for h in self.graph_name_2_nodes:
                total += best_by_h[h][act]
            total=total-EPS_MULT_END*EPS
            self.path_based_rc[act] = total



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
