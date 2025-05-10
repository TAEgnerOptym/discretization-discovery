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

        self.times_lp_times=dict()
        self.full_prob=full_prob
        full_input_dict=full_prob.full_input_dict
        self.actions_ignore=full_prob.actions_ignore

        self.dict_2_action_ignore=defaultdict(int)
        self.vars_names_ignore=self.actions_ignore.copy()

        self.graph_node_2_agg_node=graph_node_2_agg_node
        self.all_delta=full_input_dict['allDelta']

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

        self.construct_mapping_pq_action()
        self.construct_LB_or_ILP(self.OPT_use_psi,self.OPT_do_ilp)
        self.filter_constraints()

            
        if self.OPT_do_ilp==0:
            
            self.call_gurobi_solver()
            self.naive_compress_get_pi_by_h_node()
            self.naive_compress_make_f_2_new_f()
            self.Naive_make_i_2_new_f()

        else:
            self.call_gurobi_milp_solver()
            

    def  construct_mapping_pq_action(self):
        
        self.act_2_merge_name=dict()
        self.merge_name_2_act=dict()
        Nc=self.full_prob.jy_opt['num_cust_use']
        self.act_2_merge_name[self.null_action]=self.null_action
        self.merge_name_2_act[self.null_action]=self.null_action
        self.merge_name_2_act_non_null=dict()
        one_big_merge=True
        print('len(self.full_prob.all_actions_ever_seen)')
        print(len(self.full_prob.all_actions_ever_seen))
        #input('---')
        if self.full_prob.jy_opt['think_compress']>1.5:
            one_big_merge=False

        if one_big_merge==True:
            self.merge_name_2_act['ALL_unseen_terms']=[]
            self.merge_name_2_act_non_null['ALL_unseen_terms']=[]
        else:
            for u in range(0,Nc+1):
                self.merge_name_2_act['unseen_terms_'+str(u)]=[]
                self.merge_name_2_act_non_null['unseen_terms_'+str(u)]=[]


        for u in range(0,Nc+1):
            merge_name='ALL_unseen_terms'
            if one_big_merge==False:
                merge_name='unseen_terms_'+str(u)
            
            for v in range(0,Nc+2):
                act_name='act_'+str(u)+'_'+str(v)
                if act_name not in self.all_actions:
                    continue
                if act_name in self.full_prob.all_actions_ever_seen:
                    self.act_2_merge_name[act_name]=act_name
                    self.merge_name_2_act[act_name]=[act_name]
                    self.merge_name_2_act_non_null[act_name]=[act_name]
                else:
                    self.act_2_merge_name[act_name]=merge_name
                    self.merge_name_2_act[merge_name].append(act_name)
                    self.merge_name_2_act_non_null[merge_name].append(act_name)
        #print('self.merge_name_2_act_non_null.keys()')
        #print(self.merge_name_2_act_non_null.keys())
        #input('--')
                    


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
            
            all_fg_edges = self.h_fg_2_ij[h]
            self.h_fg_2_q[h] = dict()
            self.h_q_2_fg[h] = dict()
            self.h_q_2_q_id[h] = dict()
            count = 0

            for tup_fg in all_fg_edges:
                # Collect all sets of p-values quickly
                sets_to_union = (self.hij_2_P[h][tup_ij] for tup_ij in self.h_fg_2_ij[h][tup_fg])
                # Use set union in bulk
                my_set_in = set().union(*sets_to_union)
                my_set=set()
                for s in my_set_in:
                    my_set.add(self.act_2_merge_name[s])

                
                my_set=list(my_set)
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
        
        t1=time.time()
        var_type_count=dict()
        var_type_count['all_non_null_action']=0
        for var_name in self.all_non_null_action:
            self.dict_var_name_2_obj[var_name]=self.action_2_cost[var_name]
            var_type_count['all_non_null_action']=var_type_count['all_non_null_action']+1
            #print('var_name')
            #print(var_name)
            #input('')
        self.times_lp_times['help_construct_LB_make_vars_2']=time.time()-t1
        t1=time.time()
        #for 
        if do_ilp==True:
            for var_name in self.all_non_null_action:
                self.dict_var_name_2_is_binary[var_name]=1
        self.times_lp_times['help_construct_LB_make_vars_3']=time.time()-t1
        var_type_count['all_delta']=0
        t1=time.time()
        for var_name in self.all_delta:
            self.dict_var_name_2_obj[var_name]=0
            var_type_count['all_delta']=var_type_count['all_delta']+1

        self.times_lp_times['help_construct_LB_make_vars_4']=time.time()-t1
        var_type_count['all_edges']=0
        t1=time.time()
        for h in self.graph_names:
            for tup_fg in self.h_fg_2_ij[h]:
                
                f=tup_fg[0]
                g=tup_fg[1]
                var_name='EDGE_h='+h+'_f='+f+'_g='+g
                self.dict_var_name_2_obj[var_name]=0
                var_type_count['all_edges']=var_type_count['all_edges']+1

        self.times_lp_times['help_construct_LB_make_vars_5']=time.time()-t1
        t1=time.time()
        t1 = time.time()
        all_new_entries_ignore = []
        vars_names_ignore_set = set(self.vars_names_ignore)  # For O(1) lookups
        dict_update = {}  # Collect all new entries for one bulk update
        dict_update_non_null = {}  # Collect all new entries for one bulk update
        var_type_count['filler']=0
        for h in self.graph_names:
            for q in self.h_q_2_q_id[h]:
                #prefix = f"fill_PQ_h={h}_q={q}_p="
                prefix = "fill_PQ_h="+h+"_q="+str(q)+"_p="
                for pg in q:
                    var_name = prefix + pg
                    self.dict_var_name_2_obj[var_name]=0
                    var_type_count['filler']=var_type_count['filler']+1
                    if self.full_prob.jy_opt['all_vars_binary']==True and pg!=self.null_action :
                        self.dict_var_name_2_is_binary[var_name]=1

        print(var_type_count)
        #input('--')
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
            new_entries = {prefix + p: 0 for p in self.merge_name_2_act_non_null}
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
        #constraints_1 = {
         #   (p, f"action_match_h={h}_p={pg}"): -1
        #    for h in self.graph_names
         #   for pg in self.merge_name_2_act_non_null
         #   for p in self.merge_name_2_act_non_null[pg]
        #}
        constraints_1=dict()
        for h in self.graph_names:
            for pg in self.merge_name_2_act_non_null:
                con_name="action_match_h="+h+"_p="+pg
                for p in self.merge_name_2_act_non_null[pg]:
                    my_tup=tuple([p,con_name])
                    constraints_1[my_tup]=-1
                    if p=='a':
                        print(p)
                        print('pg')
                        print(pg)
                        input('error here')
        # Second set: iterate over each graph h, each q in h_q_2_fg[h], and then each p in q,
        # but precompute the fixed prefix for the variable name and constraint name so that the inner loop
        # over p (which is very large) does only the minimal string concatenation.
        
        #constraints_2 = {
        #    (prefix_var + pg, prefix_cons + pg): 1
        #    for h in self.graph_names
         #   for q in self.h_q_2_fg[h]
       #     for prefix_var, prefix_cons in [(f"fill_PQ_h={h}_q={q}_p=", f"action_match_h={h}_p=")]
       #     for pg in q if pg !=self.null_action
            #for p in self.merge_name_2_act_non_null[pg] if p != self.null_action
        #}
        constraints_2=dict()
        for h in self.graph_names:
            for q in self.h_q_2_fg[h]:
                for pg in q:
                    if pg ==self.null_action:
                        continue
                    var_name= "fill_PQ_h="+h+"_q="+str(q)+"_p="+str(pg)
                    con_name="action_match_h="+h+"_p="+str(pg)
                    my_tup_name=tuple([var_name,con_name])
                    constraints_2[my_tup_name]=1


        # Update the existing dictionary (without removing existing entries).
        self.dict_var_con_2_lhs_eq.update(constraints_1)
        self.dict_var_con_2_lhs_eq.update(constraints_2)
        
    def construct_constraints_actions_match_flow(self):
        # Build constraints from actions (p in q) with value -1.
        
        constraints_from_actions=dict()
        for h in self.graph_names:
            for q in self.h_q_2_fg[h]:
                for pg in q:
                    var_name= "fill_PQ_h="+h+"_q="+str(q)+"_p="+str(pg)
                    con_name="equiv_class="+h+"_q="+str(q)
                    my_tup=tuple([var_name,con_name])
                    constraints_from_actions[my_tup]=-1
                #prefix = f"fill_PQ_h={h}_q={q}_p="
                #cons = f"equiv_class={h}_q={q}"
                # For each p in q, simply concatenate the precomputed prefix with p.
                #constraints_from_actions.update({(prefix + pg, cons): -1 for pg in q})




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

        self.make_mappings()

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
        for my_tup in self.dict_var_con_2_lhs_eq:
            if my_tup[0]=='a':
                input('error here')
        #self.times_lp_times['help_construct_UB_LB_con']=time.time()-t1
        t1=time.time()
        self.construct_constraints_exog()
        self.times_lp_times['construct_constraints_exog']=time.time()-t1
        t1=time.time()

        self.construct_constraints_flow_in_out()
        for my_tup in self.dict_var_con_2_lhs_eq:
            if my_tup[0]=='a':
                input('error here')
        self.times_lp_times['construct_constraints_flow_in_out']=time.time()-t1
        t1=time.time()

        self.construct_constraints_actions_match_compact()
        for my_tup in self.dict_var_con_2_lhs_eq:
            if my_tup[0]=='a':
                input('error here')
        self.times_lp_times['construct_constraints_actions_match_compact']=time.time()-t1
        t1=time.time()

        self.construct_constraints_actions_match_flow()
        for my_tup in self.dict_var_con_2_lhs_eq:
            if my_tup[0]=='a':
                input('error here')
        self.times_lp_times['construct_constraints_actions_match_compact']=time.time()-t1
        t1=time.time()

        if use_psi==True:
            self.construct_constraints_prim()
        self.times_lp_times['construct_constraints_prim']=time.time()-t1

    def call_gurobi_solver(self):
    
        if self.full_prob.jy_opt['use_julians_custom_lp_solver']<0.5:

            out_solution=solve_gurobi_lp_bounds(self.dict_var_name_2_obj,
                self.CLEAN_dict_var_con_2_lhs_exog,
                self.CLEAN_dict_con_name_2_LB,
                self.CLEAN_dict_var_con_2_lhs_eq,
                self.CLEAN_dict_con_name_2_eq,self.full_prob.delta_name_2_lb,self.full_prob.delta_name_2_ub)

                
            self.lp_dual_solution=out_solution['dual_solution']
            self.lp_primal_solution=out_solution['primal_solution']
            self.lp_objective=out_solution['objective']
            self.times_lp_times['GUR_time_pre']=out_solution['time_pre']
            self.times_lp_times['GUR_time_opt']=out_solution['time_opt']
            self.times_lp_times['GUR_time_post']=out_solution['time_post']
            self.lp_time=out_solution['time_opt']
            
        self.new_actions_ignore=[]#self.full_prob.all_actions_not_source_sink_connected.copy()

        for my_act in self.full_prob.all_actions_not_source_sink_connected:
            if self.lp_primal_solution[my_act]==0:
                self.new_actions_ignore.append(my_act)

    def call_gurobi_milp_solver(self):
        out_solution=[]

        
        out_solution=solve_gurobi_milp_bounds(self.dict_var_name_2_obj,
            self.CLEAN_dict_var_con_2_lhs_exog,
            self.CLEAN_dict_con_name_2_LB,
            self.CLEAN_dict_var_con_2_lhs_eq,
            self.CLEAN_dict_con_name_2_eq,self.full_prob.delta_name_2_lb,self.full_prob.delta_name_2_ub,
            self.dict_var_name_2_is_binary,self.full_prob.jy_opt['max_ILP_time'])


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
            this_fg_sink=self.graph_node_2_agg_node[h][self.h_2_sink_id[h]]
            this_fg_source=self.graph_node_2_agg_node[h][self.h_2_source_id[h]]
            f_except_source_sink=set(self.agg_node_2_nodes[h])-set([this_fg_sink,this_fg_source])
            for f in f_except_source_sink:
                this_con_name='flow_in_out_h='+h+"_n="+f

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


            for i in self.graph_node_2_agg_node[h]:
                f=self.graph_node_2_agg_node[h][i]
                
                
                if f not in self.Naive_H_f_2_new_f[h]:
                    print('not fuond')
                    input('error here ')

                my_new_name=str(self.Naive_H_f_2_new_f[h][f])
                my_new_name=my_new_name.replace(" ", "_")
                self.NAIVE_graph_node_2_agg_node[h][i]=my_new_name

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
