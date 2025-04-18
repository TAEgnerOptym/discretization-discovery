
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

class lower_bound_LP_milp:


    def __init__(self,full_prob,graph_node_2_agg_node,OPT_do_ILP,OPT_use_psi):
        full_input_dict=full_prob.full_input_dict
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
        #print('self.h_2_source_id')
        #print(self.h_2_source_id)
        #print('self.h_2_sink_id')
        #print(self.h_2_sink_id)
        #input('---')
        #self.init_agg_graph_node_2_agg_node:  rganize first by h then by  node name then by the aggregated node
        #self.agg_graph_agg_node_2_node=full_input_dict['init_agg_graph_agg_node_2_node']

        self.graph_names=full_input_dict['allGraphNames']
        self.OPT_use_psi=OPT_use_psi
        self.OPT_do_ilp=OPT_do_ILP
        self.construct_LB_or_ILP(self.OPT_use_psi,self.OPT_do_ilp)

        if self.OPT_do_ilp==0:
            self.make_LP()
        else:
            self.solve_milp()
           

    def make_agg_node_2_nodes(self):
        self.agg_node_2_nodes=dict()
        for h in self.graph_names:
            self.agg_node_2_nodes[h]=dict()
            #print('self.graph_node_2_agg_node[h]')
            #print(self.graph_node_2_agg_node[h])
            #print('self.graph_node_2_agg_node[h]')
            #p#rint('self.graph_node_2_agg_node[h].keys()')
            #print(self.graph_node_2_agg_node[h].keys())
            for i in self.graph_node_2_agg_node[h]:
                
                #print('i above')

                f=self.graph_node_2_agg_node[h][i]
                #print('f ')
                #print(f)
                
                #input('----')
                if f not in self.agg_node_2_nodes[h]:
                    
                    self.agg_node_2_nodes[h][f]=set([])
                self.agg_node_2_nodes[h][f].add(i)

    def make_edge_fg_2_ij_reverse(self):
        self.h_fg_2_ij=dict()
        self.h_ij_2_fg=dict()
        for h in self.graph_names:
            self.h_fg_2_ij[h]=dict()
            self.h_ij_2_fg[h]=dict()
            
            edges_compact_h=self.hij_2_P[h].keys()
            #print('self.graph_node_2_agg_node[h].keys()')
            #print(self.graph_node_2_agg_node[h].keys())
            #print('edges_compact_h')
            #print(edges_compact_h)
            for tup_ij in edges_compact_h:
                #print('tup_ij')
                #print(tup_ij)
                #print('type(tup_ij)')
                #print(type(tup_ij))
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
            all_fg_edges=self.h_fg_2_ij[h]
            self.h_fg_2_q[h]=dict()
            self.h_q_2_fg[h]=dict()
            self.h_q_2_q_id[h]=dict()
            count=0
            for  tup_fg in all_fg_edges:
                my_set=set([])
                #print('-----')
                #print('len(self.h_fg_2_ij[h][tup_fg])')
                #print(len(self.h_fg_2_ij[h][tup_fg]))
                #input('----')
                for tup_ij in  self.h_fg_2_ij[h][tup_fg]:
                    #print('self.hij_2_P[h][tup_ij]')
                    #print(self.hij_2_P[h][tup_ij])
                    #print('len(self.hij_2_P[h][tup_ij]')
                    #print(len(self.hij_2_P[h][tup_ij]))
                    for p in self.hij_2_P[h][tup_ij]:
                        my_set.add(p)
                #print('list(my_set)')
                #print(list(my_set))
                #print('before')
                my_tup_pq=tuple(sorted(list(my_set)))
                #print('my_tup_pq')
                #print(my_tup_pq)
                #print('afer')
                #input('---')
                self.h_fg_2_q[h][tup_fg]=my_tup_pq
                
                if my_tup_pq not in self.h_q_2_fg[h]:
                    self.h_q_2_fg[h][my_tup_pq]=set([])
                    self.h_q_2_q_id[h][my_tup_pq]=tuple([h,count])
                    count=count+1
                self.h_q_2_fg[h][my_tup_pq].add(tup_fg)
    def make_mappings(self):
        
        self.make_agg_node_2_nodes()
        self.make_edge_fg_2_ij_reverse()
        self.make_h_fg_2_p_reverse()

    def help_construct_LB_make_vars(self):
        use_psi=self.OPT_use_psi
        do_ilp=self.OPT_do_ilp
        self.dict_var_name_2_is_binary=defaultdict(int)
        self.names_binary=[]
        if use_psi==True and do_ilp==True:
            for var_name in self.all_primitive_vars:
                self.dict_var_name_2_obj[var_name]=0
                self.dict_var_name_2_is_binary[var_name]=1
                #self.dict_var_name_2_LB[[var_name]]=0
                #self.dict_var_name_2_UB[[var_name]]=1
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
            #self.dict_var_name_2_LB[[var_name]]=0
            #self.dict_var_name_2_UB[[var_name]]=np.inf
        for h in self.graph_names:
            for tup_fg in self.h_fg_2_ij[h]:
                
                f=tup_fg[0]
                g=tup_fg[1]
            
                var_name='EDGE_h='+h+'_f='+f+'_g='+g
                #print(var_name)
                #input('var_name')

                self.dict_var_name_2_obj[var_name]=0
                #self.dict_var_name_2_LB[[var_name]]=0
                #self.dict_var_name_2_UB[[var_name]]=np.inf
        for h in self.graph_names:
            for q in self.h_q_2_q_id[h]:
                p_list=list(q)

                for p in p_list:
                    
                    var_name='fill_PQ_h='+h+'_q='+str(q)+'_p='+p
                    #print('var_name')
                    #print(var_name)
                    #input('---')
                    self.dict_var_name_2_obj[var_name]=0
                    #self.dict_var_name_2_LB[[var_name]]=0
                    #self.dict_var_name_2_UB[[var_name]]=np.inf
       # print('type(self.dict_var_name_2_obj)')
       # print(type(self.dict_var_name_2_obj))
       # print('type(self.dict_var_name_2_obj)')
       # #input('---')    
    #given dictionarys that define below I want to write a linear program in pulp.  I then want to solve this linear program as an LP and get the dual solution
    #   self.dict_var_name_2_obj:  maps a variable name as a string to a constant for objective 
    #   all variables are non-negative
    #   dict_con_name_2_LB:  maps inequality constraint name to lower bound  on RHS for constraint of the form >= 
    #   dict_con_name_2_eq:  maps constraint name RHS for equality constraint
    #   dict_var_con_2_lhs_exog:  each key is a tuple of var_name,constraint_name and the associated value is the coefficient associted with the var_name in the constraint_name for the inequality constraint with >= form
    #   dict_var_con_2_lhs_eq:  each key is a tuple of var_name,constraint_name and the associated value is the coefficient associted with the var_name in the constraint_name for the equaltity constraint
    def help_construct_UB_LB_con(self):
        
        for exog_name in self.exog_name_2_rhs:
            self.dict_con_name_2_LB[exog_name]=self.exog_name_2_rhs[exog_name]
            #if debug_on==True:
            #    if exog_name[0:8]=='time_uv_':# or exog_name[0:7]=='cap_uv_' :#or exog_name[0:4]=='time' or exog_name[0:3]=='cap':
            #        self.dict_con_name_2_LB[exog_name]=self.exog_name_2_rhs[exog_name]
            #        print('firing ')
            #        print(exog_name)
            #        #input('---')
            #    else:
            #        print('NOT firing ')
            #        print(exog_name)
            #        #input('---')
                
        if self.OPT_use_psi==True and self.OPT_do_ilp==True:
            for con_name in self.all_integCon:
                self.dict_con_name_2_eq[con_name]=0
        for h in self.graph_names:
            this_sink=self.graph_node_2_agg_node[h][self.h_2_sink_id[h]]
            this_source=self.graph_node_2_agg_node[h][self.h_2_source_id[h]]
            nodes_use=set(self.agg_node_2_nodes[h])-set([this_sink,this_source])
            for n in nodes_use:
                con_name='flow_in_out_h='+h+"_n="+n
                #print('con_name')
                #print(con_name)
                #input('---')
                self.dict_con_name_2_eq[con_name]=0
        for h in self.graph_names:
            #nodes_use=set(self.agg_node_2_nodes[h])-set([self.h_2_sink_id[h],self.h_2_source_id[h]])
            
            for q in self.h_q_2_fg[h]:
                con_name='equiv_class='+h+"_q="+str(q)
                self.dict_con_name_2_eq[con_name]=0
        for h in self.graph_names:
            for p in self.all_non_null_action:
                con_name='action_match_h='+h+"_p="+p
                self.dict_con_name_2_eq[con_name]=0

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
         for h in self.graph_names:
            for q in self.h_q_2_fg[h]:
                con_name='equiv_class='+h+"_q="+str(q)
                for p in q:
                    var_name_2='fill_PQ_h='+h+'_q='+str(q)+'_p='+p
                    self.dict_var_con_2_lhs_eq[tuple([var_name_2,con_name])]=-1
                for e in self.h_q_2_fg[h][q]:
                    f=e[0]
                    g=e[1]

                    var_name_edge='EDGE_h='+h+'_f='+f+'_g='+g
                    self.dict_var_con_2_lhs_eq[tuple([var_name_edge,con_name])]=1

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
        self.help_construct_UB_LB_con()
        
        self.construct_constraints_exog()
        self.construct_constraints_flow_in_out()
        self.construct_constraints_actions_match_compact()
        self.construct_constraints_actions_match_flow()
        if use_psi==True:
            self.construct_constraints_prim()

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
        dict_var_name_2_obj=self.dict_var_name_2_obj
        dict_con_name_2_LB=self.dict_con_name_2_LB
        dict_con_name_2_eq=self.dict_con_name_2_eq
        dict_var_con_2_lhs_exog=self.dict_var_con_2_lhs_exog
        dict_var_con_2_lhs_eq=self.dict_var_con_2_lhs_eq
        dict_var_name_2_is_binary=self.dict_var_name_2_is_binary
        import xpress as xp
        xp.init('C:/xpressmp/bin/xpauth.xpr')
        milp_prob = xp.problem("MILP_Problem")

        # Create decision variables based on the input. 
        # If a variable is binary, declare it as such, otherwise as continuous (nonnegative).
        var_dict = {}

        for var_name, obj_coeff in dict_var_name_2_obj.items():
            if dict_var_name_2_is_binary.get(var_name, 0):
                var_dict[var_name] = xp.var(name=var_name, vartype=xp.binary)
            else:
                var_dict[var_name] = xp.var(name=var_name, lb=0)

        # Define the objective function: minimize sum(objective_coefficient * variable)
        objective = xp.Sum(dict_var_name_2_obj[var_name] * var_dict[var_name] 
                            for var_name in dict_var_name_2_obj)
        milp_prob.setObjective(objective, sense=xp.minimize)

        # --- Add inequality constraints (of the form: expression >= lower bound) ---
        for con_name in dict_con_name_2_LB:
            expr = xp.Sum(dict_var_con_2_lhs_exog.get((var_name, con_name), 0) * var_dict[var_name] 
                            for var_name in var_dict)
            milp_prob.addConstraint(expr >= dict_con_name_2_LB[con_name], name=con_name + "_ineq")

        # --- Add equality constraints ---
        for con_name in dict_con_name_2_eq:
            expr = xp.Sum(dict_var_con_2_lhs_eq.get((var_name, con_name), 0) * var_dict[var_name] 
                            for var_name in var_dict)
            milp_prob.addConstraint(expr == dict_con_name_2_eq[con_name], name=con_name + "_eq")

        # --- Solve the MILP ---
        start_time = time.time()
        milp_prob.solve()
        end_time = time.time()

        self.milp_time = end_time - start_time
        self.milp_prob = milp_prob
        self.milp_solution = {var_name: var_dict[var_name].getSolution() for var_name in var_dict}
        self.milp_solution_status = milp_prob.getProbStatus()
        self.milp_solution_objective_value = milp_prob.getObjVal()
    def make_LP(self):
        dict_var_name_2_obj=self.dict_var_name_2_obj
        dict_var_con_2_lhs_exog=self.dict_var_con_2_lhs_exog
        dict_con_name_2_LB=self.dict_con_name_2_LB
        dict_var_con_2_lhs_eq=self.dict_var_con_2_lhs_eq
        dict_con_name_2_eq=self.dict_con_name_2_eq

        debug_on=False
        if debug_on==True:
            dict_con_name_2_eq=dict()
            dict_var_con_2_lhs_eq=dict()
        # --- Build the LP model ---
        lp_prob = pulp.LpProblem("MyLP", pulp.LpMinimize)

        # Create decision variables (all non-negative)
        var_dict = {}
        for var_name, coeff in dict_var_name_2_obj.items():
            var_dict[var_name] = pulp.LpVariable(var_name, lowBound=0)

        # Define the objective function (minimize sum(obj_coeff * var))
        lp_prob += pulp.lpSum(dict_var_name_2_obj[var_name] * var_dict[var_name]
                            for var_name in dict_var_name_2_obj), "Objective"

        # --- Add inequality constraints (>=) ---
        # Group terms for each inequality constraint.
        ineq_expressions = {}
        did_find_2=False
        #input('----')
        for (var_name, con_name), coeff in dict_var_con_2_lhs_exog.items():
            ineq_expressions.setdefault(con_name, 0)
            ineq_expressions[con_name] += coeff * var_dict[var_name]
            if con_name=='exog_min_veh_':

                did_find_2=True

        did_find=False
        for con_name, expr in ineq_expressions.items():
            if con_name in dict_con_name_2_LB:

                if con_name=='exog_min_veh_':
                    did_find=True
                    #input('---')
                lp_prob += expr >= dict_con_name_2_LB[con_name], con_name + "_ineq"

        if did_find==False:
            input('this is odd')
        # --- Add equality constraints ---
        # Group terms for each equality constraint.
        eq_expressions = {}
        for (var_name, con_name), coeff in dict_var_con_2_lhs_eq.items():
            eq_expressions.setdefault(con_name, 0)
            eq_expressions[con_name] += coeff * var_dict[var_name]

        # Add each equality constraint to the model.
        for con_name, expr in eq_expressions.items():
            if con_name in dict_con_name_2_eq:
                lp_prob += expr == dict_con_name_2_eq[con_name], con_name #+ "_eq"
                
                #input('----')
        # --- Solve the LP ---
        # Using the default CBC solver here.
        start_time=time.time()
        solver = pulp.PULP_CBC_CMD(msg=False)
        lp_prob.solve(solver)
        end_time=time.time()
        self.lp_time=end_time-start_time
        self.lp_prob=lp_prob
        self.lp_primal_solution=dict()
        for var_name, var in var_dict.items():
            self.lp_primal_solution[var_name]=var.varValue
        self.lp_status=pulp.LpStatus[lp_prob.status]
        self.lp_objective= pulp.value(lp_prob.objective)
        self.lp_dual_solution=dict()
        for con_name, constraint in lp_prob.constraints.items():
            self.lp_dual_solution[con_name]=constraint.pi
        if self.lp_status=='Infeasible':
            input('HOLD')
    def make_xpress_LP(self):
        import xpress as xp
        xp.init('C:/xpressmp/bin/xpauth.xpr')
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

        # Create decision variables (all non-negative)
        var_dict = {}
        for var_name, coeff in dict_var_name_2_obj.items():
            var_dict[var_name] = xp.var(name=var_name, lb=0)

        # Define the objective function (minimize sum(obj_coeff * var))
        objective = xp.Sum(dict_var_name_2_obj[var_name] * var_dict[var_name]
                            for var_name in dict_var_name_2_obj)
        lp_prob.setObjective(objective, sense=xp.minimize)

        # --- Add inequality constraints (>=) ---
        # Group terms for each inequality constraint.
        ineq_expressions = {}
        did_find_2 = False
        for (var_name, con_name), coeff in dict_var_con_2_lhs_exog.items():
            if con_name not in ineq_expressions:
                ineq_expressions[con_name] = 0
            ineq_expressions[con_name] += coeff * var_dict[var_name]
            if con_name == 'exog_min_veh_':
                did_find_2 = True

        did_find = False
        for con_name, expr in ineq_expressions.items():
            if con_name in dict_con_name_2_LB:
                if con_name == 'exog_min_veh_':
                    did_find = True
                lp_prob.addConstraint(expr >= dict_con_name_2_LB[con_name], name=con_name + "_ineq")

        if did_find == False:
            input('this is odd')

        # --- Add equality constraints ---
        # Group terms for each equality constraint.
        eq_expressions = {}
        for (var_name, con_name), coeff in dict_var_con_2_lhs_eq.items():
            if con_name not in eq_expressions:
                eq_expressions[con_name] = 0
            eq_expressions[con_name] += coeff * var_dict[var_name]

        # Add each equality constraint to the model.
        for con_name, expr in eq_expressions.items():
            if con_name in dict_con_name_2_eq:
                lp_prob.addConstraint(expr == dict_con_name_2_eq[con_name], name=con_name)

        # --- Solve the LP ---
        start_time = time.time()
        lp_prob.solve()
        end_time = time.time()

        self.lp_time = end_time - start_time
        self.lp_prob = lp_prob
        self.lp_primal_solution = dict()
        for var_name, var in var_dict.items():
            self.lp_primal_solution[var_name] = var.getSolution()

        self.lp_status = lp_prob.getProbStatus()
        self.lp_objective = lp_prob.getObjVal()
        self.lp_dual_solution = dict()

        for con_name in lp_prob.getConstraints():
            self.lp_dual_solution[con_name] = lp_prob.getDual(con_name)

        if self.lp_status == 'Infeasible':
            input('HOLD')