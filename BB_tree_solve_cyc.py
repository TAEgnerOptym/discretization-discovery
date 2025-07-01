from collections import defaultdict
from full_solver import full_solver
import heapq
import random
import json
import sys
import numpy as np
import itertools
sys.path.append("pre_process")
from naive_pre import *

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


class aux_constr:

    def __init__(self,g,rhs_amount_unsigned,is_ub):
        #g the 
        
        self.g=g
        self.my_name_pos='NEW_BOUND_Branch_pos_'+str(g)+'__'+str(is_ub)+'__'+str(rhs_amount_unsigned)
        self.my_name_neg='NEW_BOUND_Branch_neg_'+str(g)+'__'+str(is_ub)+'__'+str(rhs_amount_unsigned)
        self.is_ub=is_ub
        self.rhs_amount_unsigned=rhs_amount_unsigned
        self.my_var_name='Delta_var_branch_amount_pos'+str(g)
        self.my_var_name='Delta_var_branch_amount_neg'+str(g)
        self.my_var_name_slack_pos='Delta_var_branch_slack_pos'+str(g)
        self.my_var_name_slack_neg='Delta_var_branch_slack_neg'+str(g)
class BBNode:
    def __init__(self,my_BB_tree, parent,my_aux_constrs):
        """
        Initialize a branch-and-bound node.

        Parameters:
            must_be_zero (set[str]): Set of x_p actions that must be zero.
            must_be_one (set[str]): Set of x_p actions that must be one.
            parent (BBNode): Optional parent node.
        """
        self.my_BB_tree=my_BB_tree
        self.D=self.my_BB_tree.D.copy()

        self.M=self.my_BB_tree.BIG_M
        self.act_src_map=self.my_BB_tree.act_src_map
        #self.must_be_zero = must_be_zero
        #self.must_be_one = must_be_one
        self.parent = parent
        self.my_aux_constrs=my_aux_constrs
        self.add_aux_constrs()
        self.add_aux_bounds()
        self.actions_ignore=None#self.self.all_actions_not_source_sink_connected

        jy_opt=self.my_BB_tree.my_params
        debug_bef_vals=dict()
        self.par_lb=0
        self.parent_depth=-1
        self.parent_slack=0
        self.my_aux_constrs=my_aux_constrs
        if parent!=None:
            self.actions_ignore=self.parent.my_solver.actions_ignore
            self.parent_depth=self.parent.depth
            self.parent_slack=self.parent.slack

            self.par_lb=self.parent.LB
            PARENTS_graph_node_2_agg_node=dict()
            
            for h in self.parent.graph_node_2_agg_node:
                PARENTS_graph_node_2_agg_node[h]=dict()
                #print('h')
                #print(h)
                #print('len(self.parent.graph_node_2_agg_node:[h])')
                #print(len(self.parent.graph_node_2_agg_node[h]))
                debug_bef_vals[h]=len(set(self.parent.graph_node_2_agg_node[h].values()))
                for n in self.parent.graph_node_2_agg_node[h]:
                    PARENTS_graph_node_2_agg_node[h][n]=self.parent.graph_node_2_agg_node[h][n]
            self.D['initGraphNode2AggNode']=PARENTS_graph_node_2_agg_node
        self.depth=self.parent_depth+1
        self.my_solver=full_solver(self.D,jy_opt,'junk',self.actions_ignore)
        debug_aft_vals=dict()
        self.graph_node_2_agg_node=dict()#self.my_solver.final_graph_node_2_agg_node
        for h in self.my_solver.graph_node_2_agg_node:
            self.graph_node_2_agg_node[h]=dict()
            for n in self.my_solver.graph_node_2_agg_node[h]:
                self.graph_node_2_agg_node[h][n]=self.my_solver.graph_node_2_agg_node[h][n]
            debug_aft_vals[h]=len(set(self.graph_node_2_agg_node[h].values()))

        self.LB=self.my_solver.history_dict['lblp_lower'][-1]
        self.LB_init=self.my_solver.history_dict['lblp_lower'][0]
        if self.parent!=None and ((self.LB<-.001+self.parent.LB) or (self.LB_init<-.001+self.parent.LB)):
            print('self.LB')
            print(self.LB)
            print('self.parent.LB')
            print(self.parent.LB)
            print('self.LB_init')
            print(self.LB_init)
            print('self.must_be_zero')
            print(self.must_be_zero)
            print('self.parent.must_be_zero')
            print(self.parent.must_be_zero)
            print('self.must_be_one')
            print(self.must_be_one)
            print('self.parent.must_be_one')
            print(self.parent.must_be_one)
            input('BIG error here')
        self.primal_sol_act=dict()
        self.primal_sol_act_sparse=defaultdict(float)
        my_lp_sol=self.my_solver.my_lower_bound_LP.lp_primal_solution
        self.my_lp_sol=my_lp_sol
        self.frac_var_2_value=dict()
        branch_term_cur=None
        biggest_prod=0
        for my_act in self.my_solver.my_lower_bound_LP.all_actions:
            y=my_lp_sol[my_act]
            self.primal_sol_act[my_act]=y
            if y>0.0001:
                self.primal_sol_act_sparse[my_act]=y
            if y<.999 and y>0.001:
                self.frac_var_2_value[my_act]=my_lp_sol[my_act]
                if y*(1-y)>biggest_prod:
                    branch_term_cur=my_act
                    biggest_prod=y*(1-y)
        self.term_2_branch_on=branch_term_cur
        self.eval_separ()
        self.data=dict()
        self.data['LB']=self.LB
        self.data['LB_init']=self.LB_init
        self.data['parent_lb']=self.par_lb
        self.data['depth']=self.depth
        print(self.my_solver.history_dict['sum_lp_value_project'])
        #input('---')
        self.slack=self.my_solver.history_dict['sum_lp_value_project'][-2]
        self.data['slack']=self.slack
        self.data['parent_slack']=self.parent_slack
        self.data['prob_sizes_at_start_last_iter']=self.my_solver.history_dict['prob_sizes_at_start'][-1]
        self.data['primal_sol_act_sparse']=self.primal_sol_act_sparse
        #self.data['must_be_zero']=list(self.must_be_zero)
        #self.data['must_be_one']=list(self.must_be_one)
        self.data['lb_hist']=list(self.my_solver.history_dict['lblp_lower'])
    
    def add_aux_constrs(self):

        self.D['actionCon2Contrib']=self.my_BB_tree.D['actionCon2Contrib'].copy()
        self.D['allExogNames']=self.my_BB_tree.D['allExogNames'].copy()
        self.D['exogName2Rhs']=self.my_BB_tree.D['exogName2Rhs'].copy()
        self.D['deltaCon2Contrib']=self.my_BB_tree.D['deltaCon2Contrib'].copy()
        self.D['allDelta']=self.my_BB_tree.D['allDelta'].copy()
        self.D['delta_name_2_ub']=self.my_BB_tree.D['delta_name_2_ub'].copy()
        self.D['delta_name_2_lb']=self.my_BB_tree.D['delta_name_2_lb'].copy()
        self.D['con_names_fancy_branch']=[]
        self.D['var_int_names_fancy_branch']=[]
        self.D['var_cont_names_fancy_branch']=[]
        for my_con in self.my_aux_constrs:
            my_con_name_pos=my_con.my_name_pos
            my_con_name_neg=my_con.my_name_neg
            my_var_name=my_con.my_var_name
            my_var_slack_pos=my_con.my_var_name_slack_pos
            my_var_slack_neg=my_con.my_var_name_slack_neg
            g=my_con.g

            self.D['allExogNames'].append(my_con_name_neg)
            self.D['allExogNames'].append(my_con_name_pos)
            self.D['exogName2Rhs'][my_con_name_neg]=0
            self.D['exogName2Rhs'][my_con_name_pos]=0
            self.D['deltaCon2Contrib'][tuple([my_var_name,my_con_name_pos])]=-1
            self.D['deltaCon2Contrib'][tuple([my_var_slack_pos,my_con_name_pos])]=-1
            self.D['deltaCon2Contrib'][tuple([my_var_slack_neg,my_con_name_pos])]=1

            self.D['deltaCon2Contrib'][tuple([my_var_name,my_con_name_neg])]=1
            self.D['deltaCon2Contrib'][tuple([my_var_slack_pos,my_con_name_neg])]=1
            self.D['deltaCon2Contrib'][tuple([my_var_slack_neg,my_con_name_neg])]=-1

            self.D['allDelta'].append(my_var_name)
            self.D['allDelta'].append(my_var_slack_pos)
            self.D['allDelta'].append(my_var_slack_neg)

            self.D['delta_name_2_ub'][my_var_slack_pos]=np.inf
            self.D['delta_name_2_lb'][my_var_slack_pos]=0
            self.D['delta_name_2_ub'][my_var_slack_neg]=np.inf
            self.D['delta_name_2_lb'][my_var_slack_neg]=0
            self.D['delta_name_2_ub'][my_var_name]=np.inf
            self.D['delta_name_2_lb'][my_var_name]=0
            self.D['con_names_fancy_branch'].append(my_con_name_neg)
            self.D['con_names_fancy_branch'].append(my_con_name_pos)
            self.D['var_int_names_fancy_branch'].append(my_var_name)
            self.D['var_cont_names_fancy_branch'].append(my_var_slack_pos)
            self.D['var_cont_names_fancy_branch'].append(my_var_slack_neg)
            for act in self.my_BB_tree.Z_by_group[g]:
                self.D['actionCon2Contrib'][tuple([act,my_con_name_neg])]=-1
                self.D['actionCon2Contrib'][tuple([act,my_con_name_pos])]=1

    def add_aux_bounds(self):
        
        for my_con in self.my_aux_constrs:
            my_var_name=my_con.my_var_name
            if my_con.is_ub==True:
                self.D['delta_name_2_ub'][my_var_name]=my_con.rhs_amount_unsigned
            else:
                self.D['delta_name_2_lb'][my_var_name]=my_con.rhs_amount_unsigned


    

    def branch_on_g(self):
        """
        Branch on action 'act_u_v': return two new child nodes.
        Left: x_{uv} = 0
        Right: x_{uv} = 1 and x_{uw} = 0 for all w != v

        Parameters:
            action (str): The action string in the format "act_u_v"
        Returns:
            (BBNode, BBNode): Left and right child nodes
        """

        new_con_lb=aux_constr(self.g_star,False,self.lb_term_separ)
        new_con_ub=aux_constr(self.g_star,True,self.ub_term_separ)
        # Left child: add action to must_be_zero
        my_cosntrs_left=self.my_aux_constrs+[new_con_lb]#self.must_be_zero | {action}
        my_cosntrs_right=self.my_aux_constrs+[new_con_ub]#self.must_be_zero | {action}
        left = BBNode(self.my_BB_tree,self,my_cosntrs_left)


        right = BBNode(self.my_BB_tree,self,my_cosntrs_right)

        return left, right

    def eval_separ(self):
        print('my_lp_sol')
        print(self.primal_sol_act)
        input('---')
        amount_inside = {
            g: sum(self.primal_sol_act.get(act, 0.0) for act in self.my_BB_tree.Z_by_group[g])
            for g in self.my_BB_tree.G
        }
        print('amount_inside')
        print(amount_inside)
        print(np.unique(amount_inside.values()))
        print('np.unique(amount_inside.keys())')
        input('---')
        frac_amount = {
            g: min(
                amount_inside[g] - int(amount_inside[g]),
                np.ceil(amount_inside[g]) - amount_inside[g]
            )
            for g in self.my_BB_tree.G
        }
        self.g_star = max(self.my_BB_tree.G, key=lambda g: frac_amount[g]*self.my_BB_tree.pred_val_gain[g])
        self.lb_term_separ=np.ceil(amount_inside[self.g_star])
        self.ub_term_separ=np.floor(amount_inside[self.g_star])

        #print('frac_amount')
        #print(frac_amount)
        print('amount_inside[self.g_star]')
        print(amount_inside[self.g_star])
        print('self.g_star')
        print(self.g_star)
        print('self.lb_term_separ')
        print(self.lb_term_separ)
        print('self.ub_term_separ')
        print(self.ub_term_separ)
        input('---')

class BB_tree_solve:
    def  __init__(self,D,my_params,my_output_path):
        self.D=D
        self.use_uniform_split=True
        self.my_output_path=my_output_path
        self.my_params=my_params
        self.my_params['do_ilp']=False
        self.my_params['num_ng_use_separ']=10
        initial_max_iter=self.my_params['max_iterations_loop_compress_project']
        self.my_params['max_iterations_loop_compress_project']=100
        self.act_src_map = defaultdict(list)
        self.BIG_M=20000000
        self.pre_process_all_branching()

        for act in self.D['action2Cost']:#.act_2_cost:
            if act.startswith("act_"):
                _, u, v = act.split("_")
                self.act_src_map[u].append((v, act))
        self.root_node=BBNode(self,None,[])
        self.my_params['max_iterations_loop_compress_project']=initial_max_iter
        self.call_branch_bound()

        #self.my_params['do_ilp']=False

    def write_history(self):
        with open(self.my_output_path, 'w') as file:
            json.dump(self.data_hist, file)

    def call_branch_bound(self):
        """
        Branch-and-bound using priority queue over LP lower bounds.
        Explores the tree by branching on the most fractional variable.
        """
        # Min-heap priority queue: (LB, node count, BBNode)
        heap = []
        heapq.heappush(heap, (self.root_node.LB, 0, self.root_node))
        node_counter = 1

        best_primal = None
        best_obj = float('inf')
        self.final_node=[]
        self.lb_hist=[]
        self.data_hist=[]
        while True:
            cur_LB, _, node = heapq.heappop(heap)
            self.data_hist.append(node.data)
            self.lb_hist.append([cur_LB,node.par_lb,node.LB_init])
            self.write_history()
            #print('self.lb_hist')
            #print(self.lb_hist)
            #print('cur_LB')
            ##print('node.must_be_zero')
            #print(node.must_be_zero)
            #print('node.must_be_one')
            #print(node.must_be_one)
            print('lower hist')
            print(node.my_solver.history_dict['lblp_lower'])
            input('---')
            if node.parent!=None:
                print('lower Parent')
                print(node.parent.my_solver.history_dict['lblp_lower'])
                if (node.LB<-.001+node.parent.LB):
                    input('error here')
            #print('done at the root')
            #input('---')
            if not node.term_2_branch_on:
                # Node is LP integral
                obj_val = sum(
                    node.D['action2Cost'][act] * val
                    for act, val in node.primal_sol_act.items()
                    if act in node.D['action2Cost']
                )
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_primal = node.primal_sol_act.copy()
                self.final_node=node
                break 

            # Branch on the most fractional variable
            
            left_node, right_node = node.branch_on_g()

            heapq.heappush(heap, (left_node.LB, node_counter, left_node))
            node_counter += 1
            heapq.heappush(heap, (right_node.LB, node_counter, right_node))
            node_counter += 1

        # Store result
        self.best_primal = best_primal
        self.best_obj = best_obj
        print('self.lb_hist')
        print(self.lb_hist)
    
    def pre_process_all_branching(self):
        my_VRP=self.D['my_VRP']
        print(my_VRP)
        Nc=my_VRP.num_cust
        D=self.D
        [ng_neigh_by_cust,junk]=naive_get_LA_neigh(my_VRP,self.my_params['num_ng_use_separ'])
        G=set()
        for u in range(0,Nc):

            neighborhood = set(ng_neigh_by_cust[u]) | {u}
            for g in power_set(neighborhood):
                if g:  # skip empty set
                    G.add(frozenset(g))
        G.add(frozenset(np.arange(0,Nc)))
        self.G=G

        self.Z_by_group = defaultdict(set)  # g → set of act_u_v
        all_actions=self.D['action2Cost'].keys()
        for act in all_actions:
            _, u, v = act.split("_")
            for g in G:
                if u in g and v not in g:
                    self.Z_by_group[g].add(act)
        print(self.Z_by_group)
        print(self.Z_by_group)
        costs=self.D['action2Cost']
        self.pred_val_gain = {
            g: min(costs[act] for act in self.Z_by_group[g]) if self.Z_by_group[g] else float("inf")
            for g in G
        }
