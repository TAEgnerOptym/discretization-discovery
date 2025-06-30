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
    def __init__(self,my_BB_tree, must_be_zero, must_be_one, parent,my_aux_constrs):
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
        self.must_be_zero = must_be_zero
        self.must_be_one = must_be_one
        self.parent = parent
        self.my_aux_constrs=my_aux_constrs
        self.add_aux_constrs()
        self.add_aux_bounds()
        self.actions_ignore=None#self.self.all_actions_not_source_sink_connected
        self.pre_process_all_branching()

        self.apply_branching_penalties_fast()
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
        self.select_branch_customer()
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
        self.data['must_be_zero']=list(self.must_be_zero)
        self.data['must_be_one']=list(self.must_be_one)
        self.data['lb_hist']=list(self.my_solver.history_dict['lblp_lower'])
    
    def add_aux_constrs(self):

        self.D['exogName2Rhs']=self.my_BB_tree.D['exogName2Rhs'].copy()
        self.D['action_con_2_contrib']=self.my_BB_tree.D['action_con_2_contrib'].copy()
        self.D['all_delta']=self.my_BB_tree.D['all_delta'].copy()
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

            self.D['delta_con_2_contrib'][tuple([my_var_name,my_con_name_pos])]=-1
            self.D['delta_con_2_contrib'][tuple([my_var_slack_pos,my_con_name_pos])]=-1
            self.D['delta_con_2_contrib'][tuple([my_var_slack_neg,my_con_name_pos])]=1

            self.D['delta_con_2_contrib'][tuple([my_var_name,my_con_name_neg])]=1
            self.D['delta_con_2_contrib'][tuple([my_var_slack_pos,my_con_name_neg])]=1
            self.D['delta_con_2_contrib'][tuple([my_var_slack_neg,my_con_name_neg])]=-1

            self.D['all_delta'].append(my_var_name)
            self.D['all_delta'].append(my_var_slack_pos)
            self.D['all_delta'].append(my_var_slack_neg)

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
            for act in self.my_BB_tree.D['Z_by_group'][g]:
                self.D['action_con_2_contrib'][tuple([act,my_con_name_neg])]=-1
                self.D['action_con_2_contrib'][tuple([act,my_con_name_pos])]=1

    def add_aux_bounds(self):

        
        for my_con in self.my_aux_constrs:
            my_var_name=my_con.my_var_name
            if my_con.is_ub==True:
                self.D['delta_name_2_ub'][my_var_name]=my_con.rhs_amount_unsigned
            else:
                self.D['delta_name_2_lb'][my_var_name]=my_con.rhs_amount_unsigned

    
    def select_branch_customer(self):
        """
        Select the customer u ∈ N that minimizes max_{v} x_{uv}.
        Used to guide branching on customer successors.

        Returns:
            u_best (str): The selected customer ID as a string.
        """
        Nc=self.D['my_VRP'].num_cust
        customer_ids = [str(u) for u in range(0, Nc )]  # Excludes depots (0, Nc+1)

        best_u = None
        best_max = float('inf')

        for u in customer_ids:
            max_val = 0.0
            has_valid = False
            for v, act in self.act_src_map[u]:
                max_val = max(max_val, self.my_solver.my_lower_bound_LP.lp_primal_solution[act])
                has_valid = True
            if has_valid and max_val < best_max:
                best_max = max_val
                best_u = u
        self.cust_branch_on=best_u


    def apply_branching_penalties_fast(self):
        
        """
        Fast version of penalty application using source-action indexing.

        Parameters:
            act_2_cost (dict[str, float]): Action cost dictionary.
            must_be_zero (set[str]): Actions forced to zero (penalized).
            must_be_one (set[str]): Actions forced to one (others from same source penalized).
            M (float): Large penalty cost.
        """
        # Index actions by source node: act_src_map[u] = list of act_u_w
        

        # Apply penalties for must_be_zero directly
        must_be_zero=self.must_be_zero
        must_be_one=self.must_be_one
        M=self.M
        act_2_cost=dict()
        for p in self.D['action2Cost']:
            act_2_cost[p]=self.D['action2Cost'][p]
        for act in must_be_zero:
            act_2_cost[act] = M

        # Apply implied-zero penalties from must_be_one constraints
        for act in must_be_one:
            _, u, v_fixed = act.split("_")
            for v, other_act in self.act_src_map[u]:
                if v != v_fixed and other_act in act_2_cost:
                    act_2_cost[other_act] = M
        self.D['action2Cost']=act_2_cost

    def branch_on(self, action):
        """
        Branch on action 'act_u_v': return two new child nodes.
        Left: x_{uv} = 0
        Right: x_{uv} = 1 and x_{uw} = 0 for all w != v

        Parameters:
            action (str): The action string in the format "act_u_v"
        Returns:
            (BBNode, BBNode): Left and right child nodes
        """
        if not action.startswith("act_") or len(action.split("_")) != 3:
            raise ValueError(f"Invalid action format: {action}. Expected 'act_u_v'.")

        _, u, v = action.split("_")

        # Left child: add action to must_be_zero
        add_left_must_zeo=self.must_be_zero | {action}
        add_right_must_one=self.must_be_one | {action}
        left = BBNode(self.my_BB_tree,add_left_must_zeo,self.must_be_one,self)


        right = BBNode(self.my_BB_tree,self.must_be_zero,add_right_must_one,self)

        return left, right

    def eval_separ(self):
        amount_inside = {
            g: sum(self.my_lp_sol.get(act, 0.0) for act in self.my_BB_tree.Z_by_group[g])
            for g in self.G
        }
        frac_amount = {
            g: min(
                amount_inside[g] - int(amount_inside[g]),
                math.ceil(amount_inside[g]) - amount_inside[g]
            )
            for g in G
        }
        g_star = max(G, key=lambda g: frac_amount[g])

    def branch_on_customer(self):
        """
        Branch on successors of customer u.
        Returns two BBNode objects corresponding to:
        - left: x_{uv} = 0 for v in group A
        - right: x_{uv} = 0 for v in group B

        Greedy balanced partitioning based on LP solution values.
        """
        # Collect successors of u from act_src_map
        u=self.cust_branch_on
        primal_sol=self.my_lp_sol
        candidate_vs = [v for v, act in self.act_src_map[u] if act in primal_sol]
        
        # Compute weights with small noise for tie-breaking
        weight = {
            v: primal_sol[f"act_{u}_{v}"] + 0.00001 * random.random()
            for v in candidate_vs
        }

        # Sort successors in decreasing order of weight
        sorted_vs = sorted(weight, key=weight.get, reverse=True)

        group_A, group_B = [], []
        sum_A, sum_B = 0.0, 0.0

        for v in sorted_vs:
            if sum_A <= sum_B:
                group_A.append(v)
                sum_A += weight[v]
            else:
                group_B.append(v)
                sum_B += weight[v]
        #print('group_A')
        #print(group_A)
        #print('group_B')
        #print(group_B)
        #print('sum_B,sum_A')
        #print([sum_B,sum_A])
        #print('u')
        #print(u)
        #input('---')
        # Force actions in group_A to zero in left branch
        left_zero_set = self.must_be_zero | {f"act_{u}_{v}" for v in group_A}
        right_zero_set = self.must_be_zero | {f"act_{u}_{v}" for v in group_B}

        # Create child nodes
        left = BBNode(self.my_BB_tree, left_zero_set, self.must_be_one, parent=self)
        right = BBNode(self.my_BB_tree, right_zero_set, self.must_be_one, parent=self)

        return left, right


class BB_tree_solve:
    def  __init__(self,D,my_params,my_output_path):
        self.D=D
        self.use_uniform_split=True
        self.my_output_path=my_output_path
        self.my_params=my_params
        self.my_params['do_ilp']=False
        initial_max_iter=self.my_params['max_iterations_loop_compress_project']
        self.my_params['max_iterations_loop_compress_project']=100
        self.act_src_map = defaultdict(list)
        self.BIG_M=20000000
        for act in self.D['action2Cost']:#.act_2_cost:
            if act.startswith("act_"):
                _, u, v = act.split("_")
                self.act_src_map[u].append((v, act))
        self.root_node=BBNode(self,set([]),set([]),None,[])
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
            print('self.lb_hist')
            print(self.lb_hist)
            print('cur_LB')
            print(cur_LB)
            print('node.must_be_zero')
            print(node.must_be_zero)
            print('node.must_be_one')
            print(node.must_be_one)
            print('lower hist')
            print(node.my_solver.history_dict['lblp_lower'])
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
            left_node=[]
            right_node=[]
            if self.use_uniform_split==False:
                #input('i do nto think I want to be here')
                left_node, right_node = node.branch_on(node.term_2_branch_on)
            else:
                left_node, right_node = node.branch_on_customer()

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

        Nc=self.D['my_VRP'].num_cust
        
        [ng_neigh_by_cust,junk]=naive_get_LA_neigh(self['D'],(self['D']['my_params']['num_NG']))
        G=set()
        for u in range(0,Nc):

            neighborhood = ng_neigh_by_cust[u] | {u}
            for g in power_set(neighborhood):
                if g:  # skip empty set
                    G.add(frozenset(g))
        G.add(frozenset(np.arange(0,Nc)))
        self.G=G

        self.Z_by_group = defaultdict(set)  # g → set of act_u_v
        all_actions=self.D['action_2_cost'].keys()
        for act in all_actions:
            _, u, v = act.split("_")
            for g in G:
                if u in g and v not in g:
                    self.Z_by_group[g].add(act)

        costs=self.D['action_2_cost']
        self.pred_val_gain = {
            g: min(costs[act] for act in self.Z_by_group[g]) if self.Z_by_group[g] else float("inf")
            for g in G
        }
