from collections import defaultdict
from full_solver import full_solver
import heapq
class BBNode:
    def __init__(self,my_BB_tree, must_be_zero, must_be_one, parent):
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
        self.apply_branching_penalties_fast()
        jy_opt=self.my_BB_tree.my_params
        if parent!=None:
            PARENTS_graph_node_2_agg_node=dict()
            for h in self.parent.graph_node_2_agg_node:
                PARENTS_graph_node_2_agg_node[h]=dict()
                for n in self.parent.graph_node_2_agg_node[h]:
                    PARENTS_graph_node_2_agg_node[h][n]=self.parent.graph_node_2_agg_node[h][n]
            self.D['graph_node_2_agg_node']=PARENTS_graph_node_2_agg_node
        self.my_solver=full_solver(self.D,jy_opt,'junk')
        self.LB=self.my_solver.history_dict['lblp_lower'][-1]
        self.primal_sol_act=dict()
        my_lp_sol=self.my_solver.my_lower_bound.primal_solution
        self.frac_var_2_value=dict()
        branch_term_cur=None
        biggest_prod=0
        for my_act in self.my_solver.my_lower_bound.all_actions:
            y=my_lp_sol[my_act]
            self.primal_sol_act[my_act]=y
            if y<.999 and y>0.001:
                self.frac_var_2_value[my_act]=my_lp_sol[my_act]
                if y*(1-y)>biggest_prod:
                    branch_term_cur=my_act
                    biggest_prod=y*(1-y)
        self.term_2_branch_on=branch_term_cur
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
            for v, other_act in act_src_map[u]:
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
        left = BBNode(
            must_be_zero=self.must_be_zero | {action},
            must_be_one=self.must_be_one,
            parent=self
        )


        right = BBNode(
            must_be_zero=self.must_be_zero,
            must_be_one=self.must_be_one | {action},
            parent=self
        )

        return left, right



class BB_tree_solve:
    def  __init__(D,my_params,my_output_path):
        self.D=D
        self.my_output_path=my_output_path
        self.my_params=my_params
        self.my_params['do_ilp']=False
        self.act_src_map = defaultdict(list)
        self.BIG_M=100000
        for act in self.D['action2Cost'].act_2_cost:
            if act.startswith("act_"):
                _, u, v = act.split("_")
                self.act_src_map[u].append((v, act))
        root_node=BBNode(self,set([]),set([]),None)
        self.call_branch_bound()

        #self.my_params['do_ilp']=False

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
        while True:
            cur_LB, _, node = heapq.heappop(heap)

            if not node.term_2_branch_on:
                # Node is LP integral
                obj_val = sum(
                    node.D['action2Cost'][act] * val
                    for act, val in node.primal_sol.items()
                    if act in node.D['action2Cost']
                )
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_primal = node.primal_sol.copy()
                self.final_node=node
                break 

            # Branch on the most fractional variable
            left_node, right_node = node.branch_on(node.term_2_branch_on)

            heapq.heappush(heap, (left_node.LB, node_counter, left_node))
            node_counter += 1
            heapq.heappush(heap, (right_node.LB, node_counter, right_node))
            node_counter += 1

        # Store result
        self.best_primal = best_primal
        self.best_obj = best_obj
        