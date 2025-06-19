
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
import random
import time
import bisect
from solve_gurobi_lp import solve_gurobi_lp
class LAZY_projector:

    def __init__(self,my_full_solver,h):
        self.time_dict_proj=dict()
        t1=time.time()
        self.h=h
        self.MF=my_full_solver
        self.my_lp=self.MF.my_lower_bound_LP
        self.agg_node_2_node=self.my_lp.agg_node_2_nodes[h]
        self.graph_node_2_agg_node=self.MF.graph_node_2_agg_node[h]

        self.lp_primal_solution=self.my_lp.lp_primal_solution
        self.lp_dual_solution=self.my_lp.lp_dual_solution
        self.this_fg_sink=self.my_lp.graph_node_2_agg_node[h][self.MF.h_2_sink_id[h]]
        self.this_fg_source=self.my_lp.graph_node_2_agg_node[h][self.MF.h_2_source_id[h]]
        self.compact_sink=self.MF.h_2_sink_id[self.h]
        self.compact_source=self.MF.h_2_source_id[self.h]
        self.fg_nodes_non_source_sink=set(self.agg_node_2_node)-set([self.this_fg_sink,self.this_fg_source])
        self.null_action=self.MF.null_action
        self.non_source_sink_agg_nodes=set(self.agg_node_2_node.keys())-set([self.this_fg_sink,self.this_fg_source])

        self.ij_2_fg=self.my_lp.h_ij_2_fg[h]
        self.fg_2_ij=self.my_lp.h_fg_2_ij[h]
        self.hij_2_P_orig=self.my_lp.hij_2_P[h]
        
        self.non_source_sink=set(self.graph_node_2_agg_node)-set([self.compact_sink ,self.compact_source])
        self.time_dict_proj['prior']=time.time()-t1
        t1=time.time()
        
        self.make_digraph_object()
        
        self.make_new_splits()
        self.lp_time=0
        self.lp_objective=np.inf
    
    import networkx as nx

    def make_digraph_object(self):
        h = self.h
        null_action = self.null_action

        # Step 1: Collect dual values for actions
        action_2_dual_value = {null_action: 0}

        for p in self.MF.all_non_null_action:
            dual_name = f"action_match_h={h}_p={p}"
            action_2_dual_value[p] = self.lp_dual_solution[dual_name]
        self.NZ_action_2_dual_value=dict()
        for p in action_2_dual_value:
            if abs(action_2_dual_value[p])>0.00001:
                self.NZ_action_2_dual_value[p]=action_2_dual_value[p]
        #print('action_2_dual_value[p]')
        #print(self.NZ_action_2_dual_value)
        #input('--')
        # Step 2: Build node mapping
        node_set = {ij[0] for ij in self.hij_2_P_orig} | {ij[1] for ij in self.hij_2_P_orig}
        node_to_id = {node: idx for idx, node in enumerate(sorted(node_set))}
        id_to_node = {idx: node for node, idx in node_to_id.items()}
        #print('node_set')
        #print(node_set)
        #print('node_to_id')
        #print(node_to_id)
        #print('id_to_node[776]')
        #print(id_to_node[776])
        #input('--')
        self.node_to_id = node_to_id
        self.id_to_node = id_to_node

        # Step 3: Build weighted edge list with numeric node IDs
        nx_input = []
        for ij, P_list in self.hij_2_P_orig.items():
            p = P_list[0]
            i, j = ij
            w = -action_2_dual_value[p]+0.00001
            my_tup = (node_to_id[i], node_to_id[j], w)
            nx_input.append(my_tup)

        # Step 4: Build graph and compute shortest distances
        G = nx.DiGraph()
        G.add_weighted_edges_from(nx_input)

        my_source = node_to_id[self.compact_source]
        #self.detect_negative_cycle(G)
        self.i_2_dual = nx.single_source_bellman_ford_path_length(G, source=my_source, weight='weight')
        
        
        self.i_2_dual = {
            self.id_to_node[i]: dist for i, dist in self.i_2_dual.items()
        }
        #print('self.i_2_dual')
        #print(self.i_2_dual)
        #input('--')
        #input('ALL DONE GOOD AND CLEAN')

    def detect_negative_cycle(self,G, weight='weight'):
        # Step 1: Initialize distances and predecessors
        dist = {}
        pred = {}
        for node in G.nodes():
            dist[node] = float('inf')
            pred[node] = None

        # Pick an arbitrary starting node
        source = next(iter(G.nodes))
        dist[source] = 0

        # Step 2: Relax edges |V| - 1 times
        for _ in range(len(G.nodes) - 1):
            for u, v, w in G.edges(data=weight):
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    pred[v] = u

        # Step 3: Check for a negative cycle
        for u, v, w in G.edges(data=weight):
            if dist[u] + w < dist[v]:
                # Found a cycle, trace it back
                print("Negative cycle detected.")
                cycle = [v]
                while True:
                    v = pred[v]
                    if v in cycle:
                        cycle = cycle[cycle.index(v):]  # close the loop
                        break
                    cycle.append(v)
                cycle.reverse()

                # Print cycle with weights
                print("Cycle:")
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i+1) % len(cycle)]
                    w = G[u][v][weight]
                    #print(f"{u} -> {v} (weight {w})")
                    print(f"{self.id_to_node[u]} -> {self.id_to_node[v]} (weight {w})")
                input('LOOK')
                return cycle

        print("No negative cycle found.")
        return None

    def make_new_splits(self):

        self.NEW_node_2_agg_node=self.graph_node_2_agg_node.copy()
        start_value=0
        extra_string='rand_'+str(random.randint(0,100000000))+'_'
        i_2_dual=self.i_2_dual
        
        count_change=0
        

        self.f_2_mean_val=dict()
        self.f_2_min_val=dict()
        self.f_2_max_val=dict()
        self.do_split_f=[]
        count_find=dict()
        for i in self.graph_node_2_agg_node:
            count_find[i]=0
        for f in self.non_source_sink_agg_nodes:
            my_sum=0
            my_min=np.inf
            my_max=-np.inf
            all_terms=[]
            all_names=[]
            for i in self.agg_node_2_node[f]:
                count_find[i]=count_find[i]+1
                this_term=i_2_dual[i]
                my_sum=my_sum+this_term#i_2_dual[i]
                my_max=max([my_max,this_term])
                my_min=min([my_min,this_term])
                all_names.append(i)
                all_terms.append(this_term)

            self.f_2_mean_val[f]=my_sum/len(self.agg_node_2_node[f])
            self.f_2_min_val[f]=my_min#my_sum/len(self.agg_node_2_node[f])
            self.f_2_max_val[f]=my_max#my_sum/len(self.agg_node_2_node[f])
            if self.f_2_max_val[f]-self.f_2_min_val[f]>self.MF.jy_opt['threshold_split']:#.0001:
                self.do_split_f.append(f)
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
            
        #print('self.do_split_f')
        #print(self.do_split_f)
        #print('self.h')
        #print(self.h)
        #input('---')

        start_value=0
        num_thesh_use=self.MF.jy_opt['num_thresh_split_projector']
        for f in self.do_split_f:
            start_value=start_value+num_thesh_use
            count_pos=0
            extra_str_f=str(random.randint(0,100000000))
            tmp_dict=dict()
            for i in self.agg_node_2_node[f]:
                tmp_dict[i]=i_2_dual[i]
            [chosen, new_dict]=self.quantize_dict_to_index(tmp_dict,num_thesh_use)
            for i in self.agg_node_2_node[f]:
                self.NEW_node_2_agg_node[i]=extra_string+'_'+extra_str_f+'_'+str(new_dict[i])
                count_change=count_change+1
                count_pos=count_pos+1


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
            #print('--')
            #print(levels)
            
            #input('levels no drop')

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