import gurobipy as gp
from gurobipy import GRB
import time
from collections import defaultdict
import networkx as nx
import numpy as np
# Set desired solver options
import itertools
import gurobipy as gp
from solve_gurobi_lp import solve_gurobi_lp
import sys
from tqdm import tqdm
import json
from itertools import combinations
from New_valid_sep.graph_localized_g import graph_localized_g
from New_valid_sep.power_set import power_set
from New_valid_sep.NEW_order_object_new_sep_backwards import NEW_order_object_new_sep_backwards
class check_valid_round_2:

    def make_custom_NG(self, K):
        x = self.MF.my_lower_bound_LP.lp_primal_solution
        Nc = self.MF.my_VRP.num_cust

        # Step 1: Build a directed graph in NetworkX
        G = nx.Graph()
        for u in range(Nc):
            for v in range(Nc):
                if u == v:
                    continue
                var_name = f'act_{u}_{v}'
                if var_name in x and x[var_name]>0.0001:
                    
                    weight = 1/(.001+x[var_name])
                    G.add_edge(u, v, weight=weight)
        all_pairs_dist = dict(nx.all_pairs_dijkstra_path_length(G))

        # Step 3: For each node, find K nearest neighbors by shortest path distance
        nearest_neighbors = dict()
        for u in range(0,Nc):
            dist_u = all_pairs_dist.get(u, {})
            neighbors = [(v, d) for v, d in dist_u.items() if v != u]
            neighbors.sort(key=lambda item: item[1])
            tmp = [v for v, _ in neighbors[:K]]
            nearest_neighbors[u]=tmp

        self.u_2_NG=nearest_neighbors
        #print('nearest_neighbors')
        #print(nearest_neighbors)
        #input('---')
    def get_uni_groups(self):
        my_uni_groups=set()
        for u in self.u_2_NG:
            #print('self.u_2_NG[u]]')
            #print(self.u_2_NG[u])
            tmp=[u]+self.u_2_NG[u]
            new_set=frozenset(tmp)
            #print('new_set')
            #print(new_set)
            my_uni_groups.add(new_set)
        self.my_uni_groups=my_uni_groups
    def __init__(self,MF,do_custom_NG=False,num_LA_cutting_plane=8,max_SRI_Divisor=3,max_SRI_SET_SIZE=5):
        self.MF=MF
        self.OPT=dict()
        self.OPT['num_LA_cutting_plane']=num_LA_cutting_plane
        self.OPT['max_SRI_Divisor']=max_SRI_Divisor
        self.OPT['max_SRI_SET_SIZE']=max_SRI_SET_SIZE
        self.OPT['allow_slack_on_nodes']=True
        self.OPT['do_custom_NG']=do_custom_NG
        self.epsilon_slack_valid=.00001      
        
        
        self.make_custom_NG(self.OPT['num_LA_cutting_plane'])
        self.get_uni_groups()
        self.tot_viol=0
        for my_subset_cust in self.my_uni_groups:
            my_graph=graph_localized_g(self.MF,my_subset_cust,self.OPT)
            #my_LP_structure=m(my_graph,self.MF,g,self.OPT)
            #self.tot_viol=tot_viol+self.tot_viol
            #input('done the whole grap h stuff')

