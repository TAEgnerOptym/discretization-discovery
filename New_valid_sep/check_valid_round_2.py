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
from New_valid_sep.my_LP_sep_structure import my_LP_sep_structure
class check_valid_round_2:

    def make_custom_NG(self, K):
        x = self.MF.my_lower_bound_LP.lp_primal_solution
        Nc = self.MF.my_VRP.num_cust

        # Step 1: Build a directed graph in NetworkX
        tot_weight=0
        G = nx.Graph()
        #print('K')
        #print(K)
        #input('--')
        for u in range(Nc):
            for v in range(u+1,Nc):

                var_name_1 = f'act_{u}_{v}'
                var_name_2 = f'act_{v}_{u}'
                amount_1=0
                amount_2=0
                if var_name_1 in x and x[var_name_1]>0.0001:
                    amount_1= x[var_name_1]
                if var_name_2 in x and x[var_name_2]>0.0001:
                    amount_2= x[var_name_2]
                amount_use=max([amount_1+amount_2])
                if amount_use>0.0001:
                    
                    weight = 1/(.001+amount_use)
                    G.add_edge(u, v, weight=weight)
                    tot_weight=tot_weight+weight
                    #print('uv')
                    #print([u,v])
                    #print('weight')
                    #print(weight)

        #print(tot_weight)
        #input('--')
        all_pairs_dist = dict(nx.all_pairs_dijkstra_path_length(G))
        #print('all_pairs_dist')
        #print(all_pairs_dist)
        dist_u = all_pairs_dist.get(4, {})
        #print('dist_u')
        #print(dist_u)
        #input('---')
        # Step 3: For each node, find K nearest neighbors by shortest path distance
        nearest_neighbors = dict()
        for u in range(0,Nc):
            dist_u = all_pairs_dist.get(u, {})
            neighbors = [(v, d) for v, d in dist_u.items() if v != u]
            neighbors.sort(key=lambda item: item[1])
            tmp = [v for v, _ in neighbors[:K]]
            nearest_neighbors[u]=tmp
            #print('u')
            #print(u)
            #print('dist_u')
            #print(dist_u)

        self.u_2_NG=nearest_neighbors
        #print('dist_u')
        #print(dist_u)
        #input('--')
        #print(nearest_neighbors)
        #input('nearest_neighbors')
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
    def __init__(self,MF,do_custom_NG=False,num_LA_cutting_plane=10,max_SRI_Divisor=3,max_SRI_SET_SIZE=5):
        self.MF=MF
        self.OPT=dict()
        self.OPT['num_LA_cutting_plane']=num_LA_cutting_plane
        self.OPT['max_SRI_Divisor']=max_SRI_Divisor
        self.OPT['max_SRI_SET_SIZE']=max_SRI_SET_SIZE
        self.OPT['allow_slack_on_nodes']=True
        self.OPT['do_custom_NG']=do_custom_NG
        self.OPT['my_magnanti']=0.00001
        #'aggAllOutgoingIncomingEdges', 'aggAllOutgoingByNode', 'AggFineDisc'
        #self.OPT['match_external_option']='aggAllOutgoingIncomingEdges'
        self.epsilon_slack_valid=.00001      
        
        
        self.make_custom_NG(self.OPT['num_LA_cutting_plane'])
        self.get_uni_groups()
        self.tot_viol=0
        print('self.my_uni_groups')
        print(self.my_uni_groups)
        #input('heyo')
        #if self.MF.my_VRP.num_cust==6:
        #    self.my_uni_groups=set([frozenset([0,1,2,3,4,5])])
        for my_subset_cust in self.my_uni_groups:
            my_graph=graph_localized_g(self.MF,my_subset_cust,self.OPT)

            my_LP_structure=my_LP_sep_structure(my_graph,self.MF,self.OPT)
            self.tot_viol+=my_LP_structure.out_solution['objective']
            #input('all done')
        print('self.tot_viol')
        print(self.tot_viol)
        print('self.tot_viol')
        print('******')
        print('******')
        print('******')
        print('******')
        print('******')
        