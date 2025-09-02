
import pickle
from collections import defaultdict
from typing import Dict, DefaultDict, Set, List
import numpy as np
import pulp as pl
from pulp import LpProblem, LpVariable, LpMaximize, PULP_CBC_CMD
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xpress as xp
import networkx as nx
import time
import heapq
from scipy.sparse import csr_matrix
from lower_bound_LP_milp import lower_bound_LP_milp
from pre_process.naive_pre import *

#from exper_lower_bound_LP_MILP import lower_bound_LP_milp
import pulp
from compressor import compressor
from class_new_valid import complete_separater_end_to_end 
from power_set import power_set
from baseline_solver import baseline_solver
import json
from New_valid_sep.check_valid_round_2 import check_valid_round_2
from projector_on_lb import projector_on_lb
from benders_repo_new import benders_repo_new

with open("NEWINTERPlayHere.pkl", "rb") as f:
    loaded_solver = pickle.load(f)
    #[primal_solution,dual_solution,lp_objective]=loaded_solver.call_lp(loaded_solver.OPT_X_input)
    
    for my_var in loaded_solver.dict_var_name_2_obj:
        loaded_solver.dict_var_name_2_LB[my_var]=0
        loaded_solver.dict_var_name_2_UB[my_var]=0
    for ng_edge in loaded_solver.my_sub_prob.my_ng_graph.DEBUG_my_candid_edges:
        var_name='ng_EDGE_'+str(ng_edge)
        if var_name not in loaded_solver.dict_var_name_2_obj:
            print('----')
            print('----')
            print('----')
            print('----')
            print('----')
            if ng_edge not in loaded_solver.my_sub_prob.my_ng_graph.ng_edges:
                print('hihi')
                print('not here')
                input('---')
            else:
                print('IS here')
                print('ng_edge')
                print(ng_edge)
                print('--')
                for q in loaded_solver.my_sub_prob.my_ng_graph.DEBUG_my_candid_edges:
                    if q==ng_edge:
                        print('q is ')
                        print(q)
                        print('**')
                #print('loaded_solver.my_sub_prob.sub_prob_name')
                #print(loaded_solver.my_sub_prob.sub_prob_name)
                #print('loaded_solver.my_sub_prob.DEBUG_t')
                #print(loaded_solver.my_sub_prob.DEBUG_t)
                #print('loaded_solver.my_sub_prob.DEBUG_edge_name')
                #print(loaded_solver.my_sub_prob.DEBUG_edge_name)
                #print('loaded_solver.my_sub_prob.var_2_cost[var_name]')
                #print(loaded_solver.my_sub_prob.var_2_cost[var_name])
            #print('var_name in loaded_solver.my_sub_prob.sub_prob_y_obj')
            #print(var_name in loaded_solver.my_sub_prob.sub_prob_y_obj)
            print('var_name')
            print(var_name)
            
            input('error')
        else:
            print('ok')
            print('var_name')
            print(var_name)
            edge_name='ng_EDGE_'+str(ng_edge)
            print('loaded_solver.my_sub_prob.var_2_cost[var_name]')
            print(loaded_solver.my_sub_prob.var_2_cost[var_name])
            #input('--')
        loaded_solver.dict_var_name_2_UB[var_name]=1
    loaded_solver.generate_benders_cut(loaded_solver.in_x,loaded_solver.OPT_X_input)