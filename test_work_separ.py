
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
from scipy.sparse import csr_matrix
from lower_bound_LP_milp import lower_bound_LP_milp
#from exper_lower_bound_LP_MILP import lower_bound_LP_milp
import pulp
from compressor import compressor
from class_new_valid import complete_separater_end_to_end 

#from experimental_compressor_additive import compressor
#from projector import projector
#from experimental_projector_simp import projector
#from experimental_projector_simp_eq import projector
from exper_proj_new import projector
#from  import projector
#from experimental_projector_simp_no_neg import projector

#from experimental_projector_simp_no_neg_w_removal import projector
from baseline_solver import baseline_solver
import json
from class_new_valid import complete_separater_end_to_end 
from New_valid_sep.check_valid_round_2 import check_valid_round_2
import pickle


loaded_object=[]

print('staritng load ')
#with open("../ALL_JSON_BIG/GOOD_my_object.pkl", "rb") as f:
my_file_name="GOOD_my_object.pkl"
my_file_name="R104_iter_1.pkl"
my_file_name="LAST_112_my_object.pkl"
my_file_name="Play_my_object.pkl"
with open(my_file_name, "rb") as f:
    loaded_object = pickle.load(f)
tmp=check_valid_round_2(loaded_object,do_custom_NG=True,num_LA_cutting_plane=7,max_SRI_Divisor=3,max_SRI_SET_SIZE=5)
print('done load ')
input('paused')
num_LA_cutting_plane=10
max_SRI_Divisor=3
max_SRI_SET_SIZE=5
use_custom_ng=True


#ILP_ONE_my_lower_bound_ILP=lower_bound_LP_milp(loaded_object,loaded_object.graph_node_2_agg_node,True,False)

my_adder=complete_separater_end_to_end(loaded_object,use_custom_ng,num_LA_cutting_plane,max_SRI_Divisor,max_SRI_SET_SIZE)
my_adder.update_given_solution(loaded_object.running_average_sol)
ILP_TWO_my_lower_bound_ILP=lower_bound_LP_milp(loaded_object,loaded_object.graph_node_2_agg_node,True,False)
#val_1=ILP_ONE_my_lower_bound_ILP.milp_solution_objective_value
val_2=ILP_TWO_my_lower_bound_ILP.milp_solution_objective_value
if abs(val_1-val_2)>0.01:
    print('val_2')
    print(val_2)
    print('val_1')
    print(val_1)
    input('big error')