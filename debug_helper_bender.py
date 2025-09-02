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

with open("PlayHere.pkl", "rb") as f:
    loaded_solver = pickle.load(f)

    num_calls_ineq=0
    debug_on=True
    print('making repo')
    x=loaded_solver.my_lower_bound_LP.lp_primal_solution
    loaded_solver.my_bender_repo=benders_repo_new(loaded_solver)
    [this_cut_value,did_gen_cut]=loaded_solver.my_bender_repo.generate_cuts(x)
