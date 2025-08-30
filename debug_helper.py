import xpress as xp
import pickle
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
import re
from scipy.sparse import csr_matrix
import pulp
import sys
import itertools
import heapq

from pre_process.naive_pre import *
sys.path.append("exper_ideas")
from jy_active_set_lp import jy_active_set_lp
from jy_active_set_lp import jy_active_set_lp_primal_dual
from warm_start_lp import warm_start_lp
from warm_start_lp import forbidden_variables_loop
from warm_start_lp import forbidden_variables_loop_dual
from warm_start_lp import warm_start_lp_using_class
from warm_start_lp import warm_start_lp_using_class_gurobi

from solve_gurobi_lp import solve_gurobi_lp
from solve_gurobi_lp import solve_gurobi_lp_bounds
from solve_gurobi_lp import solve_gurobi_milp
from solve_gurobi_lp import solve_gurobi_milp_bounds

loaded_solver=[]
with open("solver_checkpoint_BEF_.pkl", "rb") as f:
    loaded_solver = pickle.load(f)
    #for i in range(0,50):
    #    q=loaded_solver.full_prob.ng_neigh_by_cust_power[i]
    #    print(i)
    #    print(q)
    #    input('---')
    loaded_solver.full_prob.jy_opt['maxVarsAdd_in_ITER']=10
    #loaded_solver.full_prob.G= {g for g in loaded_solver.full_prob.G_power if len(g) >= 10}
    loaded_solver.iterative_ilp_la_DIVE()
    loaded_solver
    #loaded_solver.iterative_ilp_la()
    loaded_solver.filter_constraints()
    loaded_solver.call_gurobi_milp_solver()

