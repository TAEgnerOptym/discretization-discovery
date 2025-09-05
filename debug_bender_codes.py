import functools

import random
import re
from collections import defaultdict
from solve_gurobi_lp import solve_gurobi_lp_bounds
import numpy as np
import sys
from itertools import chain, combinations
import pickle
sys.path.append("pre_process")
import math
from naive_pre import *

from typing import Dict, Hashable, Tuple
from EXPER_benders_repo_new import benders_repo_new

#MIN_LP_OBJECTIVE_CUT=0.1
#COVER_EPSILON=0.0001
#OFFSET_COST_CUT=0.0000001
#EPSILON_STANDARD=0.0001
#EPSILON_RHS_SUB=0.000001
#EPSILON_EDGE=0.0001
#EPSILON_MULT_PARETO_OBJ=0.01#-0.0000001
#USE_RAND=1
#AX_SIZE_CHOOSE_K=230
VAL_STOP_ADDING_CUTS=100000000
MY_SIZES_USE=[9]

with open("PlayHere.pkl", "rb") as f:
    loaded_solver = pickle.load(f)
    TOT_gen_cut=0
    tot_cut_value=0
    tot_time_opt=0
    max_time_opt=0
    
    for my_bend_prob in loaded_solver.my_list_benders_cut_generator:#random.sample(loaded_solver.my_list_benders_cut_generator,len(loaded_solver.my_list_benders_cut_generator)):
#           print('generating cut ')
#           print('my_bend_prob.sub_prob_name')
#            print(my_bend_prob.sub_prob_name)
#            print('----')

        [this_cut_value,did_gen_cut,this_time_opt]=my_bend_prob.generate_benders_cut(loaded_solver.x_solution)
        if this_cut_value>.01:
            TOT_gen_cut=TOT_gen_cut+1
            tot_cut_value=tot_cut_value+this_cut_value
        tot_time_opt=tot_time_opt+this_time_opt
        max_time_opt=max([max_time_opt,this_time_opt])
        print('this_cut_value:  '+str(this_cut_value))
        print('tot_cut_value,TOT_gen_cut]:  '+str([tot_cut_value,TOT_gen_cut]))
        print('tot_time_opt.  '+str([this_time_opt,tot_time_opt]))
        if VAL_STOP_ADDING_CUTS<tot_cut_value:
            break
    print('[tot_cut_value,TOT_gen_cut]')
    print([tot_cut_value,TOT_gen_cut])
