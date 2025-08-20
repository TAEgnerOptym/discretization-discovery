import sys
sys.path.append("pre_process")
from convert_dict_keys_str_to_tuple import *
from  make_problem_instance import *
from clean_up_json_input_post_process import *
#from full_solver import full_solver
from clean_full_solver import full_solver
from grab_default_params import grab_default_params
from grab_params import grab_params
import json
import ast
import copy



def call_and_run_code(input_file_path,my_params_path,my_json_file_path,my_output_path):

    my_params=grab_params(my_params_path)

    print('Reading file and creating input')
    my_instance=make_problem_instance(input_file_path,my_params,my_json_file_path)
    print('DONE Reading file and creating input')
    # Open and load the JSON file

    print('Loading file file and adjusting for domian')

    D=None
    with open(my_json_file_path, 'r') as file:
        D = json.load(file)
    if D==None:
        input('error here')
    D=clean_up_json_input_post_process(D)
    print('DONE Loading file file and adjusting for domian')

    print('Calling the solver')

    D['my_VRP']=my_instance

    if 1<0:
        my_solver=full_solver(D,my_params,my_output_path)
    else:
        D1 = copy.deepcopy(D)

        my_params['do_ilp']=False
        my_solver=full_solver(D,my_params,my_output_path)
        #input('----')
        my_params['do_ilp']=True

        OUT_all_actions_inclumbent=my_solver.all_actions_inclumbent
        D1['initGraphNode2AggNode']=my_solver.graph_node_2_agg_node
        OUT_hist_terms_phase_one=my_solver.history_dict
        my_solver=full_solver(D1,my_params,my_output_path,all_actions_inclumbent=OUT_all_actions_inclumbent,hist_terms_phase_one=OUT_hist_terms_phase_one)