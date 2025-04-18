import sys
sys.path.append("pre_process")
from convert_dict_keys_str_to_tuple import *
from  make_problem_instance import *
from clean_up_json_input_post_process import *
from full_solver import full_solver
from grab_default_params import grab_default_params
import json
import ast

import ast

input_file_path='data/jy_c103.txt'
my_params=grab_default_params()
#my push
my_json_file_path = "jy_data_end_end.json"
my_output_path="jy_out_history.json"
print('Reading file and creating input')
make_problem_instance(input_file_path,my_params,my_json_file_path)
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

my_solver=full_solver(D,my_params,my_output_path)