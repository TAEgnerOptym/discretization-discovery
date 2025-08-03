from call_and_run_code import call_and_run_code

import argparse

#default_input='data/jy_C104.txt'


default_input='data/jy_C104.txt'
default_option_path='my_params_C.json'
#default_input='data/jy_RC208.txt'
#default_option_path='my_params_RC.json'
#default_input='data/jy_R203.txt'
#default_input='data/jy_RC103.txt'
#default_option_path='my_params_R200.json'

#default_option_path='my_params_debug_branch_RMP.json'
#default_option_path='my_params_debug_branch_RMP.json'
#for any given group of 7 customers.  The number of routes that use 
# 3,4 of those Plus 2* number that use 5,6,7 can not exceed 2
#default_input='data/jy_R104.txt'
#default_option_path='my_params.json'
#speed up the inner LPs.  
#speed up the projector LPs

default_my_json='../ALL_JSON_BIG/sample_json_input_description.json'
default_out_file_path='../ALL_JSON_BIG/sample_json_output.json'
parser = argparse.ArgumentParser(description="Description of your program")
parser.add_argument("input_file_path", type=str,nargs="?", help="input_file_path",default=default_input)
parser.add_argument("my_json_input_path", type=str,nargs="?", help="my_json_input_path",default=default_my_json)
parser.add_argument("output_file_path", type=str,nargs="?", help="output_file_path",default=default_out_file_path)
parser.add_argument("options_file_path", type=str,nargs="?", help="options_file_path",default=default_option_path)

args = parser.parse_args()
input_file_path=args.input_file_path
output_file_path=args.output_file_path
my_json_input_path=args.my_json_input_path
param_file_path=args.options_file_path
call_and_run_code(input_file_path,param_file_path,my_json_input_path,output_file_path)