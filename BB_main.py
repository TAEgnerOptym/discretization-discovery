from BB_call_and_run_code import BB_call_and_run_code
import argparse

#default_input='data/jy_C104.txt'

default_input='jy_nyc_4.txt'

do_BIG_PROBLEM=True
if do_BIG_PROBLEM==True:
    #default_input='data/jy_RC203.txt'
    default_input='data/jy_C104.txt'
    #default_input='data/jy_RC204.txt'
    default_input='data/jy_RC107.txt'
#default_option_path='my_params_BBB.json'
#default_option_path='my_params_BBB_C104.json'
#default_option_path='my_params_BBB_R100.json'
default_option_path='my_params_BBB_RC100.json'

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
BB_call_and_run_code(input_file_path,param_file_path,my_json_input_path,output_file_path)