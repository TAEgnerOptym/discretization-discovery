from call_and_run_code import call_and_run_code

import argparse

parser = argparse.ArgumentParser(description="Description of your program")
parser.add_argument("input_file_path", type=str, help="input_file_path")
parser.add_argument("my_json_input_path", type=str, help="my_json_input_path")
parser.add_argument("output_file_path", type=str, help="output_file_path")
parser.add_argument("options_file_path", type=str, help="options_file_path")

args = parser.parse_args()
input_file_path=args.input_file_path
output_file_path=args.output_file_path
my_json_input_path=args.my_json_input_path
param_file_path=args.options_file_path
call_and_run_code(input_file_path,param_file_path,my_json_input_path,output_file_path)