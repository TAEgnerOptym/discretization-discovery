from call_and_run_code import call_and_run_code
import os

param_file_path=[]

all_files_param=[]
all_files=[]
all_files.append("data/jy_c101.txt")
all_files.append("data/jy_c102.txt")
all_files.append("data/jy_c103.txt")
all_files.append("data/jy_c104.txt")
all_files.append("data/jy_c105.txt")
all_files.append("data/jy_c106.txt")
all_files.append("data/jy_c107.txt")
all_files.append("data/jy_c108.txt")
all_files.append("data/jy_c109.txt")

all_files_param.append('params_WW/params_c100_50_UB_given_cut/c101.json')
all_files_param.append('params_WW/params_c100_50_UB_given_cut/c102.json')
all_files_param.append('params_WW/params_c100_50_UB_given_cut/c103.json')
all_files_param.append('params_WW/params_c100_50_UB_given_cut/c104.json')
all_files_param.append('params_WW/params_c100_50_UB_given_cut/c105.json')
all_files_param.append('params_WW/params_c100_50_UB_given_cut/c106.json')
all_files_param.append('params_WW/params_c100_50_UB_given_cut/c107.json')
all_files_param.append('params_WW/params_c100_50_UB_given_cut/c108.json')
all_files_param.append('params_WW/params_c100_50_UB_given_cut/c109.json')

all_out_files=[]
all_out_files.append("../all_willem_rez_given_cut/WillemRez_c100_50_UB/jy_c101.txt")
all_out_files.append("../all_willem_rez_given_cut/WillemRez_c100_50_UB/jy_c102.txt")
all_out_files.append("../all_willem_rez_given_cut/WillemRez_c100_50_UB/jy_c103.txt")
all_out_files.append("../all_willem_rez_given_cut/WillemRez_c100_50_UB/jy_c104.txt")
all_out_files.append("../all_willem_rez_given_cut/WillemRez_c100_50_UB/jy_c105.txt")
all_out_files.append("../all_willem_rez_given_cut/WillemRez_c100_50_UB/jy_c106.txt")
all_out_files.append("../all_willem_rez_given_cut/WillemRez_c100_50_UB/jy_c107.txt")
all_out_files.append("../all_willem_rez_given_cut/WillemRez_c100_50_UB/jy_c108.txt")
all_out_files.append("../all_willem_rez_given_cut/WillemRez_c100_50_UB/jy_c109.txt")







in_fold="data/"

my_json_input_path="mid_jnk"
for k in range(0,9):

    
    input_file_path=all_files[k]
    param_file_path=all_files_param[k]
    output_file_path=all_out_files[k]
    print('input_file_path')
    print(input_file_path)
    if  not os.path.exists(output_file_path):
    #if   not os.path.exists(output_file_path):
        print("input_file_path")
        print(input_file_path)
        print("param_file_path")
        print(param_file_path)
        print("_----")
        call_and_run_code(input_file_path, param_file_path, my_json_input_path, output_file_path)
    else:
        print(f"Output file {output_file_path} already exists. Skipping call.")