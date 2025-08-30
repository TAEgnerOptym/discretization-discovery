from call_and_run_code import call_and_run_code
import os

param_file_path=[]

all_files_param=[]
all_files=[]
all_files.append("data/jy_c201.txt")
all_files.append("data/jy_c202.txt")
all_files.append("data/jy_c203.txt")
all_files.append("data/jy_c204.txt")
all_files.append("data/jy_c205.txt")
all_files.append("data/jy_c206.txt")
all_files.append("data/jy_c207.txt")
all_files.append("data/jy_c208.txt")

all_files_param.append('params_WW/params_c200_100_UB_given/c201.json')
all_files_param.append('params_WW/params_c200_100_UB_given/c202.json')
all_files_param.append('params_WW/params_c200_100_UB_given/c203.json')
all_files_param.append('params_WW/params_c200_100_UB_given/c204.json')
all_files_param.append('params_WW/params_c200_100_UB_given/c205.json')
all_files_param.append('params_WW/params_c200_100_UB_given/c206.json')
all_files_param.append('params_WW/params_c200_100_UB_given/c207.json')
all_files_param.append('params_WW/params_c200_100_UB_given/c208.json')

all_out_files=[]
all_out_files.append("../all_willem_rez/WillemRez_c200_100_UB/jy_c201.txt")
all_out_files.append("../all_willem_rez/WillemRez_c200_100_UB/jy_c202.txt")
all_out_files.append("../all_willem_rez/WillemRez_c200_100_UB/jy_c203.txt")
all_out_files.append("../all_willem_rez/WillemRez_c200_100_UB/jy_c204.txt")
all_out_files.append("../all_willem_rez/WillemRez_c200_100_UB/jy_c205.txt")
all_out_files.append("../all_willem_rez/WillemRez_c200_100_UB/jy_c206.txt")
all_out_files.append("../all_willem_rez/WillemRez_c200_100_UB/jy_c207.txt")
all_out_files.append("../all_willem_rez/WillemRez_c200_100_UB/jy_c208.txt")
all_out_files.append("../all_willem_rez/WillemRez_c200_100_UB/jy_c209.txt")







in_fold="data/"

my_json_input_path="mid_jnk"
for k in range(0,9):

    
    input_file_path=all_files[k]
    param_file_path=all_files_param[k]
    output_file_path=all_out_files[k]
    print('input_file_path')
    print(input_file_path)
    if not os.path.exists(output_file_path):
    #if   not os.path.exists(output_file_path):
        print("input_file_path")
        print(input_file_path)
        print("param_file_path")
        print(param_file_path)
        print("_----")
        call_and_run_code(input_file_path, param_file_path, my_json_input_path, output_file_path)
    else:
        print(f"Output file {output_file_path} already exists. Skipping call.")