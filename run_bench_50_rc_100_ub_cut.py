from call_and_run_code import call_and_run_code
import os

param_file_path=[]

all_files_param=[]
all_files=[]
all_files.append("data/jy_rc101.txt")
all_files.append("data/jy_rc102.txt")
all_files.append("data/jy_rc103.txt")
all_files.append("data/jy_rc104.txt")
all_files.append("data/jy_rc105.txt")
all_files.append("data/jy_rc106.txt")
all_files.append("data/jy_rc107.txt")
all_files.append("data/jy_rc108.txt")


all_files_param.append('params_WW/params_rc100_50_UB_given_cut/rc101.json')
all_files_param.append('params_WW/params_rc100_50_UB_given_cut/rc102.json')
all_files_param.append('params_WW/params_rc100_50_UB_given_cut/rc103.json')
all_files_param.append('params_WW/params_rc100_50_UB_given_cut/rc104.json')
all_files_param.append('params_WW/params_rc100_50_UB_given_cut/rc105.json')
all_files_param.append('params_WW/params_rc100_50_UB_given_cut/rc106.json')
all_files_param.append('params_WW/params_rc100_50_UB_given_cut/rc107.json')
all_files_param.append('params_WW/params_rc100_50_UB_given_cut/rc108.json')

all_out_files=[]
all_out_files.append("../all_willem_rez_given_cut/WillemRez_rc100_50_UB/jy_rc101.txt")
all_out_files.append("../all_willem_rez_given_cut/WillemRez_rc100_50_UB/jy_rc102.txt")
all_out_files.append("../all_willem_rez_given_cut/WillemRez_rc100_50_UB/jy_rc103.txt")
all_out_files.append("../all_willem_rez_given_cut/WillemRez_rc100_50_UB/jy_rc104.txt")
all_out_files.append("../all_willem_rez_given_cut/WillemRez_rc100_50_UB/jy_rc105.txt")
all_out_files.append("../all_willem_rez_given_cut/WillemRez_rc100_50_UB/jy_rc106.txt")
all_out_files.append("../all_willem_rez_given_cut/WillemRez_rc100_50_UB/jy_rc107.txt")
all_out_files.append("../all_willem_rez_given_cut/WillemRez_rc100_50_UB/jy_rc108.txt")






in_fold="data/"

my_json_input_path="mid_jnk"
for k in range(0,8):

    
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