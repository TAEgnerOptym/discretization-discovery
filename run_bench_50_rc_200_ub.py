from call_and_run_code import call_and_run_code
import os

param_file_path=[]

all_files_param=[]
all_files=[]
all_files.append("data/jy_rc201.txt")
all_files.append("data/jy_rc202.txt")
all_files.append("data/jy_rc203.txt")
all_files.append("data/jy_rc204.txt")
all_files.append("data/jy_rc205.txt")
all_files.append("data/jy_rc206.txt")
all_files.append("data/jy_rc207.txt")
all_files.append("data/jy_rc208.txt")
all_files.append("data/jy_rc209.txt")
all_files.append("data/jy_rc210.txt")
all_files.append("data/jy_rc211.txt")

all_files_param.append('params_WW/params_rc200_50_UB_given/rc_201.json')
all_files_param.append('params_WW/params_rc200_50_UB_given/rc_202.json')
all_files_param.append('params_WW/params_rc200_50_UB_given/rc_203.json')
all_files_param.append('params_WW/params_rc200_50_UB_given/rc_204.json')
all_files_param.append('params_WW/params_rc200_50_UB_given/rc_205.json')
all_files_param.append('params_WW/params_rc200_50_UB_given/rc_206.json')
all_files_param.append('params_WW/params_rc200_50_UB_given/rc_207.json')
all_files_param.append('params_WW/params_rc200_50_UB_given/rc_208.json')
all_files_param.append('params_WW/params_rc200_50_UB_given/rc_209.json')
all_files_param.append('params_WW/params_rc200_50_UB_given/rc_210.json')
all_files_param.append('params_WW/params_rc200_50_UB_given/rc_211.json')

all_out_files=[]
all_out_files.append("../all_willem_rez/WillemRez_r200_50_UB/jy_rc201.txt")
all_out_files.append("../all_willem_rez/WillemRez_r200_50_UB/jy_rc202.txt")
all_out_files.append("../all_willem_rez/WillemRez_r200_50_UB/jy_rc203.txt")
all_out_files.append("../all_willem_rez/WillemRez_r200_50_UB/jy_rc204.txt")
all_out_files.append("../all_willem_rez/WillemRez_r200_50_UB/jy_rc205.txt")
all_out_files.append("../all_willem_rez/WillemRez_r200_50_UB/jy_rc206.txt")
all_out_files.append("../all_willem_rez/WillemRez_r200_50_UB/jy_rc207.txt")
all_out_files.append("../all_willem_rez/WillemRez_r200_50_UB/jy_rc208.txt")








in_fold="data/"

my_json_input_path="mid_jnk"
for k in range(0,12):

    
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