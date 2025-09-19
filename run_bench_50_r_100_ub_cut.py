from call_and_run_code import call_and_run_code
import os

param_file_path=[]

all_files_param=[]
all_files=[]
all_files.append("data/jy_r101.txt")
all_files.append("data/jy_r102.txt")
all_files.append("data/jy_r103.txt")
all_files.append("data/jy_r104.txt")
all_files.append("data/jy_r105.txt")
all_files.append("data/jy_r106.txt")
all_files.append("data/jy_r107.txt")
all_files.append("data/jy_r108.txt")
all_files.append("data/jy_r109.txt")
all_files.append("data/jy_r110.txt")
all_files.append("data/jy_r111.txt")
all_files.append("data/jy_r112.txt")

all_files_param.append('params_WW/params_r100_50_UB_given_cut/r_101.json')
all_files_param.append('params_WW/params_r100_50_UB_given_cut/r_102.json')
all_files_param.append('params_WW/params_r100_50_UB_given_cut/r_103.json')
all_files_param.append('params_WW/params_r100_50_UB_given_cut/r_104.json')
all_files_param.append('params_WW/params_r100_50_UB_given_cut/r_105.json')
all_files_param.append('params_WW/params_r100_50_UB_given_cut/r_106.json')
all_files_param.append('params_WW/params_r100_50_UB_given_cut/r_107.json')
all_files_param.append('params_WW/params_r100_50_UB_given_cut/r_108.json')
all_files_param.append('params_WW/params_r100_50_UB_given_cut/r_109.json')
all_files_param.append('params_WW/params_r100_50_UB_given_cut/r_110.json')
all_files_param.append('params_WW/params_r100_50_UB_given_cut/r_111.json')
all_files_param.append('params_WW/params_r100_50_UB_given_cut/r_112.json')

all_out_files=[]
all_out_files.append("../all_willem_rez_given_cut/WillemRez_r100_50_UB/jy_r101.txt")
all_out_files.append("../all_willem_rez_given_cut/WillemRez_r100_50_UB/jy_r102.txt")
all_out_files.append("../all_willem_rez_given_cut/WillemRez_r100_50_UB/jy_r103.txt")
all_out_files.append("../all_willem_rez_given_cut/WillemRez_r100_50_UB/jy_r104.txt")
all_out_files.append("../all_willem_rez_given_cut/WillemRez_r100_50_UB/jy_r105.txt")
all_out_files.append("../all_willem_rez_given_cut/WillemRez_r100_50_UB/jy_r106.txt")
all_out_files.append("../all_willem_rez_given_cut/WillemRez_r100_50_UB/jy_r107.txt")
all_out_files.append("../all_willem_rez_given_cut/WillemRez_r100_50_UB/jy_r108.txt")
all_out_files.append("../all_willem_rez_given_cut/WillemRez_r100_50_UB/jy_r109.txt")
all_out_files.append("../all_willem_rez_given_cut/WillemRez_r100_50_UB/jy_r110.txt")
all_out_files.append("../all_willem_rez_given_cut/WillemRez_r100_50_UB/jy_r111.txt")
all_out_files.append("../all_willem_rez_given_cut/WillemRez_r100_50_UB/jy_r112.txt")







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