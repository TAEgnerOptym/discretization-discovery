from call_and_run_code import call_and_run_code
import os

param_file_path=[]

all_files_param=[]
all_files=[]

param_fold='params_WW/params_c100_100_UB_given_cut_and_sol/'


all_files.append("data/jy_c101.txt")
all_files.append("data/jy_c102.txt")
all_files.append("data/jy_c103.txt")
all_files.append("data/jy_c104.txt")
all_files.append("data/jy_c105.txt")
all_files.append("data/jy_c106.txt")
all_files.append("data/jy_c107.txt")
all_files.append("data/jy_c108.txt")
all_files.append("data/jy_c109.txt")

all_files_param.append(param_fold+'c101.txt')
all_files_param.append(param_fold+'c102.txt')
all_files_param.append(param_fold+'c103.txt')
all_files_param.append(param_fold+'c104.txt')
all_files_param.append(param_fold+'c105.txt')
all_files_param.append(param_fold+'c106.txt')
all_files_param.append(param_fold+'c107.txt')
all_files_param.append(param_fold+'c108.txt')
all_files_param.append(param_fold+'c109.txt')

out_fold_path="../WillRezAbl/C100_100_cust/"


params_prempt=dict()

option_use="cuts_off_graph_on"

if option_use=="cuts_off_graph_on":
    out_fold_path=out_fold_path+"cuts_off_graphs_on/"
    params_prempt['use_ineq']=False

if option_use=="normal":
    out_fold_path=out_fold_path+"normal/"
if option_use=="graphs_offCutsOn":
    out_fold_path=out_fold_path+"graphsOffCutsOn/"
    params_prempt['use_NG_graph']=False
    params_prempt['use_time_graph']=False
    params_prempt['use_dem_graph']=False
    params_prempt['use_ineq']=True
if option_use=="no_cuts_or_graphs":
    out_fold_path=out_fold_path+"graphsOffCutsOn/"
    params_prempt['use_NG_graph']=False
    params_prempt['use_time_graph']=False
    params_prempt['use_dem_graph']=False
    params_prempt['use_ineq']=False





all_out_files=[]
all_out_files.append(out_fold_path+"jy_c101.txt")
all_out_files.append(out_fold_path+"jy_c102.txt")
all_out_files.append(out_fold_path+"jy_c103.txt")
all_out_files.append(out_fold_path+"jy_c104.txt")
all_out_files.append(out_fold_path+"jy_c105.txt")
all_out_files.append(out_fold_path+"jy_c106.txt")
all_out_files.append(out_fold_path+"jy_c107.txt")
all_out_files.append(out_fold_path+"jy_c108.txt")
all_out_files.append(out_fold_path+"jy_c109.txt")


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
        call_and_run_code(input_file_path, param_file_path, my_json_input_path, output_file_path,params_prempt)
    else:
        print(f"Output file {output_file_path} already exists. Skipping call.")