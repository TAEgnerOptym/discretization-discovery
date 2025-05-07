from call_and_run_code import call_and_run_code
import os
all_files=[]

if 0<1:

    all_files.append("jy_r101.txt")
    all_files.append("jy_r107.txt")
    all_files.append("jy_r102.txt")
    all_files.append("jy_r103.txt")
    all_files.append("jy_r104.txt")
    all_files.append("jy_r105.txt")
    all_files.append("jy_r106.txt")
    all_files.append("jy_r108.txt")
    all_files.append("jy_r109.txt")
    all_files.append("jy_r110.txt")
    all_files.append("jy_r111.txt")
    all_files.append("jy_r112.txt")

if 0>1:
    all_files.append("jy_rc101.txt")
    all_files.append("jy_rc102.txt")
    all_files.append("jy_rc103.txt")
    all_files.append("jy_rc104.txt")
    all_files.append("jy_rc105.txt")
    all_files.append("jy_rc106.txt")
    all_files.append("jy_rc107.txt")
    all_files.append("jy_rc108.txt")


if 0>1:
    all_files.append("jy_c101.txt")
    all_files.append("jy_c102.txt")
    all_files.append("jy_c103.txt")
    all_files.append("jy_c104.txt")
    all_files.append("jy_c105.txt")
    all_files.append("jy_c106.txt")
    all_files.append("jy_c107.txt")
    all_files.append("jy_c108.txt")
    all_files.append("jy_c109.txt")

if 0>1:


    all_files.append("jy_rc201.txt")
    all_files.append("jy_rc202.txt")
    all_files.append("jy_rc203.txt")
    all_files.append("jy_rc204.txt")
    all_files.append("jy_rc205.txt")
    all_files.append("jy_rc206.txt")
    all_files.append("jy_rc207.txt")
    all_files.append("jy_rc208.txt")


    all_files.append("jy_r201.txt")
    all_files.append("jy_r202.txt")
    all_files.append("jy_r203.txt")
    all_files.append("jy_r204.txt")
    all_files.append("jy_r205.txt")
    all_files.append("jy_r206.txt")
    all_files.append("jy_r207.txt")
    all_files.append("jy_r208.txt")
    all_files.append("jy_r209.txt")
    all_files.append("jy_r210.txt")
    all_files.append("jy_r211.txt")


    all_files.append("jy_c201.txt")
    all_files.append("jy_c202.txt")
    all_files.append("jy_c203.txt")
    all_files.append("jy_c204.txt")
    all_files.append("jy_c205.txt")
    all_files.append("jy_c206.txt")
    all_files.append("jy_c207.txt")
    all_files.append("jy_c208.txt")
#all_files.append("jy_c109.txt")

in_fold="data/"
out_fold="no_cap_post_fix_alledged/out_"
my_json_input_path="mid_jnk"
param_file_path="my_params_50.json"
for my_file in all_files:
    input_file_path=in_fold+my_file
    output_file_path=out_fold+my_file
    print('input_file_path')
    print(input_file_path)
    print("my_file")
    print(my_file)
    if 1>0 or not os.path.exists(output_file_path):
        call_and_run_code(input_file_path, param_file_path, my_json_input_path, output_file_path)
    else:
        print(f"Output file {output_file_path} already exists. Skipping call.")