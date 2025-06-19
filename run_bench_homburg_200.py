from call_and_run_code import call_and_run_code
import os
all_files=[]

if 0<1:

    all_files.append("jyHOM_C1_2_1.txt")
    all_files.append("jyHOM_C1_2_2.txt")
    all_files.append("jyHOM_C1_2_3.txt")
    all_files.append("jyHOM_C1_2_4.txt")
    all_files.append("jyHOM_C1_2_5.txt")
    all_files.append("jyHOM_C1_2_6.txt")
    all_files.append("jyHOM_C1_2_7.txt")
    all_files.append("jyHOM_C1_2_8.txt")
    all_files.append("jyHOM_C1_2_9.txt")
    all_files.append("jyHOM_C1_2_10.txt")

if 0>1:
    all_files.append("jyHOM_C2_2_1.txt")
    all_files.append("jyHOM_C2_2_2.txt")
    all_files.append("jyHOM_C2_2_3.txt")
    all_files.append("jyHOM_C2_2_4.txt")
    all_files.append("jyHOM_C2_2_5.txt")
    all_files.append("jyHOM_C2_2_6.txt")
    all_files.append("jyHOM_C2_2_7.txt")
    all_files.append("jyHOM_C2_2_8.txt")
    all_files.append("jyHOM_C2_2_9.txt")
    all_files.append("jyHOM_C2_2_10.txt")


if 0>1:
    all_files.append("jyHOM_R1_2_1.txt")
    all_files.append("jyHOM_R1_2_2.txt")
    all_files.append("jyHOM_R1_2_3.txt")
    all_files.append("jyHOM_R1_2_4.txt")
    all_files.append("jyHOM_R1_2_5.txt")
    all_files.append("jyHOM_R1_2_6.txt")
    all_files.append("jyHOM_R1_2_7.txt")
    all_files.append("jyHOM_R1_2_8.txt")
    all_files.append("jyHOM_R1_2_9.txt")
    all_files.append("jyHOM_R1_2_10.txt")

if 0>1:


    all_files.append("jyHOM_R2_2_1.txt")
    all_files.append("jyHOM_R2_2_2.txt")
    all_files.append("jyHOM_R2_2_3.txt")
    all_files.append("jyHOM_R2_2_4.txt")
    all_files.append("jyHOM_R2_2_5.txt")
    all_files.append("jyHOM_R2_2_6.txt")
    all_files.append("jyHOM_R2_2_7.txt")
    all_files.append("jyHOM_R2_2_8.txt")
    all_files.append("jyHOM_R2_2_9.txt")
    all_files.append("jyHOM_R2_2_10.txt")



if 0>1:


    all_files.append("jyHOM_RC1_2_1.txt")
    all_files.append("jyHOM_RC1_2_2.txt")
    all_files.append("jyHOM_RC1_2_3.txt")
    all_files.append("jyHOM_RC1_2_4.txt")
    all_files.append("jyHOM_RC1_2_5.txt")
    all_files.append("jyHOM_RC1_2_6.txt")
    all_files.append("jyHOM_RC1_2_7.txt")
    all_files.append("jyHOM_RC1_2_8.txt")
    all_files.append("jyHOM_RC1_2_9.txt")
    all_files.append("jyHOM_RC1_2_10.txt")


if 0>1:


    all_files.append("jyHOM_RC2_2_1.txt")
    all_files.append("jyHOM_RC2_2_2.txt")
    all_files.append("jyHOM_RC2_2_3.txt")
    all_files.append("jyHOM_RC2_2_4.txt")
    all_files.append("jyHOM_RC2_2_5.txt")
    all_files.append("jyHOM_RC2_2_6.txt")
    all_files.append("jyHOM_RC2_2_7.txt")
    all_files.append("jyHOM_RC2_2_8.txt")
    all_files.append("jyHOM_RC2_2_9.txt")
    all_files.append("jyHOM_RC2_2_10.txt")

#all_files.append("jy_c109.txt")

in_fold="dataHOM/"
#out_fold="../ALL_JSON_BIG/out_R100_50_yes_delta/"
#out_fold="../ALL_JSON_BIG/out_R100_50_no_reset_yes_reset_each_end/"
out_fold="../ALL_JSON_BIG/OUT_Homburg_200_2/"
my_json_input_path="mid_jnk"
param_file_path="my_params_Homburg_200.json"
for my_file in all_files:
    input_file_path=in_fold+my_file
    output_file_path=out_fold+my_file
    print('input_file_path')
    print(input_file_path)
    print("my_file")
    print(my_file)
    if 1>0 :#or not os.path.exists(output_file_path):
    #if   not os.path.exists(output_file_path):
        call_and_run_code(input_file_path, param_file_path, my_json_input_path, output_file_path)
    else:
        print(f"Output file {output_file_path} already exists. Skipping call.")