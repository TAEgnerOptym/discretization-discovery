import json

# Step 1: Load the JSON file

#my_file_name='FANCY_WITHOUT_BINARY_sample_json_output'
my_file_name_list=[]
my_file_name_list.append('FANCY_WITHOUT_BINARY_sample_json_output')
my_file_name_list.append('NO_FANCY_NO_BINARY_sample_json_output')
my_file_name_list.append('FANCY_WITH_BIN_sample_json_output')
my_file_name_list.append('NO_FANCY_WITH_BIN_COMPRESS_sample_json_output')

for my_file_name in my_file_name_list:

    my_input='../ALL_JSON_BIG/'+my_file_name+'.json'
    my_output='../ALL_JSON_BIG/OUT_'+my_file_name+'.txt'
    print(my_input)
    with open(my_input, "r") as f:
        data = json.load(f)

    # Step 2: Extract the log string
    log_str = data.get("OUR_gurobi_MILP_str", "")

    # Step 3: Replace \r with \n
    cleaned_str = log_str.replace('\r', '\n')

    # Step 4: Write to a new file
    with open(my_output, "w") as f_out:
        f_out.write(cleaned_str)