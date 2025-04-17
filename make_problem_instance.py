import sys
sys.path.append("pre_process")
from vrp_instance_class import vrp_instance_class
from grab_default_params import grab_default_params
from grab_params import grab_params
from naive_pre import *
from time_graph import time_graph
from dem_graph import dem_graph
from jy_make_input_file_no_la import jy_make_input_file_no_la
import json
def make_problem_instance(input_file_path,my_params,my_json_file_path):
    my_instance=vrp_instance_class(input_file_path,my_params)
    dem_thresh=naive_get_dem_thresh_list(my_instance,int(my_params['dem_step_sz']))
    time_thresh=naive_get_time_thresh_list(my_instance,int(my_params['time_step_sz']))

    #print(my_instance.NC)
    my_dem_graph=dem_graph(my_instance,dem_thresh)
    my_time_graph=time_graph(my_instance,time_thresh)

    data=[]
    my_object_no_la=jy_make_input_file_no_la(my_instance,my_dem_graph,my_time_graph,int(my_params['num_terms_per_bin_init_construct']))
    data=my_object_no_la.out_dict


    with open(my_json_file_path, 'w') as file:
        json.dump(data, file)