from vrp_instance_class import vrp_instance_class
from grab_default_params import grab_default_params
from grab_params import grab_params
from naive_pre import *
from time_graph import time_graph
from dem_graph import dem_graph
from jy_make_input_file_no_la import jy_make_input_file_no_la
import json
import ast

import ast


input_file_path='../data/jy_c103.txt'
my_params=grab_default_params()
print(my_params)
my_instance=vrp_instance_class(input_file_path,my_params)
dem_thresh=naive_get_dem_thresh_list(my_instance,int(my_params['dem_step_sz']))
time_thresh=naive_get_time_thresh_list(my_instance,int(my_params['time_step_sz']))

print('dem_thresh')
print(dem_thresh)
print('time_thresh')
print(time_thresh)
print('time_thresh[1]')
print(time_thresh[1])

#print(my_instance.NC)
my_dem_graph=dem_graph(my_instance,dem_thresh)
my_time_graph=time_graph(my_instance,time_thresh)

data=[]
num_terms_per_bin=100
my_object_no_la=jy_make_input_file_no_la(my_instance,my_dem_graph,my_time_graph,num_terms_per_bin)
data=my_object_no_la.out_dict

filename = "jy_data.json"

with open(filename, 'w') as file:
    json.dump(data, file)

