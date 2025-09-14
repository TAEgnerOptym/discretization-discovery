import sys
sys.path.append("pre_process")
from vrp_instance_class import vrp_instance_class
from grab_default_params import grab_default_params
from grab_params import grab_params
from naive_pre import *
from time_graph import time_graph
from dem_graph_2 import dem_graph
#from dem_graph import dem_graph
from ng_graph import ng_graph
from jy_make_input_file_no_la import jy_make_input_file_no_la
import json
from valid_ineq_helper_ng_la import *
def make_problem_instance(input_file_path,my_params,my_json_file_path):
    my_instance=vrp_instance_class(input_file_path,my_params)
    dem_thresh=naive_get_dem_thresh_list(my_instance,(my_params['dem_step_sz']))
    time_thresh=naive_get_time_thresh_list(my_instance,(my_params['time_step_sz']))
    
    print('making dem graph')
    my_dem_graph=dem_graph(my_instance,dem_thresh)
    #input('done dem')
    print('done dem graph')
    #input('start dem')
    my_time_graph=time_graph(my_instance,time_thresh)
    my_ng_graph=None
    ng_neigh_by_cust=[]
    if my_params['use_ng']>0.5:
        print('gettign navie neigh')
        [ng_neigh_by_cust,junk]=naive_get_LA_neigh(my_instance,(my_params['num_NG']))
        #print('staritng ng making ')
        #for i in range(0,50):
        #    print('cust neigh.  '+ str(i))
        #    print(ng_neigh_by_cust[i])
        #    print('my_params[num_NG]')
        #    print(my_params['num_NG'])
        #    input('--')
        #input('--')
        if my_params['use_fancy_ng_graph']<0.5:
            
            my_ng_graph=ng_graph(my_instance,ng_neigh_by_cust)
        else:
            #if 1>0:
            #from ng_graph_fancy_slow import ng_graph_fancy_slow
            from ng_neigh_fancy_paper import ng_graph_fancy_slow

            my_ng_graph=ng_graph_fancy_slow(my_instance,ng_neigh_by_cust)
            #else:
            #    from TEST_ng_fancy import ng_graph_fancy_slow
#
#                my_ng_graph=ng_graph_fancy_slow(my_instance,ng_neigh_by_cust)

        #print('done ng making ')
    data=[]
    #print('here to debug this')
    #ng_help_valid_ineq(my_instance,ng_neigh_by_cust)
    #input('done')
    my_object_no_la=jy_make_input_file_no_la(my_instance,my_dem_graph,my_time_graph,int(my_params['num_terms_per_bin_init_construct']),my_ng_graph)
    
    if my_params['use_time_graph']<0.5:
        print('removing time')
        del my_object_no_la.out_dict['h2sinkid']['timeGraph']
        del my_object_no_la.out_dict['h2SourceId']['timeGraph']
        del my_object_no_la.out_dict['graphName2Nodes']['timeGraph']
        del my_object_no_la.out_dict['initGraphNode2AggNode']['timeGraph']
        del my_object_no_la.out_dict['hij2P']['timeGraph']
        my_object_no_la.out_dict['allGraphNames'].remove('timeGraph')

    if my_params['use_dem_graph']<0.5:
        print('removign demand')
        del my_object_no_la.out_dict['h2sinkid']['capGraph']
        del my_object_no_la.out_dict['h2SourceId']['capGraph']
        del my_object_no_la.out_dict['graphName2Nodes']['capGraph']
        del my_object_no_la.out_dict['initGraphNode2AggNode']['capGraph']
        del my_object_no_la.out_dict['hij2P']['capGraph']
        my_object_no_la.out_dict['allGraphNames'].remove('capGraph')
        #del my_object_no_la.out_dict['allGraphNames']['capGraph']
    
    ng_neigh_by_cust2=ng_neigh_by_cust.copy()
    ng_neigh_by_cust2=ng_neigh_by_cust2[0:-2]
    ng_neigh_by_cust2.append([])
    ng_neigh_by_cust2.append([])
    ng_neigh_by_cust2 = [[int(x) for x in sublist] for sublist in ng_neigh_by_cust2]
    my_object_no_la.out_dict['ng_neigh_by_cust']=ng_neigh_by_cust2
    #my_object_no_la.out_dict['ng_graph']=ng_graph
    #print(ng_neigh_by_cust)
    #input('---')
    data=my_object_no_la.out_dict

    

    with open(my_json_file_path, 'w') as file:
        json.dump(data, file)
    return my_instance