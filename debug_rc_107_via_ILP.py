from debug_baseline import baseline_solver
import pickle
import numpy as np
from typing import Dict, DefaultDict, Set, List
from full_solver import full_solver
loaded_object=[]
with open("my_object.pkl", "rb") as f:
    loaded_object = pickle.load(f)

    fabian_solution=[[50],
		[11, 12, 14, 47, 17, 16, 15 ,13, 9, 10],
		[23, 25, 21, 49, 19, 18, 48, 22, 20, 24], 
		[2, 6, 7, 8, 5, 3, 1, 45, 46, 4],
		[31, 29, 27, 28, 26, 34, 32, 30, 33], 
		[41, 38, 42, 44, 43, 40, 37, 35, 36, 39],
	]
    jy_route_list=[]
    start_depot=50
    end_depot=51
    jy_costs=[]
    all_valid_actions=[]
    for my_route in fabian_solution:
        this_jy_route=[start_depot]
        for z_tmp in my_route:
            z=z_tmp-1
            this_jy_route.append(z)
        this_jy_route.append(end_depot)
        jy_route_list.append(this_jy_route)
    all_valid_edges=[]
    all_invalid_edges=loaded_object.action_2_cost.keys()
    all_invalid_edges=set(all_invalid_edges)
    for this_route in jy_route_list:
        for i in range(0,len(this_route)-1):
            u=this_route[i]
            v=this_route[i+1]
            this_act='act_'+str(u)+'_'+str(v)
            all_valid_edges.append(this_act)
            all_invalid_edges.remove(this_act)

    tot_cost=0
    print('jy_route_list')
    print(jy_route_list)
    M=loaded_object.my_VRP
    my_sol=DefaultDict(float)

    for my_route in jy_route_list:
        cur_cap_rem=M.vehicle_capacity
        cur_cost=0
        cur_time_rem=M.early_start_full[start_depot]
        
        for i in range(0,len(my_route)-1):
            cur_loc=my_route[i]
            next_loc=my_route[i+1]
            
            cur_cap_rem_dep_next=cur_cap_rem-M.dem_full[next_loc]
            if cur_cap_rem_dep_next<-0.0001:
                input('error here cap infeasible')
            if M.orig_dist_mat_full[cur_loc,next_loc]==np.inf or M.dist_mat_full[cur_loc,next_loc]==np.inf:
                input('edge infeasible')
            cur_cost=cur_cost+M.orig_dist_mat_full[cur_loc,next_loc]
            arrival_time_next=cur_time_rem-M.dist_mat_full[cur_loc,next_loc]
            early_time_next=M.early_start_full[next_loc]
            late_time_next=M.late_start_full[next_loc]
            if late_time_next>arrival_time_next:
                print('my_route')
                print(my_route)
                print('i')
                print(i)
                print('arrival_time_next')
                print(arrival_time_next)
                print('early_time_next,late_time_next')
                print([early_time_next,late_time_next])
                input('error here time infeasible')
            #if abs(arrival_time_next-early_time_next)<0.5:
            #    print(arrival_time_next)
            #    print(early_time_next)
            #    print('[cur_loc,next_loc]')
            #    print([cur_loc,next_loc])
            #    input('maybe this is the issue')
            cur_time_rem=min([arrival_time_next,early_time_next])  

            #act
            act_var='act_'+str(cur_loc)+'_'+str(next_loc)
            my_sol[act_var]=1
            if next_loc!=51:
                time_var='delta_timeRem_'+str(next_loc)
                cap_var='delta_capRem_'+str(next_loc)
                my_sol[time_var]=cur_time_rem
                my_sol[cap_var]=cur_cap_rem
            cur_cap_rem=cur_cap_rem_dep_next

        tot_cost=tot_cost+cur_cost

    for act in all_invalid_edges:
        loaded_object.full_input_dict['action2Cost'][act]=1000000

    my_base=baseline_solver(loaded_object,True,False,all_valid_edges,all_invalid_edges,my_sol)
    input('ready run')
    my_input_dict=loaded_object.full_input_dict
    my_opt=loaded_object.jy_opt
    output_file_path=loaded_object.output_file_path
    tmp=full_solver(my_input_dict,my_opt,output_file_path)