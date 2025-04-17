import numpy as np

def grab_default_params():

	my_params=dict()
	#my_params['num_LA']=0
	my_params['turn_off_time_windows']=False
	my_params['deactivate_time_graph']=False
	my_params['deactivate_demand_graph']=False
	my_params['epsilon']=.0001
	my_params['weight_compress']=.01	#my_params['max_ILP_time']=300
	#my_params['use_integ_y']=0
	my_params['num_cust_use']=25
	my_params['dem_step_sz']=1
	my_params['time_step_sz']=10
	my_params['use_compression']=True
	my_params['max_iterations_loop_compress_project']=200
	#my_params['pre_opt_LA']=True
	#my_params['use_testing_LP_better']=True
	#my_params['use_adapted_time_thresh']=True
	#my_params['use_adapted_dem_thresh']=True
	#my_params['max_iter_no_improve']=10
	#my_params['do_remove_la']=1 #option that is not really turned on
	#my_params['ignore_bad_duals']=0# option for ignoring flow issue
	#my_params['epsilon_LA_neigh']=.0001
	#my_params['min_inc_2_reset']=.001
	my_params['do_round_dist_times']=1
	my_params['my_shift_bet_time_win']=1
	#my_params['do_restart_LA_at_count_0']=0
	#my_params['prefer_branch_on_y']=0
	my_params['num_terms_per_bin_init_construct']=100
	return my_params