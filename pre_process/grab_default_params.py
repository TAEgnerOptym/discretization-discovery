import numpy as np

def grab_default_params():

	my_params=dict()
	my_params['turn_off_time_windows']=False
	my_params['deactivate_time_graph']=False
	my_params['deactivate_demand_graph']=False
	my_params['epsilon']=.0001
	my_params['weight_compress']=.01	#my_params['max_ILP_time']=300
	my_params['num_cust_use']=25
	my_params['dem_step_sz']=1
	my_params['time_step_sz']=10
	my_params['use_compression']=True
	my_params['max_iterations_loop_compress_project']=200
	my_params['do_round_dist_times']=1
	my_params['my_shift_bet_time_win']=1
	my_params['num_terms_per_bin_init_construct']=10000
	my_params['allOneBig_init']=1
	my_params['min_inc_2_compress']=.01
	my_params['save_graph_each_iter']=0
	my_params['use_Xpress']=False
	#my_params['xpress_file_loc']='/Users/julian/Documents/FICO\ Xpress\ Config/xpauth.xpr'
	my_params['xpress_file_loc']='/Users/julian/Documents/FICO_Xpress_Config/xpauth.xpr'
	my_params['run_baseline']=True #running hte baseline solver
	my_params['verbose']=False #running hte baseline solver
	my_params['use_NG_graph']=1
	my_params['num_NG']=4
	my_params['in_demo_mode']=0
	my_params['threshold_split']=0.01
	my_params['offset_cost_edge_project']=0#-.00001
	return my_params