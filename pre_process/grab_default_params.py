import numpy as np

def grab_default_params():

	my_params=dict()
	my_params['ParetoEps']=0.00001
	my_params['turn_off_time_windows']=False
	my_params['deactivate_time_graph']=False
	my_params['deactivate_demand_graph']=False
	my_params['epsilon']=.0001
	my_params['weight_slack_projector']=.0000001
	my_params['weight_compress']=.01	#my_params['max_ILP_time']=300
	my_params['num_cust_use']=25
	my_params['dem_step_sz']=1
	my_params['time_step_sz']=10
	my_params['use_compression']=True
	my_params['max_iterations_loop_compress_project']=200
	my_params['do_round_dist_times']=1
	my_params['my_shift_bet_time_win']=0.0001
	my_params['num_terms_per_bin_init_construct']=10000
	my_params['allOneBig_init']=1
	my_params['min_inc_2_compress']=.01
	my_params['save_graph_each_iter']=0
	my_params['use_Xpress']=False
	my_params['use_ineq']=0 #use the inequalities
	#my_params['xpress_file_loc']='/Users/julian/Documents/FICO\ Xpress\ Config/xpauth.xpr'
	my_params['xpress_file_loc']='/Users/julian/Documents/FICO_Xpress_Config/xpauth.xpr'
	my_params['run_baseline']=True #running hte baseline solver
	my_params['verbose']=False #running hte baseline solver
	my_params['use_NG_graph']=1
	my_params['num_NG']=4
	my_params['in_demo_mode']=0
	my_params['threshold_split']=0.01
	my_params['offset_cost_edge_project']=0#-.00001
	my_params['use_classic_compress']=1
	my_params['use_classic_compress_last']=1
	my_params['num_thresh_split_projector']=10#number of split points in projector 
	my_params['lplb_solver']=1 #default solver 
	my_params['compresss_solver']=0 #default solver 
	my_params['proj_solver']=1 #default solver 
	my_params['use_julians_custom_lp_solver']=0
	my_params['do_ilp']=1# do ilp 
	my_params['roundingDiscretization_num_digits_keep']=3
	my_params["doExrtraRemovalEdgesLB"]=0
	my_params["use_gurobi"]=0
	my_params["max_ILP_time"]=1000
	my_params['use_fancy_ng_graph']=1
	my_params['use_dem_graph']=1
	my_params['use_time_graph']=1
	my_params['num_multiplys_demand']=0
	my_params['DEBUG_ALLOW_DUMB_EDGES']=0
	my_params['add_all_ng_split_at_end']=0
	my_params['do_split_based_init']=0
	my_params["think_compress"]=0,
	my_params["use_delta_in_milp"]=1,
	my_params["use_delta_in_lp"]=1
	my_params["turn_off_non_active_fg_project"]=0
	my_params['use_new_valid_ineq']=0
	my_params['digit_mult_use']=10#used for rounding distances

	my_params['use_branch_on_g']=True
	my_params['LAB_MP_ON']=True
	my_params['LAB_MP_neigh_use_power']=10
	my_params['LAB_MP_num_ineq_use']=10
	my_params['do_remove_actions_from_incumbant']=True
	my_params['ub_use_remove']=-1 #meaning dont use 
	my_params['use_packing_in_construction']=False
	my_params['maxVarsAdd_in_ITER']=10
	my_params['use_ng_size_init']=0
	my_params['do_presolve']=0
	my_params['use_diving_ineq']=0



	my_params['BEND_MIN_LP_OBJECTIVE_CUT']=0.1
	my_params['BEND_USE_RAND']=0
	my_params['BEND_MAX_SIZE_CHOOSE_K']=230
	my_params['BEND_VAL_STOP_ADDING_CUTS']=100000000
	my_params['BEND_MY_SIZES_USE']=[9]

	my_params['NUMERICAL_BEND_OFFSET_COST_CUT']=0.0001
	my_params['NUMERICAL_BEND_EPSILON_MULT_PARETO_OBJ']=0.001

	my_params['NO_PARETO_NUMERICAL_BEND_OFFSET_COST_CUT']=0#0.0001
	my_params['NO_PARETO_BEND_VAL_STOP_ADDING_CUTS']=1


	return my_params