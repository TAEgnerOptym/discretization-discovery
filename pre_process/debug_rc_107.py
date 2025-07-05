
	def eval_vrp_rC107(self):
		fabian_solution=[[50],
		[11, 12, 14, 47, 17, 16, 15 ,13, 9, 10]
		[23, 25, 21, 49, 19, 18, 48, 22, 20, 24], 
		[2, 6, 7, 8, 5, 3, 1, 45, 46, 4],
		[31, 29, 27, 28, 26, 34, 32, 30, 33], 
		[41, 38, 42, 44, 43, 40, 37, 35, 36, 39],
		]
		jy_route_list=[]
		start_depot=50
		end_depot=51
		jy_costs=[]
		for my_route in fabian_solution:
			jy_route=[start_depot]
			cur_cap_rem=M.capacity
			cur_cost=0
			cur_time_rem=M.early_start_full[start_depot]
			cur_location=start_depot
			for z_tmp in my_route:
				z=z_tmp-1
				jy_route.append(z)
			jy_route_list.append(end_depot)
		
		tot_cost=0
		for my_route in jy_route_list:
			cur_cap_rem=self.capacity
			cur_cost=0
			cur_time_rem=self.early_start_full[start_depot]
			
			for i in range(0,len(my_route)-1):
				cur_loc=my_route[i]
				next_loc=my_route[i+1]
				cur_cap_rem_dep_next=cur_cap_rem-M.dem_full[next_loc]
				if cur_cap_rem_dep_next<-0.0001:
					input('error here cap infeasible')
				if self.orig_dist_mat_full[cur_loc,next_loc]==np.inf or self.dist_mat_full[cur_loc,next_loc]==inf:
					input('edge infeasible')
				cur_cost=cur_cost+self.orig_dist_mat_full[cur_loc,next_loc]
				arrival_time_next=cur_time_rem-M.dist_mat_full[cur_loc,next_loc]
				early_time_next=self.early_start_full[next_loc]
				late_time_next=self.late_start_full[next_loc]
				if late_time_next<arrival_time_next:
					input('error here time infeasible')
				cur_time_rem=min([arrival_time_next,cur_time_rem])  
			tot_cost=tot_cost+cur_cost
		
		print('tot_cost')
		print(tot_cost)
		if abs(tot_cost-642.7):
			input('cost infeasbile ')
