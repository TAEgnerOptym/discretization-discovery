import numpy as np
class vrp_instance_class:


	#PROBLEM DESCIPTION ON OUTPUT
	#num_cust:  number of customers ;; NC
	#dist_mat_full:  distnace PLUS SERVICE matrix with Nc and NC+1 being starting and ending depot
	#orig_dist_mat_full:  distance without service
	#service_time:  has the service time for each customer
	#dem:  demand
	#early_start:  earlies start
	#late_start:  latest start
	
	
	def  get_dist_2_depot(self,cust_1):
	
		x1=self.x_loc[cust_1]
		y1=self.y_loc[cust_1]
		
		x2=self.depot_x
		y2=self.depot_y
		
		xd=np.power(x1-x2,2)
		yd=np.power(y1-y2,2)
		act_dist=np.sqrt(xd+yd)
		if self.my_params['do_round_dist_times']>0.5:
			act_dist=np.floor(act_dist*10)/10
		return act_dist
			
	def	 get_dist_cust_2_cust(self,cust_1,cust_2):
		
		x1=self.x_loc[cust_1]
		y1=self.y_loc[cust_1]
		d1=self.dem[cust_1]
		e1=self.early_start[cust_1]
		f1=self.late_start[cust_1]
		
		x2=self.x_loc[cust_2]
		y2=self.y_loc[cust_2]
		d2=self.dem[cust_2]
		e2=self.early_start[cust_2]
		f2=self.late_start[cust_2]
		
		xd=np.power(x1-x2,2)
		yd=np.power(y1-y2,2)
		act_dist=np.sqrt(xd+yd)
		act_dist=act_dist+self.service_time[cust_1]
		if e1-act_dist<f2:
			act_dist=np.inf
		
		if d2+d1>self.vehicle_capacity:	
			act_dist=np.inf
		if cust_1==cust_2:
			act_dist=np.inf
		if self.my_params['do_round_dist_times']>0.5:
			act_dist=np.floor(act_dist*10)/10

		return act_dist
	
	
	def round_times_first(self):
		self.early_start=np.floor(self.early_start)
		self.late_start=np.floor(self.late_start)
	def round_restrict(self):
		self.early_start=np.floor(self.early_start)
		self.late_start=np.floor(self.late_start)
		self.orig_dist_mat_full=np.floor(self.orig_dist_mat_full)
		self.early_start_full=np.floor(self.early_start_full)
		self.late_start_full=np.floor(self.late_start_full)
		self.dist_mat_full=np.floor(self.dist_mat_full)
		self.dist_2_depot=np.floor(self.dist_2_depot)
		self.dist_mat=np.floor(self.dist_mat)
	def __init__(self,input_file_path,my_params):

		input_matrix = np.loadtxt(input_file_path, dtype='double', delimiter=',')
		self.my_params=my_params
		self.depot_x=input_matrix[0,0]
		self.depot_y=input_matrix[0,1]
		self.depot_start_time=100000
		if input_matrix.shape[1]==6:
			self.depot_start_time=input_matrix[0,3]

		self.vehicle_capacity=input_matrix[0,2]
		
		self.num_cust=np.min([my_params['num_cust_use'],input_matrix.shape[0]-1])



		self.x_loc=input_matrix[1:self.num_cust+1,0]
		self.y_loc=input_matrix[1:self.num_cust+1,1]
		self.x_loc=np.concatenate([self.x_loc,[self.depot_x,self.depot_x]])
		self.y_loc=np.concatenate([self.y_loc,[self.depot_y,self.depot_y]])

		self.dem=input_matrix[1:self.num_cust+1,2]
		self.dem_full=np.concatenate([input_matrix[1:(self.num_cust+1),2],np.zeros(2)],0)


		if input_matrix.shape[1]==6:
			self.early_start=input_matrix[1:self.num_cust+1,3]
			self.late_start=input_matrix[1:self.num_cust+1,4]
			self.service_time=input_matrix[1:self.num_cust+1,5]
		else:
			self.early_start=(self.x_loc*0)+100000
			self.late_start=(self.x_loc*0)
			self.service_time=(self.x_loc*0)
			self.depot_start_time=100000
		
		
		if my_params['turn_off_time_windows']==True:
			self.early_start=(self.early_start*0)+10000
			self.late_start=(self.early_start*0)
			self.service_time=(self.early_start*0)
			self.depot_start_time=10000
		self.turn_off_time_windows=my_params['turn_off_time_windows']
		#if self.my_params['do_round_dist_times']>0.5:
			#self.round_times_first()

		tmp=np.asarray([self.depot_start_time,0])
		self.late_start_full=np.concatenate([self.late_start,tmp],0)
		tmp=np.asarray([self.depot_start_time,self.depot_start_time])
		self.early_start_full=np.concatenate([self.early_start,tmp],0)
		
		#print(self.early_start_full)
		#input('---')
		
		
		self.dist_mat=np.zeros((self.num_cust,self.num_cust));
		self.dist_2_depot=np.zeros(self.num_cust);
		
		for i in range(0,self.num_cust):
			self.dist_2_depot[i]=self.get_dist_2_depot(i)
			
			self.early_start[i]=min([self.early_start[i],self.depot_start_time-self.dist_2_depot[i]])
			
			for j in range(0,self.num_cust):
				self.dist_mat[i,j]=self.get_dist_cust_2_cust(i,j)
		self.dist_mat_full=np.zeros((self.num_cust+2,self.num_cust+2));
		self.orig_dist_mat_full=np.zeros((self.num_cust+2,self.num_cust+2));
		for i in range(0,self.num_cust+2):
			for j in range(0,self.num_cust+2):
				if i>=self.num_cust and j<self.num_cust:
					self.dist_mat_full[i,j]=self.dist_2_depot[j]
					self.orig_dist_mat_full[i,j]=self.dist_2_depot[j]
				if i<self.num_cust and j>=self.num_cust:
					self.dist_mat_full[i,j]=self.dist_2_depot[i]+self.service_time[i]
					self.orig_dist_mat_full[i,j]=self.dist_2_depot[i]
					
				if i<self.num_cust and j<self.num_cust:
					self.dist_mat_full[i,j]=self.dist_mat[i,j]
					self.orig_dist_mat_full[i,j]=self.dist_mat[i,j]-self.service_time[i]

				if i>=self.num_cust and j>=self.num_cust:
					self.dist_mat_full[i,j]=np.inf
					self.orig_dist_mat_full[i,j]=np.inf
		self.dist_mat_full[self.num_cust+1,:]=np.inf
		self.dist_mat_full[:,self.num_cust]=np.inf
		self.orig_dist_mat_full[self.num_cust+1,:]=np.inf
		self.orig_dist_mat_full[:,self.num_cust]=np.inf
		
		
		#self.round_restrict()