

import numpy as np

class dem_graph:  


	def check_if_con_poss(self,d1,d2,d_inter):

		my_state=False
		if  (d1-d_inter)>=d2:
			my_state=True
		return my_state
	def check_add_edge(self,i,j):
		MV=self.my_vrp
		Nc=self.Nc
		u=i[0]
		v=j[0]
		di_minus=i[1]
		di_plus=i[2]

		dj_minus=j[1]
		dj_plus=j[2]


		i_has_child=(di_minus>MV.dem_full[u]) and (u<Nc)
		j_has_parent=(dj_plus<MV.vehicle_capacity) and (v<Nc)
		cond2=False
		cond3=False
		if u!=v:
			cond1=self.check_if_con_poss(di_plus,dj_minus,MV.dem_full[u])
			if i_has_child:
				cond2=self.check_if_con_poss(di_minus-self.my_shift_bet_time_win,dj_minus,MV.dem_full[u])
			if j_has_parent:
				cond3=self.check_if_con_poss(di_plus,dj_plus+self.my_shift_bet_time_win,MV.dem_full[u])
		else:
			cond1=(di_minus-self.dem_step_sz==dj_plus)
		if cond1==True and cond2==False and cond3==False:
			this_tup=tuple([i,j])

			#print('adding')
			#print(this_tup)
			#print('di_minus')
			#print(di_minus)
			#print('dj_plus')
			#print(dj_plus)
			#print('self.dem_step_sz')
			#print(self.dem_step_sz)
			#input('--')
			self.E.append(this_tup)	
			self.E_orig_backup.append(this_tup)
		
			dist_term=0
			if u!=v and v!=Nc+1:
				dist_term=MV.dem_full[u]
			self.arrival_dem[this_tup]=min([di_plus-dist_term,dj_plus])
			self.depart_dem[this_tup]=min([di_plus,dj_plus+dist_term])
	def make_edges(self):
		MV=self.my_vrp
		DF=MV.dist_mat_full
		Nc=self.Nc
		self.E=[]
		self.E_orig_backup=[]

		CAP=MV.vehicle_capacity
		self.arrival_dem=dict()
		self.depart_dem=dict()
		for u in range(0,Nc+1):
			for v in range(0,Nc+2):

				if DF[u,v]<np.inf or ( u==v and u!=Nc ):
					for i in self.node_list[u]:
						for j in self.node_list[v]:
							self.check_add_edge(i,j)

	def make_nodes(self):
		Nc=self.Nc
		MV=self.my_vrp
		self.d0=MV.vehicle_capacity
		d0=self.d0
		self.node_list=dict()
		self.node_list[Nc]=[]
		self.node_list[Nc].append(tuple([Nc,d0,d0]))
		self.node_list[Nc+1]=[]
		self.node_list[Nc+1].append(tuple([Nc+1,0,d0]))
		for u in range(0,Nc):

			last_val=MV.dem[u]-self.my_shift_bet_time_win#-self.my_shift_bet_time_win#self.dem_step_sz
			self.node_list[u]=[]
			#print(self.dem_thresh[u])
			for this_dem in self.dem_thresh[u]:
				start_d=last_val+self.my_shift_bet_time_win
				end_d=this_dem
				self.node_list[u].append(tuple([u,start_d,end_d]))
				#print('tuple([u,start_d,end_d]')
				#print(tuple([u,start_d,end_d]))
				#input('---')
				last_val=end_d
			if last_val<d0:
				self.node_list[u].append(tuple([u,last_val+self.dem_step_sz,d0]))
			#print('nodes ')
			#print(self.node_list[u])
			#print('MV.dem[u]')
			#print(MV.dem[u])
			#input('--')
	def __init__(self,my_vrp,dem_thresh):
		#dem_thresh has the intermediate thresholds to be used
		self.my_vrp=my_vrp
		self.Nc=my_vrp.num_cust
		self.dem_thresh=dem_thresh
		self.dem_step_sz=self.my_vrp.my_params['dem_step_sz']#-1
		self.my_shift_bet_time_win=1#self.my_vrp.my_params['my_shift_bet_time_win']
		#print('ZZZ making dem nodes')
		#self.make_nodes()
		#print('ZZZ making dem edges')
		#input('--')
		self.make_nodes()
		self.NEW_make_edges()
		debug_on=False
		if debug_on==True:
			self.make_edges()
			self.debug_do_check_agree()
		#print('done dem edges')
		#input('ALL GOOD DUDEs')

	def debug_do_check_agree(self):

		E1=set(self.E_orig_backup)
		E2=set(self.E_novel_back)
		print('running check operations ')
		Err1=set(self.E_orig_backup)-set( self.E_novel_back)
		Err2=set(self.E_novel_back)-set( self.E_orig_backup)
		if len(Err1)>0 or len(Err2)>0:
			input('error here')
			for e1 in self.E_orig_backup:
				if e1 not in self.E_novel_back:
					print('e1')
					print(e1)
					#print('self.node_list[45]')
					#print(self.node_list[45])
					#print('self.node_list[44]')
					#print(self.node_list[44])
					print('self.my_vrp.dem_full[0]')
					print(self.my_vrp.dem_full[0])
					print('self.my_vrp.dem_full[1]')
					print(self.my_vrp.dem_full[1])
					input('errror here pos 1; ')
			
			for e2 in self.E_novel_back:
				if e2 not in self.E_orig_backup:
					print('e2')
					print(e2)
					input('errror here pos 2; ')


	def NEW_make_edges(self):
		MV=self.my_vrp
		DF=MV.dist_mat_full
		DEM=MV.dem_full
		self.DEBUG_edges_uv_tup=dict()
		Nc=self.Nc
		self.E_novel_back=[]
		self.E=[]

		self.node_2_least_cap=dict()
		self.node_2_most_cap=dict()

		for u in range(0,Nc+2):
			self.node_2_least_cap[u]=[]
			self.node_2_most_cap[u]=[]
			for i in range(0,len(self.node_list[u])):
				ti_minus=self.node_list[u][i][1]
				ti_plus=self.node_list[u][i][2]
				self.node_2_least_cap[u].append(ti_minus)
				self.node_2_most_cap[u].append(ti_plus)
		

		for u in range(0,Nc):
			self.DEBUG_edges_uv_tup[tuple([u,u])]=[]
			Nu_i=len(self.node_list[u])
			for i in range(Nu_i-1,0,-1):
				this_tup=tuple([self.node_list[u][i],self.node_list[u][i-1]])
				self.E_novel_back.append(this_tup)
				self.E.append(this_tup)
				self.DEBUG_edges_uv_tup[tuple([u,u])].append(this_tup)
		
		for u in range(0,Nc+1):
			for v in range(0,Nc+2):
				if DF[u,v]<np.inf and u!=v:
					self.NEW_make_edges_helper(u,v)



	def NEW_make_edges_helper(self,u,v):
		MV=self.my_vrp
		veh_cap=MV.vehicle_capacity
		dem_u=MV.dem_full[u]
		Nu_i=len(self.node_list[u])
		Nv_j=len(self.node_list[v])
		self.DEBUG_edges_uv_tup[tuple([u,v])]=[]
		my_j_pos=Nv_j-1
		for i in range(Nu_i-1,-1,-1):
			#earliest_arrival_from_i_to_v=self.node_2_early[u][i]-my_dist
			most_cap_rem_from_i_to_v=self.node_2_most_cap[u][i]-dem_u#veh_cap-dem_u
			
			#earliest_arrival_from_i_CHILD_to_v=-np.inf
			most_cap_rem_from_i_to_CHILD_to_v=-np.inf
			if i>0:
				#earliest_arrival_from_i_CHILD_to_v=self.node_2_early[u][i-1]-my_dist
				most_cap_rem_from_i_to_CHILD_to_v=self.node_2_most_cap[u][i-1]-dem_u
			for j in range(my_j_pos,-1,-1):
				#latest_arrival_allowed=self.node_2_late[v][j]
				least_cap_arrival_allowed=self.node_2_least_cap[v][j]
				flag_cond=0
				#if earliest_arrival_from_i_to_v>= latest_arrival_allowed and latest_arrival_allowed>earliest_arrival_from_i_CHILD_to_v:#-.0001:
				this_tup=tuple([self.node_list[u][i],self.node_list[v][j]])

				#if u==0 and v==1:
				#	print(' in before this_tup')
				#	print(this_tup)
				#	input('--')
				if most_cap_rem_from_i_to_v>= least_cap_arrival_allowed and least_cap_arrival_allowed>most_cap_rem_from_i_to_CHILD_to_v:#-.0001:
					self.E_novel_back.append(this_tup)
					self.E.append(this_tup)
					
					self.DEBUG_edges_uv_tup[tuple([u,v])].append(this_tup)
					my_j_pos=j-1
					#if u==0 and v==1:
					#	print('adding')
					#	print(this_tup)
					#	input('--')
					break
				#else:
				#	print('NOT adding')
				#	print(this_tup)
				#	input('--')
#
				
			