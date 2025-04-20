import numpy as np

class time_graph:  


	def check_if_con_poss(self,t1,t2,dist_term):

		my_state=False
		if  (t1-dist_term)>=t2-.0001:
			my_state=True
		return my_state
	def check_add_edge(self,i,j):
		MV=self.my_vrp
		Nc=self.Nc
		u=i[0]
		v=j[0]
		ti_minus=i[1]
		ti_plus=i[2]

		tj_minus=j[1]
		tj_plus=j[2]

		i_has_child=(ti_minus>MV.late_start_full[u]) and (u<Nc)
		j_has_parent=(tj_plus<MV.early_start_full[v]) and (v<Nc)
		cond2=False
		cond3=False
		cond1=False
		dist_term=MV.dist_mat_full[u,v]
		if u!=v:
			cond1=self.check_if_con_poss(ti_plus,tj_minus,MV.dist_mat_full[u,v])
			if i_has_child:
				cond2=self.check_if_con_poss(ti_minus-self.my_shift_bet_time_win,tj_minus,MV.dist_mat_full[u,v])
			if j_has_parent and self.my_vrp.my_params['deactivate_time_graph']==False:
				cond3=self.check_if_con_poss(ti_plus,tj_plus+self.my_shift_bet_time_win,MV.dist_mat_full[u,v])
		else:
			#cond1=(ti_minus-self.my_shift_bet_time_win==tj_plus)
			cond1=.0001>np.abs(ti_minus-self.my_shift_bet_time_win-tj_plus)
		#print('[cond1,cond2,cond3]')
		#print([cond1,cond2,cond3])

		did_add=False

		if cond1==True and cond2==False and cond3==False:
			this_tup=tuple([i,j])
			self.E.append(this_tup)
			self.E_orig_backup.append(this_tup)
			self.DEBUG_E_orig_tup[tuple([u,v])].append(this_tup)
			dist_term=0
			if u!=v:
				dist_term=ti_plus-MV.dist_mat_full[u,v]
			self.arrival_time[this_tup]=min([ti_plus-dist_term,tj_plus])
			self.departure_time[this_tup]=min([ti_plus,tj_plus+dist_term])
			did_add=True

		return did_add


	
	def make_edges(self):
		MV=self.my_vrp
		DF=MV.dist_mat_full
		Nc=self.Nc
		self.E=[]
		self.E_orig_backup=[]
		self.DEBUG_E_orig_tup=dict()
		self.arrival_time=dict()
		self.departure_time=dict()
		for u in range(0,Nc+1):
			for v in range(0,Nc+2):
				if DF[u,v]<np.inf or ( u==v and u!=Nc ):		
					self.DEBUG_E_orig_tup[tuple([u,v])]=[]
					did_add_yet=False
					for i in self.node_list[u]:
						for j in self.node_list[v]:
							did_add_this=self.check_add_edge(i,j)
							#if i[0]==0 and i[1]==120 :
							#	print('***')
							#	print(i)
							#	print(j)
							#	print(did_add_this)
							#	input('--')
							did_add_yet=did_add_yet+did_add_this
					if False==did_add_yet and ((u!=v) or (len(self.node_list[u])>1)):
						print('not foudn')
						print('u,v')
						print([u,v])
						print('DF[u,v]')
						print(DF[u,v])
						print(self.node_list[u])
						print(self.node_list[v])
						input('[u,v]')
	def make_nodes(self):
		Nc=self.Nc
		MV=self.my_vrp
		t_init=MV.depot_start_time
		self.node_list=dict()
		self.node_list[Nc]=[tuple([Nc,t_init,t_init])]
		self.node_list[Nc+1]=[tuple([Nc+1,0,t_init])]
		for u in range(0,Nc):

			last_val=MV.late_start[u]-self.my_shift_bet_time_win
			self.node_list[u]=[]

			for this_time in self.time_thresh[u]:

				start_t=last_val+self.my_shift_bet_time_win
				end_t=this_time
				self.node_list[u].append(tuple([u,start_t,end_t]))
				last_val=end_t
			t_latest=MV.early_start[u]
			if last_val<t_latest:
				self.node_list[u].append(tuple([u,last_val+self.my_shift_bet_time_win,t_latest]))

			#print(MV.late_start[u],MV.early_start[u])
			#print(self.node_list[u])
			#input('hiya')

	def __init__(self,my_vrp,time_thresh):
		#dem_thresh has the intermediate thresholds to be used
		self.my_vrp=my_vrp
		self.my_shift_bet_time_win=self.my_vrp.my_params['my_shift_bet_time_win']

		self.Nc=my_vrp.num_cust
		self.time_thresh=time_thresh
		self.make_nodes()
		self.NEW_make_edges()
		debug_on=False
		if debug_on==True:
			self.make_edges()
			self.debug_do_check_agree()
		#print('in here')
		#print('--')
		#input('---')

	def debug_do_check_agree(self):

		E1=set(self.E_orig_backup)
		E2=set(self.E_novel_back)
		for e1 in self.E_orig_backup:
			if e1 not in self.E_novel_back:
				print('e1')
				print(e1)
				input('errror here')
		
		for e2 in self.E_novel_back:
			if e2 not in self.E_orig_backup:
				print('e2')
				print(e2)
				input('errror here 2')


	def NEW_make_edges_helper(self,u,v):
		MV=self.my_vrp

		my_dist=MV.dist_mat_full[u,v]
		Nu_i=len(self.node_list[u])
		Nv_j=len(self.node_list[v])
		self.DEBUG_edges_uv_tup[tuple([u,v])]=[]
		my_j_pos=Nv_j-1
		for i in range(Nu_i-1,-1,-1):
			#print('self.node_2_early[u]')
			#print(self.node_2_early[u])
			earliest_arrival_from_i_to_v=self.node_2_early[u][i]-my_dist
			earliest_arrival_from_i_CHILD_to_v=-np.inf
			if i>0:
				earliest_arrival_from_i_CHILD_to_v=self.node_2_early[u][i-1]-my_dist
			#print('trying' )
			#print("u = "+str(u))
			#print("i = "+str(i))
			#print("earliest_arrival_from_i_to_v = "+str(earliest_arrival_from_i_to_v))
			#print("earliest_arrival_from_i_CHILD_to_v = "+str(earliest_arrival_from_i_CHILD_to_v))
			for j in range(my_j_pos,-1,-1):
				latest_arrival_allowed=self.node_2_late[v][j]
				flag_cond=0
				#if u==0 and v==5 and self.node_2_early[u][i]==629 and self.node_2_late[v][j]==534:
				#	flag_cond=1
				#if flag_cond==1:
				#	print('earliest_arrival_from_i_to_v')
				#	print(earliest_arrival_from_i_to_v)
				#	print('latest_arrival_allowed')
				#	print('earliest_arrival_from_i_CHILD_to_v')
				#	print(earliest_arrival_from_i_CHILD_to_v)
				#	input('---')
				if earliest_arrival_from_i_to_v>= latest_arrival_allowed and latest_arrival_allowed>earliest_arrival_from_i_CHILD_to_v:#-.0001:
					this_tup=tuple([self.node_list[u][i],self.node_list[v][j]])
					self.E_novel_back.append(this_tup)
					self.E.append(this_tup)
					self.DEBUG_edges_uv_tup[tuple([u,v])].append(this_tup)

					my_j_pos=j-1
					#print('this_tup')
					#print(this_tup)
					#p#rint('my_dist')
					#print(my_dist)
					#input('---')
					break
			#if u==0 and 250>earliest_arrival_from_i_to_v:
			#	input('---')

	def NEW_make_edges(self):
		MV=self.my_vrp
		DF=MV.dist_mat_full
		self.DEBUG_edges_uv_tup=dict()
		Nc=self.Nc
		self.E_novel_back=[]
		self.E=[]


		self.node_2_early=dict()
		self.node_2_late=dict()

		for u in range(0,Nc+2):
			self.node_2_late[u]=[]
			self.node_2_early[u]=[]
			for i in range(0,len(self.node_list[u])):
				ti_minus=self.node_list[u][i][1]
				ti_plus=self.node_list[u][i][2]
				self.node_2_late[u].append(ti_minus)
				self.node_2_early[u].append(ti_plus)
		
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
