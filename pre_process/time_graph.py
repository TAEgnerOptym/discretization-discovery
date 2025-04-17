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
		self.arrival_time=dict()
		self.departure_time=dict()
		for u in range(0,Nc+1):
			for v in range(0,Nc+2):
				if DF[u,v]<np.inf or ( u==v and u!=Nc ):		

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
		self.make_edges()