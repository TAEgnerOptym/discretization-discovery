

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
				cond2=self.check_if_con_poss(di_minus-self.dem_step_sz,dj_minus,MV.dem_full[u])
			if j_has_parent:
				cond3=self.check_if_con_poss(di_plus,dj_plus+self.dem_step_sz,MV.dem_full[u])
		else:
			cond1=(di_minus-self.dem_step_sz==dj_plus)
		if cond1==True and cond2==False and cond3==False:
			this_tup=tuple([i,j])
			self.E.append(this_tup)			
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

			last_val=MV.dem[u]-self.dem_step_sz
			self.node_list[u]=[]
			for this_dem in self.dem_thresh[u]:
				start_d=last_val+self.dem_step_sz
				end_d=this_dem
				self.node_list[u].append(tuple([u,start_d,end_d]))
				last_val=end_d
			if last_val<d0:
				self.node_list[u].append(tuple([u,last_val+self.dem_step_sz,d0]))
			
	def __init__(self,my_vrp,dem_thresh):
		#dem_thresh has the intermediate thresholds to be used
		self.my_vrp=my_vrp
		self.Nc=my_vrp.num_cust
		self.dem_thresh=dem_thresh
		self.dem_step_sz=self.my_vrp.my_params['dem_step_sz']
		print('making dem nodes')
		self.make_nodes()
		print('making dem edges')

		self.make_edges()
		print('done dem edges')
