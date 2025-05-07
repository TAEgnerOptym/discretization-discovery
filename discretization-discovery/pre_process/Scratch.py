


def NEW_make_edges_helper(self,u,v):

	my_dist=MV.dist_mat_full[u,v]
	Nu_i=len(self.node_list[u])
	Nv_j=len(self.node_list[v])
	
	my_j_pos=Nv_j-1
	for i in range(Nu_i-1,-1,-1):
		earliest_arrival_from_i_to_v=self.node_2_early[u][i]-my_dist
		earliest_arrival_from_i_CHILD_to_v=-np.inf
		if i>0:
			earliest_arrival_from_i_CHILD_to_v=self.node_2_early[u][i]-my_dist
		for j in range(my_j_pos,-1,-1):
			latest_arrival_allowed=self.late_sorted[j]
			if earliest_arrival_from_i_to_v> latest_arrival_allowed and latest_arrival_allowed>earliest_arrival_from_i_CHILD_to_v:
				this_tup=tuple([self.node_list[u][i],self.node_list[v][j]])
				self.E.append(this_tup)

def NEW_make_edges(self):
	MV=self.my_vrp
	DF=MV.dist_mat_full
	Nc=self.Nc
	self.E=[]


	self.node_2_early=dict()
	self.node_2_late=dict()
	self.early_sorted=[]
	self.late_sorted=[]
	for u in range(0,Nc+1):
		self.node_2_late[u]=dict()
		self.node_2_early[u]=dict()
		for i in range(0,len(self.node_list[u])):
			ti_minus=self.node_list[u][i][1]
			ti_plus=self.node_list[u][i][2]
			self.early_sorted.append(ti_minus)
			self.late_sorted.append(ti_plus)
	
	for u in range(0,Nc):
		Nu_i=len(self.node_list[u])
		for i in range(Nu_i-1,-1,-1):
			this_tup=tuple([self.node_list[u][i],self.node_list[u][i-1]])
			self.E.append(this_tup)
	for u in range(0,Nc+1):
		for v in range(0,Nc+2):
			if DF[u,v]<np.inf and u!=v:
				self.NEW_make_edges_helper(u,v)
