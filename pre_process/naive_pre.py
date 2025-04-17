import numpy as np
def naive_get_dem_thresh_list(my_vrp,thresh_jmp):
	Nc=my_vrp.num_cust
	d0=my_vrp.vehicle_capacity
	dem_thresh_list=[];
	for u in range(0,Nc):
		my_list=[]
		for d_end in range(int(my_vrp.dem_full[u]),int(d0)+1,int(thresh_jmp)):
			my_list.append(d_end)
		dem_thresh_list.append(my_list)

	return dem_thresh_list

def naive_get_time_thresh_list(my_vrp,thresh_jmp):

	Nc=my_vrp.num_cust
	d0=my_vrp.vehicle_capacity
	time_thresh_list=[];
	for u in range(0,Nc):
		my_list=[]
		t_min=my_vrp.late_start[u]
		t_max=my_vrp.early_start[u]
		for t_end in range(int(t_min+thresh_jmp),int(t_max),int(thresh_jmp)):
			my_list.append(t_end)
		time_thresh_list.append(my_list)

	return time_thresh_list
def naive_get_LA_neigh(my_vrp,num_la):

	Nc=my_vrp.num_cust
	DM=my_vrp.dist_mat
	LA_neigh_list=[[]]*Nc
	LA_neigh_list_unsorted=[[]]*Nc
	for u in range(0,Nc):
		this_ord=np.argsort(DM[u,:])

		this_list=[];
		for i in range(0,num_la):
			if DM[u,this_ord[i]]<np.inf:
				this_list=this_list+[this_ord[i]]
			else:
				break
		LA_neigh_list_unsorted[u]=this_list
		LA_neigh_list[u]=sorted(this_list)
	return LA_neigh_list,LA_neigh_list_unsorted