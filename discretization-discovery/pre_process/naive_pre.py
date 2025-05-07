import numpy as np
def naive_get_dem_thresh_list(my_vrp,thresh_jmp):
	Nc=my_vrp.num_cust
	d0=my_vrp.vehicle_capacity
	dem_thresh_list=[];
	for u in range(0,Nc):
		my_list=[]
		#for d_end in range(int(my_vrp.dem_full[u]),int(d0)+1,int(thresh_jmp)):
		#print('thresh_jmp')
		#print(thresh_jmp)
		#print('d0')
		#print(d0)
		#print('my_vrp.dem_full[u]')
		#print(my_vrp.dem_full[u])
		for d_end in np.arange(my_vrp.dem_full[u], d0 + thresh_jmp, thresh_jmp):
			my_list.append(round(d_end,5))
		dem_thresh_list.append(my_list)
		#print('dem_thresh_list')
		#print(dem_thresh_list)
		#print('thresh_jmp')
		#print(thresh_jmp)
		#input('--')
	return dem_thresh_list

def naive_get_time_thresh_list(my_vrp,thresh_jmp):

	Nc=my_vrp.num_cust
	d0=my_vrp.vehicle_capacity
	time_thresh_list=[];
	for u in range(0,Nc):
		my_list=[]
		t_min=my_vrp.late_start[u]
		t_max=my_vrp.early_start[u]
		num_steps=int((t_max-t_min)/thresh_jmp)
		for ti in range(1,num_steps+1):
			t_end=t_min+(ti*thresh_jmp)
			my_list.append(t_end)
		time_thresh_list.append(my_list)

	return time_thresh_list

def old_naive_get_time_thresh_list(my_vrp,thresh_jmp):

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
	#DM=my_vrp.orig_dist_mat_full[:Nc,:Nc]
	LA_neigh_list=[[]]*Nc
	LA_neigh_list_unsorted=[[]]*Nc
	for u in range(0,Nc):
		
		this_ord=np.argsort(DM[u,:])
		#this_ord=np.argsort(DM[:,u])

		this_list=[];
		for i in range(0,num_la):
			if DM[u,this_ord[i]]<np.inf:
			#if DM[this_ord[i],u]<np.inf:
			
				this_list=this_list+[this_ord[i]]
			else:
				break
		LA_neigh_list_unsorted[u]=this_list
		LA_neigh_list[u]=sorted(this_list)
	return LA_neigh_list,LA_neigh_list_unsorted

#if 1<0:
#			trm1=DM[u,:]#+DM[:,u]#.transpose
#			trm2=DM[:,u]
##		
#			this_ord=np.argsort(trm1+trm2)