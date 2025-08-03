import numpy as np
import networkx as nx
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


def dag_all_pairs_shortest_paths(DM):
    n = DM.shape[0]
    G = nx.DiGraph()

    # Build graph from adjacency matrix
    for i in range(n):
        for j in range(n):
            if not np.isinf(DM[i, j]):
                G.add_edge(i, j, weight=DM[i, j])

    # Compute shortest path lengths
    all_pairs = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))

    # Initialize result matrix
    dist_matrix = np.full((n, n), np.inf)
    for i in range(n):
        for j, d in all_pairs.get(i, {}).items():
            dist_matrix[i, j] = d

    # Set diagonal to inf
    np.fill_diagonal(dist_matrix, np.inf)

    return dist_matrix

def naive_get_LA_neigh(my_vrp,num_la):

	Nc=my_vrp.num_cust
	DM1=my_vrp.dist_mat
	DM=dag_all_pairs_shortest_paths(DM1)
	#print('DM')
	#print(DM)
	#print('num_la')
	#print(num_la)
	#input('--')
	#DM=my_vrp.orig_dist_mat_full[:Nc,:Nc]
	LA_neigh_list=[[]]*Nc
	LA_neigh_list_unsorted=[[]]*Nc
	for u in range(0,Nc):
		
		this_ord=np.argsort(DM[u,:])
		#this_ord=np.argsort(DM[:,u])

		this_list=[]
		for i in range(0,num_la):
			if DM[u,this_ord[i]]<np.inf:
			#if DM[this_ord[i],u]<np.inf:
			
				this_list=this_list+[this_ord[i]]
			else:
				break
		#print('u')
		##print(u)
		#print('this_list')
		#print(this_list)
		#input('--')
		LA_neigh_list_unsorted[u]=this_list
		LA_neigh_list[u]=sorted(this_list)
		
	#print('DM')
	#print(DM)
	#for u in range(0,Nc):
	#	print('u. '+ str(u))
	#	print(LA_neigh_list[u])
	#input('hi')

	return LA_neigh_list,LA_neigh_list_unsorted

#if 1<0:
#			trm1=DM[u,:]#+DM[:,u]#.transpose
#			trm2=DM[:,u]
##		
#			this_ord=np.argsort(trm1+trm2)