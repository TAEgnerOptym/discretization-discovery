    def get_agglomerative_dictionary(self,X_all,h):
        
        num_cust=max(self.graphNameNode_2_cust[h].values())-2
        out_member_2_cluster_id=dict()
        out_cluster_id_2_member=dict()
        out_cluster_id_2_member_all_all=dict()
        out_cluster_id_2_member_all=dict()
        Y=self.graphNameNode_2_cust[h]
        cluster_count=0
        for u in range(0,num_cust+2):
            out_member_2_cluster_id[u]=dict()
            out_cluster_id_2_member[u]=dict()
            X = {k: v for k, v in X_all.items() if Y.get(k) == u}
            keys = list(X.keys())
            n = len(keys)

            # Create a 2-D array of scalar values (each value is a 1-D point).
            data = np.array([X[k] for k in keys]).reshape(-1, 1)

            if n == 1:
                # Create the single node clustering information.
                cluster_id_2_member = {cluster_count: [keys[0]]}
                member_2_cluster_id = {keys[0]: [cluster_count]}
                
                # The output dictionaries use tuples (h, cluster_id)
                out_cluster_id_2_member[u] = { (h, cluster_count): [keys[cluster_count]] }
                out_member_2_cluster_id[u] = { keys[cluster_count]: [(h, cluster_count)] }
                continue

            # Compute the linkage matrix using SciPy's hierarchical clustering.
            Z = linkage(data, method="ward")
            
            cluster_id_2_member = {}
            for i in range(n):
                cluster_id_2_member[i] = [keys[i]]
                
        
            # Process each merge from the linkage matrix.
            # Each row in Z is of the form: [idx1, idx2, distance, sample_count]
            # New cluster ids are assigned as n, n+1, ... (as is standard in SciPy)
            for i, row in enumerate(Z):
                idx1, idx2, distance, count = row
                idx1, idx2 = int(idx1), int(idx2)
                new_cluster_id = n + i+cluster_count  # new cluster id for this merge
                # The new cluster contains all leaves from the clusters idx1 and idx2.
                cluster_id_2_member[new_cluster_id] = cluster_id_2_member[idx1] + cluster_id_2_member[idx2]
            
            
            member_2_cluster_id=dict()
            for my_node in X:
                member_2_cluster_id[my_node]=[]
            for cluster_id in cluster_id_2_member:
                for q in cluster_id_2_member[cluster_id]:
                    member_2_cluster_id[q].append(cluster_id)
            
            out_cluster_id_2_member=dict()
            for i in cluster_id_2_member:
                i_out=tuple([h,i])
                out_cluster_id_2_member[i_out]=cluster_id_2_member[i]
            
            cluster_count=cluster_id_2_member.keys()
            out_member_2_cluster_id[u][h]=dict()
            for i in member_2_cluster_id:
                out_member_2_cluster_id[u][h][i]=[]
                out_cluster_id_2_member_all[h][i]=[]
                for j in member_2_cluster_id[i]:
                    out_member_2_cluster_id[u][h][i].append(tuple([h,j]))
                    out_cluster_id_2_member_all[h][i].append(tuple([h,j]))

            for u in range(0,num_cust)

        self.H_ell_2_list_leaf[h]=out_cluster_id_2_member_all
        self.H_leaf_2_list_ell[h]=out_member_2_cluster_id_all

