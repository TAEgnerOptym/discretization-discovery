
class novel_ng_graph_basic:


    def generate_subsets_to_consider(self):
        #generate all subsets 
        my_subsets=set([])
        for u in range(0,self.NC):
           


            tmp=self.u_2_NG[u]+[u]
            my_power_set=power_set(tmp)
            for p in my_power_set:
                if len(p)<=self.max_SRI_SET_SIZE:
                    p=sorted(p)
                    tmp=frozenset(p)
                    my_subsets.add(tmp)
        self.subset_2_make_SRI=my_subsets


    def  generate_SRI(self):
        my_SRI=[]
        max_SRI_size=self.max_SRI_SET_SIZE
        print('subset_2_make_SRI')
        print(self.subset_2_make_SRI)
        input('---')
        for p in self.subset_2_make_SRI:
            max_sz=np.ceil(len(p)/2)
            max_sz=np.min([max_sz,max_SRI_size])
            max_sz=int(max_sz)
            for k in range(2,max_sz+1):
                if np.remainder(len(p),k)!=0:
                    new_SRI=dict()
                    new_SRI['customers']=p
                    new_SRI['my_divisor']=k
                    new_SRI['my_RHS']=np.floor(len(p)/k)
                    print('new_SRI')
                    print(new_SRI)
                    my_SRI.append(new_SRI)
        self.my_SRI=my_SRI
        print('my_SRI')
        print(my_SRI)
        input('my_SRI')


    def __init__(self,u_2_NG,my_VRP,max_SRI_Divisor,max_SRI_SET_SIZE):
        self.my_VRP=my_VRP
        self.u_2_NG=u_2_NG
        self.NC=len(self.u_2_NG)
        self.max_SRI_Divisor=max_SRI_Divisor
        self.max_SRI_SET_SIZE=max_SRI_SET_SIZE
        self.source_node=tuple([self.NC,set([])])
        self.sink_node=tuple([self.NC+1,set([])])
        self.generate_nodes()
        self.generate_edges()
        self.generate_subsets_to_consider()
        self.generate_SRI()
        self.generate_edge_2_SRI_contrib()
        self.non_source_sink_cust=np.arange(0,self.NC)#set(u_2_NG.keys())-set([self.my_VRP.NC,self.my_VRP.NC+1])

    def generate_edge_2_SRI_contrib(self):
        self.dict_valid_ineq_name_2_rhs = {}
        self.dict_valid_ineq_name_edge_2_coeff = {}

        E_2_lost_terms = self.E_2_lost_terms
        my_SRI = self.my_SRI

        count=0
        for q in my_SRI:
            count=count+1
            #print([count,len(my_SRI)])
            Nhat = set(q['customers'])
            k = q['my_divisor']
            rhs = q['my_RHS']
            k_inv = 1.0 / k  # precompute
            q_name = f"{frozenset(Nhat)}_{k}_{rhs}"

            # RHS of valid inequality
            self.dict_valid_ineq_name_2_rhs[q_name] = -np.floor(len(Nhat) * k_inv)

            # Build edge contribution dict
            tmp_dict = {
                e: -np.floor(len(terms & Nhat) * k_inv)
                for e, terms in E_2_lost_terms.items()
                if len(terms & Nhat) >= k  # early filter if no contribution
            }

            # Keep only nonzero entries
            print('tmp_dict')
            print(tmp_dict)
            input('---')
            tmp_dict = {e: coeff for e, coeff in tmp_dict.items() if coeff < 0}
            self.dict_valid_ineq_name_edge_2_coeff[q_name] = tmp_dict


    def OLD_generate_edge_2_SRI_contrib(self):

        self.dict_valid_ineq_name_2_rhs=dict()
        self.dict_valid_ineq_name_edge_2_coeff=dict()
        my_SRI=self.my_SRI
        for q in my_SRI:
            Nhat=set(q['customers'])
            k=q['my_divisor']
            rhs=q['my_RHS']
            q_name=str(Nhat)+'_'+str(k)+'_'+str(rhs)
            self.dict_valid_ineq_name_2_rhs[q_name]=-np.floor(len(Nhat)/k)
            tmp_dict=dict()
            for e in self.E_2_lost_terms:
                my_lost_terms=self.E_2_lost_terms[e]
                this_inter=my_lost_terms.intersection(Nhat)
                coeff=np.floor(len(this_inter)/k)
                if coeff>0:
                    tmp_dict[e]=-coeff
                    
            self.dict_valid_ineq_name_edge_2_coeff[q_name]=tmp_dict

    def generate_nodes(self):
        self.nodes=[]
        VRP=self.my_VRP
        NC=self.NC

        sink_node=tuple([NC+1,set([])])
        source_node=tuple([NC,set([])])
        self.nodes.append(sink_node)
        self.nodes.append(source_node)
        self.non_source_sink_nodes=[]
        self.non_source_sink_cust=np.arange(0,NC)
        #self.node_2_ng_allowed
        for u in self.non_source_sink_cust:

            my_power_set=power_set(self.u_2_NG[u])
            for my_sub in my_power_set:
                my_sub=set(sorted(list(my_sub)))
                my_new_node=tuple([u,my_sub])
                self.nodes.append(my_new_node)
                self.non_source_sink_nodes.append(my_new_node)
    def generate_edges(self):

        VRP=self.my_VRP
        Dist=VRP.dist_mat_full
        self.E_2_uv=dict()
        self.uv_2_E=dict()
        self.E_2_lost_terms=dict()
        self.E=[]
        NC=self.NC
        for u in range(0,NC+1):
            for v in range(0,NC+2):
                if Dist[u,v]<np.inf:
                    self.uv_2_E[tuple([u,v])]=[]
        self.SET_u_2_ng_set=dict()
        self.SET_u_2_ng_set[NC]=set([])
        self.SET_u_2_ng_set[NC+1]=set([])

        for u in range(0,NC):
            self.SET_u_2_ng_set[u]=set(self.u_2_NG[u])

        for n in self.nodes:
            #print('n')
            #print(n)
            u=n[0]
            N_n=n[1]
            nodes_plus_u=set(N_n).union(set([u]))
            for w in range(0,NC+2):
                if u!=w and w!=NC and Dist[u,w]<np.inf and w not in N_n :
                    self.make_new_edge(n,w,nodes_plus_u)

    def make_new_edge(self, i, w,orig_terms):

        i0=i[0]

        if w < self.my_VRP.num_cust:
            this_ng_set = self.SET_u_2_ng_set[w]
            new_set = orig_terms & this_ng_set
        else:
            new_set = set([])

        lost_terms = orig_terms - new_set
        new_set = set(sorted(new_set))

        j = (w, new_set)
        e = (i, j)
        if j not in self.nodes:
            print('j')
            print(j)
            print('BIG ERROR')
            input('BIG ERROR')
        uv = (i0, w)

        self.E.append(e)
        self.uv_2_E[uv].append(e)
        e_key = str(e)
        self.E_2_uv[e_key] = uv
        self.E_2_lost_terms[e_key] = lost_terms

