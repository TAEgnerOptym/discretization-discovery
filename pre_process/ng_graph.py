import itertools
import numpy as np

class ng_graph:
    
    def power_set(self,s):
        """
        Returns the power set of the input collection `s` as a list of tuples.
        The power set is defined as all possible subsets of `s`.
        
        Example:
            power_set({1, 2}) returns [(), (1,), (2,), (1, 2)]
        """
        s_list = list(s)  # Ensure the input is ordered
        return list(itertools.chain.from_iterable(
            itertools.combinations(s_list, r) for r in range(len(s_list) + 1)
        ))

    def make_edges_slow(self):
        self.E=[]
        for u in range(0,self.Nc+2):
            for v in range(0,self.Nc+2):
                if u!=v and np.inf==self.my_instance.dist_mat_full[u,v]:
                    continue
                for my_node in self.my_nodes[u]:
                    if u==v and u<self.Nc:
                        self.add_sucessors_slow(my_node)
                    if u!=v and np.inf>self.my_instance.dist_mat_full[u,v]:
                        self.add_non_self_succ_slow(v,my_node)

    def add_sucessors_slow(self,my_node):
        #print('my_node')
        #print(my_node)
        u=my_node[0]
        N_i=my_node[1]
        N_i=set(N_i)
        for w in self.ng_neigh_by_cust[u]:
            if w in N_i:
                continue
            #print(N_i)
            #print('type(N_i)')
            #print(type(N_i))
            #print('type(N_i)')
            new_set=N_i.copy()
            new_set.add(w)
            #print('w')
            #print(w)
            #print('new_set')
            #print(new_set)
            new_set=set(sorted(list(new_set)))
            new_node=tuple([u,new_set])
            #if u==13:
            #    print('new_node')
            #    print(new_node)
            ##    print('N_i')
            #    print(N_i)
                #input('---')
            if new_node not in self.my_nodes[u]:
                print('self.node_list[u]')
                print(self.node_list[u])
                print('new_node')
                print(new_node)
                input('error here')
            new_edge=tuple([my_node,new_node])
            #print('new_edge')
            #print(new_edge)
            #input('--')
            self.E.append(new_edge)
            


    def add_non_self_succ_slow(self,v,my_node):
        u=my_node[0]
        N_i=my_node[1]
        #print('N_i')
        #print(N_i)
        #print('type(N_i')
        #print(type(N_i))
        if v ==u:
            input('error here')
        if v in N_i:
            return
        N_j=list(N_i)
        N_j.append(u)
        N_j=sorted(N_j)
        N_j=set(N_j)
        N_j=N_j.intersection(self.ng_neigh_by_cust[v])
        N_j=list(N_j)
        N_j=sorted(N_j)
        N_j=set(N_j)
        new_node=tuple([v,N_j])
        
        #print()
        #if len(N_i.intersection(N_j))==0:
        N_minus_nv=self.non_neigh_cust[v]
        N_u_minus_Ni=self.node_2_ng_allowed[str(my_node)]
        did_add=False
        new_edge=tuple([my_node,new_node])

        if len(N_minus_nv.intersection(N_u_minus_Ni))==0:
            self.E.append(new_edge)
            did_add=True
        debug=False
        if  debug==True:
            print('N_minus_nv')
            print(N_minus_nv)
            print('N_u_minus_Ni')
            print(N_u_minus_Ni)
            print('u:   '+str(u))
            print('v:   '+str(v))
            print('self.ng_neigh_by_cust[u]')
            print(self.ng_neigh_by_cust[u])
            print('self.ng_neigh_by_cust[v]')
            print(self.ng_neigh_by_cust[v])
            print('did_add')
            print(did_add)
            print('new_edge')
            print(new_edge)
            input('--')
            
        


        #print('hi')

    def make_nodes(self):
        self.my_nodes=[]
        self.my_nodes=dict()
        self.source_node=tuple([self.Nc,set([])])
        self.sink_node=tuple([self.Nc+1,set([])])
        self.my_nodes[self.Nc]=[self.source_node]
        self.my_nodes[self.Nc+1]=[self.sink_node]
        self.ng_neigh_by_cust.append(set([]))
        self.ng_neigh_by_cust.append(set([]))
        self.my_power_set=dict()
        self.non_neigh_cust=dict()
        self.non_neigh_cust[self.Nc]=set(np.arange(0,self.Nc))
        self.non_neigh_cust[self.Nc+1]=set(np.arange(0,self.Nc))
        self.node_2_ng_allowed=dict()
        self.node_2_ng_allowed[str(self.source_node)]=set([])
        self.node_2_ng_allowed[str(self.sink_node)]=set([])
        for u in range(0,self.Nc):
            my_set=set(np.arange(0,self.Nc))
            my_set.remove(u)
            for w in self.ng_neigh_by_cust[u]:
                my_set.remove(w)
            self.non_neigh_cust[u]=my_set
        for u in range(0,self.Nc):
            self.my_power_set[u]=self.power_set(self.ng_neigh_by_cust[u])
            self.my_nodes[u]=[]
            for my_sub in self.my_power_set[u]:
                my_sub=set(sorted(list(my_sub)))
                my_new_node=tuple([u,my_sub])
                self.my_nodes[u].append(my_new_node)
                self.node_2_ng_allowed[str(my_new_node)]=set(self.ng_neigh_by_cust[u])-set(my_sub)
                #print(my_new_node)
                ##print('self.ng_neigh_by_cust[u]')
                #print(self.ng_neigh_by_cust[u])
                #print('self.node_2_ng_allowed[str(my_new_node)]')
                #print(self.node_2_ng_allowed[str(my_new_node)])
                #input('--')

    def __init__(self,my_instance,ng_neigh_by_cust):
        self.my_instance=my_instance
        self.ng_neigh_by_cust=ng_neigh_by_cust
        self.Nc=len(self.ng_neigh_by_cust)

        self.make_nodes()
        
        #print('self.my_nodes[13]')
        ##print(self.my_nodes[13])
        #input('--')
        self.make_edges_slow()

        self.clean_order()
    def clean_order(self):
        self.node_list=dict()

        for u in self.my_nodes:
            new_nodes=[]
            for n in self.my_nodes[u]:
                tmp=sorted(list(n[1]))
                this_node=[n[0],tmp]
                new_nodes.append(this_node)

                #this_node_str=str(this_node)
                #this_node_str=this_node_str.replace(' ','_')
                #new_nodes.append(this_node_str)
            self.node_list[u]=new_nodes
        E_2=[]
        for e in self.E:
            i=e[0]
            j=e[1]
            tmp1=sorted(list(i[1]))
            this_node_1=[i[0],tmp1]
            tmp2=sorted(list(j[1]))
            this_node_2=[j[0],tmp2]
            
            this_node_1_str=str(this_node_1)
            this_node_2_str=str(this_node_2)
            this_node_1_str=this_node_1_str.replace(' ','_')
            this_node_2_str=this_node_2_str.replace(' ','_')
            #if this_node_1_str not in self.node_list[i[0]]:
            #    input('error 1')
            #if this_node_2_str not in self.node_list[j[0]]:
            #    input('error 2')
            #print('this_node_1_str')
            #print(this_node_1_str)
            #print('this_node_2_str')
            #print(this_node_2_str)
            #input('--')
            #this_new_edge=tuple([this_node_1_str,this_node_2_str])
            this_new_edge=tuple([this_node_1,this_node_2])
            E_2.append(this_new_edge)
        self.E=E_2
        #print(self.E)
        #input('---')
