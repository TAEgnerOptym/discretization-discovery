import itertools
import numpy as np


class order_object:
    def __init__(self,my_order_name,pred_order,my_instance):
        self.my_order_name=my_order_name
        self.my_instance=my_instance

        self.pred_order=pred_order
        self.u=my_order_name[0]
        self.w=my_order_name[1]
        self.dist_serve_add=self.my_instance.dist_mat_full[self.u,self.w]

        self.compute_cost()#dict()
        self.compute_early_depart_wo_wait()
        self.compute_latest_depart()
        #if self.my_instance.my_params['DEBUG_NG_turn_off_CLEAN']==False:
        if self.lateDepart>self.my_instance.early_start[self.u]:
            self.cost=np.inf
                #if 1>0:
                #    self.pretty_print_order()

                    
               #     input('set something to infinite')
        #if len(my_order_name)==2 and my_order_name[-1]==22 and my_order_name[-2]==24:
        #    self.pretty_print_order()
        #    input('len 2')
        #if len(my_order_name)==3 and my_order_name[-1]==22 and my_order_name[-2]==24 and my_order_name[-3]==23:
        #    self.pretty_print_order()
        #    input('len3 ')
        
        #if len(my_order_name)==4 and my_order_name[-1]==22 and my_order_name[-2]==24 and my_order_name[-3]==23 and  my_order_name[-4]==19:
        #    self.pretty_print_order
        #    input('len 4')
    def extend_order(self,new_u):
        
        trm1=tuple([new_u])
        NEW_my_order_name=trm1+self.my_order_name
        my_new_order=order_object(NEW_my_order_name,self,self.my_instance)
        #print('extension')
        #my_new_order.pretty_print_order()
        #print('original')
        #self.pretty_print_order()
        #input('---')
        return my_new_order

    def pretty_print_order(self):
        print('pretty printint order')
        print('my_order_name')
        print(self.my_order_name)
        print('cost')
        print(self.cost)
        print('dist_serve_add')
        print(self.dist_serve_add)
        print('lateDepart')
        print(self.lateDepart)
        print('self.earlyArrical')
        print(self.earlyArrival)
        print('self.early_depart_wo_wait')
        print(self.early_depart_wo_wait)
        print('late_start_u')
        print(self.late_start_u)
        print('early_depart_prev')
        print(self.early_depart_prev)
        print('self.early_start_u')
        print(self.early_start_u)
        print('early_depart_prev')
        print(self.early_depart_prev)
        print('--- now the terms ---')
        DF=self.my_instance.dist_mat_full
        Q=self.my_order_name
        Early_start=self.my_instance.early_start
        Late_start=self.my_instance.late_start
        for i in range(0,len(self.my_order_name)):
            print('i '+str(i))
            print('u[i] '+str(Q[i]))
            if i<len(self.my_order_name)-1:
                print('dist '+str(DF[Q[i],Q[i+1]]))
            print('Early_start[Q[i]]'+str(Early_start[Q[i]]))
            print('Late_start[Q[i]]'+str(Late_start[Q[i]]))
    def compute_cost(self):
        self.cost=0#self.my_instance.dist_mat_full[u,v]
        u=self.u
        w=self.w

        if self.pred_order!=None:
            self.cost=self.pred_order.cost+self.dist_serve_add#self.my_instance.dist_mat_full[u,w]
        else:
            self.cost=self.dist_serve_add#self.my_instance.dist_mat_full[u,w]
            
    def compute_early_depart_wo_wait(self):
        early_depart_prev=self.my_instance.early_start[self.w]
        if self.pred_order!=None:
            early_depart_prev=self.dist_serve_add+self.pred_order.early_depart_wo_wait
        self.early_depart_prev=early_depart_prev
        trm1=self.dist_serve_add+early_depart_prev
        trm2=self.my_instance.early_start[self.u]
        self.early_start_u=trm2
        self.early_depart_wo_wait=min([trm1,trm2])
        self.earlyArrival=self.early_depart_wo_wait-self.dist_serve_add
        
    def compute_latest_depart(self):
        late_depart_prev=self.my_instance.late_start[self.w]
        if self.pred_order!=None:
            late_depart_prev=self.pred_order.lateDepart
        self.late_depart_prev=late_depart_prev
        trm1=self.my_instance.late_start[self.u]
        trm2=self.dist_serve_add+late_depart_prev
        self.lateDepart=max([trm1,trm2])
        self.late_start_u=trm1
class ng_graph_fancy_slow:
    
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
        u=my_node[0]
        N_i=my_node[1]
        N_i=set(N_i)
        for w in self.ng_neigh_by_cust[u]:
            if w in N_i:
                continue
            new_set=N_i.copy()
            new_set.add(w)
            new_set=set(sorted(list(new_set)))
            new_node=tuple([u,new_set])
            if new_node not in self.my_nodes[u]:
                print('self.node_list[u]')
                print(self.node_list[u])
                print('new_node')
                print(new_node)
                input('error here')
            new_edge=tuple([my_node,new_node])
            self.E.append(new_edge)
            


    def add_non_self_succ_slow(self,v,my_node):
        u=my_node[0]
        N_i=my_node[1]
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
        
        N_minus_nv=self.non_neigh_cust[v]
        N_u_minus_Ni=self.node_2_ng_allowed[str(my_node)]
        new_edge=tuple([my_node,new_node])

        if len(N_minus_nv.intersection(N_u_minus_Ni))==0:
            self.E.append(new_edge)
        

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
                

    def __init__(self,my_instance,ng_neigh_by_cust):
        self.my_instance=my_instance
        self.ng_neigh_by_cust=ng_neigh_by_cust
        self.Nc=len(self.ng_neigh_by_cust)
        self.my_instance.my_params['DEBUG_NG_turn_off_CLEAN']=False
        self.make_nodes()
        self.make_edges_slow()

        self.LAish_remove_edges_slow()
        self.clean_order()

    def LA_ish_update_efficient(self,my_tup):
        u=my_tup[0]
        Nhat=my_tup[1]
        v=my_tup[2]
        list_my_tup_2_used=[]
        flag_stuff=False
        if 0>1 and u==19 and v==22 and 24 in Nhat and 23 in Nhat:
            flag_stuff=True 
           
            input('-starting the flag for this-')
        if len(Nhat)==0:
            if self.my_instance.dist_mat_full[u,v]<np.inf:
                base_case_order=order_object(tuple([u,v]),None,self.my_instance)
                self.eff_front_tup_2_list[my_tup]=set([base_case_order])
            else: 
                self.eff_front_tup_2_list[my_tup]=[]
            return  
        all_possible_orders_for_tuple=set()

        
        #if self.my_instance.my_params['DEBUG_NG_turn_off_CLEAN']==False:
        if 1>0:
            tot_cap=self.my_instance.vehicle_capacity
            tot_cap=tot_cap-self.my_instance.dem_full[u]
            tot_cap=tot_cap-self.my_instance.dem_full[v]
            for w in set(Nhat).intersection(self.ng_neigh_by_cust[v]):
                tot_cap=tot_cap-self.my_instance.dem_full[w]
            if tot_cap<-0.001:
                #print('tot_cap')
                #print(tot_cap)
                #input('awesoem killed one')
                self.eff_front_tup_2_list[my_tup]=[]
                return
        for w in Nhat:
            u2=w
            NHat2=Nhat-set([w])
            NHat2=frozenset(NHat2)
            v2=v
            my_tup_2=tuple([u2,NHat2,v2])
            if my_tup_2 not in self.eff_front_tup_2_list:
                self.LA_ish_update_efficient(my_tup_2)
                #if len(self.eff_front_tup_2_list[my_tup_2])==0:
                ##    print('my_tup_2')
                #    input('foudn an inner empty')
            list_my_tup_2_used.append(my_tup_2)
            if flag_stuff>0.5:
                print(my_tup_2)
                print('len(self.eff_front_tup_2_list[my_tup_2])')
                print(len(self.eff_front_tup_2_list[my_tup_2]))
            for p_pred in  self.eff_front_tup_2_list[my_tup_2]:
                new_candid_name=tuple([u])+p_pred.my_order_name
                if new_candid_name not in self.dict_ord_name_2_ord:
                    new_candid=p_pred.extend_order(u)
                    self.dict_ord_name_2_ord[new_candid_name]=new_candid
                this_candid=self.dict_ord_name_2_ord[new_candid_name]
                if flag_stuff>0.5:
                    this_candid.pretty_print_order()
                if this_candid.cost<np.inf:
                    all_possible_orders_for_tuple.add(this_candid)
        #do_remove=set()
        #for ord1 in all_possible_orders_for_tuple:
         #   for ord2 in all_possible_orders_for_tuple:
         #       if ord2.does_dominate_input(ord1)==True:
         #           do_remove.add
        len_before=len(all_possible_orders_for_tuple)
        #print('Just Generated ')
        #print(my_tup)
        #print('lenBefore')
        #print(len(all_possible_orders_for_tuple))
        all_possible_orders_for_tuple_out=self.compute_efficient_frontier(all_possible_orders_for_tuple)#all_possible_orders_for_tuple-do_remove
        len_after=len(all_possible_orders_for_tuple_out)
        if 0>1 and len_after<len_before:
            print('len_after,len_before')
            print([len_after,len_before])
            for k in all_possible_orders_for_tuple:
                print('IS FOUDN')
                print (k in all_possible_orders_for_tuple_out)
                print('---')
                k.pretty_print_order()
            input('AWESOME')
        #if self.my_instance.my_params['DEBUG_NG_turn_off_CLEAN']==False:

        self.eff_front_tup_2_list[my_tup]=all_possible_orders_for_tuple_out
        if flag_stuff>0.5:
            print('all_possible_orders_for_tuple_out')
            print(all_possible_orders_for_tuple_out)
            input('done  with te falgesd')
        #else:
        #    self.eff_front_tup_2_list[my_tup]=all_possible_orders_for_tuple
    def compute_efficient_frontier(self,objects):
    # Sort by cost ascending, then earlyArrival descending, then lateDepart ascending
        objects_sorted = sorted(objects, key=lambda x: (x.cost, -x.earlyArrival, x.lateDepart))

        frontier = []
        best_early = float('-inf')
        best_late = float('inf')

        for obj in objects_sorted:
            # Only keep obj if it's not dominated by previous ones
            if obj.earlyArrival > best_early or obj.lateDepart < best_late:
                frontier.append(obj)
                best_early = max(best_early, obj.earlyArrival)
                best_late = min(best_late, obj.lateDepart)

        return frontier


    def LAish_remove_edges_slow(self):
        num_kill=0
        num_spared=0
        new_E=[]
        self.eff_front_tup_2_list=dict()
        self.dict_ord_name_2_ord=dict()
       

        for e in self.E:
            u=e[0][0]
            v=e[1][0]
            N_i=e[0][1]
            N_j=e[1][1]
            if u==v or u==self.Nc or v== self.Nc+1 :
                #or  True==self.LAish_check_valid_edge(e):
                new_E.append(e)
                continue
            N_i=set(e[0][1])
            N_j=set(e[1][1])
            if len(N_j)==0:
                new_E.append(e)
                continue
            if len(N_j)==1 and u in N_j:
                new_E.append(e)
                continue
            poss_first=N_j-set([u])
            #print('poss_first')
            #print(poss_first)
            did_add_ever=False
            my_tups_used=[]
            for w in poss_first:
                N_q=set([u]).union(N_j)-set([w])
                N_q=frozenset(N_q)
                this_tup=tuple([w,N_q,v])
                if this_tup not in self.eff_front_tup_2_list:
                    self.LA_ish_update_efficient(this_tup)
                    #if len(self.eff_front_tup_2_list[this_tup])==0:
                    #    print('this_tup')
                    #    print(this_tup)
                    #    input('ok i found an empty')
                my_tups_used.append(this_tup)
                list_of_orders=self.eff_front_tup_2_list[this_tup]
                for my_order in list_of_orders:
                    if my_order.my_order_name[-2]==u:
                        did_add_ever=True
                        break
                if did_add_ever==True:
                    break
            #u_break=set(np.arange(24,25))
            if   did_add_ever==False:# and u==24 and v==22 and len(N_i)==2 and len(N_j)==2 and 19 in N_i and 23 in N_j:
                if 1<0:
                    print('my_tups_used')
                    print(my_tups_used)
                    for this_tup in my_tups_used:
                        list_of_orders=self.eff_front_tup_2_list[this_tup]
                        print('this_tup')
                        print(this_tup)
                        print('len(list_of_orders)')
                        print(list_of_orders)
                        for my_order in list_of_orders:
                            print(my_order.my_order_name)
                print('Killing e: u= '+str(e[0][0])+'  v=  '+str(e[1][0])+'  e='+str(e))
                num_kill=num_kill+1
            else: #did_add_ever==True:
                new_E.append(e)
                num_spared=num_spared+1
        print('len(new_E)')
        print(len(new_E))
        print('len(self.E)')
        print(len(self.E))
        print('num_spared,num_kill')
        print([num_spared,num_kill])
        self.E=new_E
        #input('all edges removed are avboce')
    def clean_order(self):
        self.node_list=dict()

        for u in self.my_nodes:
            new_nodes=[]
            for n in self.my_nodes[u]:
                tmp=sorted(list(n[1]))
                this_node=[n[0],tmp]
                new_nodes.append(this_node)
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
            this_new_edge=tuple([this_node_1,this_node_2])
            E_2.append(this_new_edge)
        self.E=E_2
