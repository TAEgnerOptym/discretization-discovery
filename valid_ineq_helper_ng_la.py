

import numpy as np
# Set desired solver options
import itertools
import networkx as nx

def power_set(s):
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



class NEW_order_object:
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
        if self.lateDepart>self.my_instance.early_start_full[self.u]:
            self.cost=np.inf

        self.dem_in_arc=0
        for u in self.my_order_name:
            self.dem_in_arc+=self.my_instance.dem_full[u]
        if self.dem_in_arc>self.my_instance.vehicle_capacity:
            self.cost=np.inf
       #if self.u==self.my_instance.num_cust and self.w==self.my_instance.num_cust+1:
       #     print('self.cost')
       #     print(self.cost)
       #     input('my cost')
    def extend_order(self,new_u):
        
        trm1=[new_u]
        #print('self.my_order_name')
        #print(self.my_order_name)
        #print('trm1')
        #print(trm1)
        NEW_my_order_name=trm1+self.my_order_name
        my_new_order=NEW_order_object(NEW_my_order_name,self,self.my_instance)

        return my_new_order

    def compute_cost(self):
        self.cost=0#self.my_instance.dist_mat_full[u,v]
        u=self.u
        w=self.w

        if self.pred_order!=None:
            self.cost=self.pred_order.cost+self.dist_serve_add#self.my_instance.dist_mat_full[u,w]
        else:
            self.cost=self.dist_serve_add#self.my_instance.dist_mat_full[u,w]
            
    def compute_early_depart_wo_wait(self):
        
        early_depart_prev=np.inf#self.my_instance.early_start[self.w]
        
        if self.pred_order!=None:
            early_depart_prev=self.dist_serve_add+self.pred_order.early_depart_wo_wait
        self.early_depart_prev=early_depart_prev
        trm1=self.dist_serve_add+early_depart_prev
        trm2=self.my_instance.early_start_full[self.u]
        self.early_start_u=trm2
        self.early_depart_wo_wait=min([trm1,trm2])
        self.earlyArrival=self.early_depart_wo_wait-self.dist_serve_add
        
    def compute_latest_depart(self):
        late_depart_prev=self.my_instance.late_start_full[self.w]
        if self.pred_order!=None:
            late_depart_prev=self.pred_order.lateDepart
        self.late_depart_prev=late_depart_prev
        trm1=self.my_instance.late_start_full[self.u]
        trm2=self.dist_serve_add+late_depart_prev
        self.lateDepart=max([trm1,trm2])
        self.late_start_u=trm1

class ng_help_valid_ineq:
    
    def __init__(self,my_VRP,ng_neigh_by_cust,max_SRI_Divisor=2,max_SRI_SET_SIZE=3):
        self.max_SRI_Divisor=max_SRI_Divisor
        self.max_SRI_SET_SIZE=max_SRI_SET_SIZE
        self.my_instance=my_VRP
        self.my_instance.early_start_full=list(self.my_instance.early_start)
        self.my_instance.early_start_full.append(np.inf)
        self.my_instance.early_start_full.append(np.inf)
        self.my_instance.late_start_full=list(self.my_instance.late_start)
        self.my_instance.late_start_full.append(0)
        self.my_instance.late_start_full.append(0)
        #list().append(np.inf)
        #if self.w<self.my_instance.num_cust:
        #    early_depart_prev=self.my_instance.early_start[self.w]
        self.ng_neigh_by_cust=ng_neigh_by_cust
        self.u_2_NG=ng_neigh_by_cust
        self.Nc=len(self.ng_neigh_by_cust)
        self.NC=self.Nc
        self.make_all_potential_nodes()
        self.make_all_arcs_need_eval()
        self.make_all_arcs_2_pred()
        self.construct_orderings()
        self.construct_edge_candidates()
        self.generate_subsets_to_consider()
        self.generate_SRI()
        self.generate_edge_2_SRI_contrib()
        self.uv_2_E=dict()
        print('self.E')
        print(self.E)
        for e in self.E:
            u=e[0][0]
            v=e[1][0]
            uv=tuple([u,v])
            if u==self.Nc and v==self.Nc+1 or v==self.Nc or u==self.Nc+1:
                print('[u,v]')
                print([u,v])
                print('e[0]')
                print(e[0])
                print('e[1]')
                print(e[1])
                input('error')
            if uv not in self.uv_2_E:
                self.uv_2_E[uv]=[]
            self.uv_2_E[uv].append(e)
        #print('self.uv_2_E')
        #print(self.uv_2_E)
        #input('---')
        self.non_source_sink_cust=np.arange(0,self.NC)#set(u_2_NG.keys())-set([self.my_VRP.NC,self.my_VRP.NC+1])
        
        
    def get_new_set_and_lost(self,i,v):
        u=i[0]
        Ni=i[1]
        ng_v=self.set_ng_by_v[v]
        ideal=set([u]).union(set(Ni))
        actual=ng_v.intersection(ideal)
        lost_terms=ideal-actual
        new_node=tuple([v,frozenset(actual)])
        return new_node,lost_terms
    def construct_edge_candidates(self):
        E=[]
        E_2_lost_terms=dict()
        for n in self.node_candidates:
            for v in np.arange(0,self.Nc+2):
                p=tuple([n[0],n[1],v])
                did_make=False
                if p in self.arc_2_orderings and len(self.arc_2_orderings[p])>0:
                    did_make=True
                    [new_node,lost_terms]= self.get_new_set_and_lost(n,v)           
                    if new_node not in self.node_candidates:
                        input('error here')
                    e=tuple([n,new_node])
                    E.append(e)
                    E_2_lost_terms[e]=lost_terms
                if 1<0:
                    print('p')
                    print(p)
                    print(did_make)
                    print(did_make)
                    print('self.Nc+2')
                    print(self.Nc+2)
                    input('--')
        G = nx.DiGraph()
        source=tuple([self.Nc,frozenset([])])
        G.add_edges_from(E)  # E is a list of (u, v) tuples
        if source not in G:
            return set()  # source not in graph

        reachable = nx.descendants(G, source)
        self.source_node=source
       
        reachable.add(source)  # include the source itself
        #print('node_candidates')
        #print(self.node_candidates)
        #3print('len(node_candidates)')
        #print(len(self.node_candidates))
        #print('E')
        #print(E)

        self.E = [edge for edge in E if edge[0] in reachable and edge[1] in reachable]
        #print('self.E ')
        #print(self.E )
        #input('---')
        self.E_2_lost_terms=dict()
        for e in self.E:
            self.E_2_lost_terms[e]=E_2_lost_terms[e]
        #print('reachable')
        #print(reachable)
        self.nodes=reachable
        self.sink_node=None
        for s in reachable:
            if s[0]==self.Nc+1:
                self.sink_node=s

        self.non_source_sink_nodes=reachable.copy()
        self.non_source_sink_nodes.remove(self.source_node)
        self.non_source_sink_nodes.remove(self.sink_node)

    

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
        #print('subset_2_make_SRI')
        #print(self.subset_2_make_SRI)
        #input('---')
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
                    #print('new_SRI')
                    #print(new_SRI)
                    my_SRI.append(new_SRI)
        self.my_SRI=my_SRI
        #print('my_SRI')
        #print(my_SRI)
        #input('my_SRI')
    def make_all_arcs_2_pred(self):
        self.arc_2_pred_arcs=dict()
        for p in self.my_arcs:
            u=p[0]
            bigN=set(p[1])
            v=p[2]
            preds = set()
            for w in bigN:

                new_bigN = bigN - {w}
                new_bigN=frozenset(new_bigN)
                pred = tuple([w, new_bigN, v])
                if pred not in self.my_arcs:
                    
                    input('error here')
                preds.add(pred)

            self.arc_2_pred_arcs[p] = preds
    def make_all_arcs_need_eval(self):
        self.my_arcs=set([])
        Nc=self.Nc
        all_inner_plus_candidates=set([])
        for n in self.node_candidates:
            all_cust=set([n[0]]).union(set(n[1]))
            all_cust=frozenset(all_cust)
            my_inner_set=power_set(all_cust)
            for p in my_inner_set:
                all_inner_plus_candidates.add(frozenset(p))
        for n in all_inner_plus_candidates:
            
            for u in n:#n[1]:
                inter_cust=n-set([u])
                if len(inter_cust)>0 and max(inter_cust)>=self.Nc-0.5 or u>self.Nc+0.5:
                    continue
                next_cust=set(np.arange(0,Nc+2))
                next_cust.remove(Nc)
                next_cust=next_cust-inter_cust
                if u in next_cust:
                    next_cust.remove(u)
                for v in next_cust:
                    this_arc=tuple([u,frozenset(inter_cust),v])
                    self.my_arcs.add(this_arc)
                    #if n==frozenset([5]):
                    #    print(this_arc)
                #if n==frozenset([5]):
                #    print('next_cust')
                #    print(next_cust)
                #    input('---')
    def construct_orderings(self):
        sorted_arcs = sorted(self.my_arcs, key=lambda arc: len(arc[1]))
        self.arc_2_orderings=dict()
        for p in sorted_arcs:
            if len(p[1])==0:
                new_ordering=NEW_order_object([p[0],p[2]],None,self.my_instance)
                if new_ordering.cost<np.inf:
                    self.arc_2_orderings[p]=[new_ordering]
                else:
                    self.arc_2_orderings[p]=[]


            else:
                all_pred_orderings=[]
                u=p[0]
                for my_pred_arc in self.arc_2_pred_arcs[p]:
                    for my_ord in self.arc_2_orderings[my_pred_arc]:
                        #print('u')
                        #print(u)
                        #print('p')
                        #print(p)
                        new_ord=my_ord.extend_order(u)
                        if new_ord.cost<np.inf:
                            all_pred_orderings.append(new_ord)
                self.arc_2_orderings[p]=self.compute_efficient_frontier(all_pred_orderings)
        print('done making base case orderings')
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

    def make_all_potential_nodes(self):
        Nc=self.Nc
        self.pow_set_by_u=dict()
        dist=self.my_instance.dist_mat_full
        self.node_candidates=set([])
        #self.node_head_candidates=set([])
        #self.node_2_node_head=dict()
        #self.possible_arcs=dict()
        #self.E_candid=dict()
        self.ng_neigh_by_cust.append([])
        self.ng_neigh_by_cust.append([])
        #print('Nc')
        self.set_ng_by_v=dict()
        #print(Nc)
        #input('--')
        for v in range(0,Nc+2):
            self.pow_set_by_u[v]=power_set(self.ng_neigh_by_cust[v])
            self.set_ng_by_v[v]=set(self.ng_neigh_by_cust[v])
            for p in self.pow_set_by_u[v]:
                
                new_node=tuple([v,frozenset(p)])
                self.node_candidates.add(new_node)
                #self.node_2_node_head[new_node]=set([])
                #set_p=set(p)
                #for u in p:
                #    tmp_set=set(p)-set([new_node])
                #    tmp_set=frozenset(tmp_set)
                #    new_arc=tuple([u,tmp_set,v])
                #    self.node_head_candidates.add(new_arc)
                #    #self.node_2_node_head[new_node].add(new_arc)
                #next_cust=set(np.arange([0,Nc+2]))
                #next_cust.remove(Nc)
                #next_cust.remove(u)
                #next_cust=next_cust-set(set_p)
                

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
            #print('tmp_dict')
            #print(tmp_dict)
            #input('---')
            tmp_dict = {e: coeff for e, coeff in tmp_dict.items() if coeff < 0}
            self.dict_valid_ineq_name_edge_2_coeff[q_name] = tmp_dict

                
    