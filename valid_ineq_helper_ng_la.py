

import numpy as np
# Set desired solver options
import itertools
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
import time
from itertools import combinations

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
        #or u in self.my_order_name:
        #    self.dem_in_arc+=self.my_instance.dem_full[u]
        if pred_order==None:
            self.dem_in_arc = sum(self.my_instance.dem_full[u] for u in self.my_order_name)
        else:
            self.dem_in_arc=self.my_instance.dem_full[self.u]+pred_order.dem_in_arc
        if self.dem_in_arc>self.my_instance.vehicle_capacity:
            self.cost=np.inf

    def extend_order(self,new_u):
        
        trm1=[new_u]
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
        #print('making nodes')
        self.make_all_potential_nodes()
        #print('make_all_arcs_need_eval')

        self.make_all_arcs_need_eval()
        #print('make_all_arcs_2_pred')
        self.make_all_arcs_2_pred()
        #print('construct_orderings')
        self.construct_orderings()
        #print('construct_edge_candidates')
        self.construct_edge_candidates()
        #print('generate_subsets_to_consider')
        self.generate_subsets_to_consider()
        #print('generate_SRI')
        self.generate_SRI()
        #print('generate_edge_2_SRI_contrib')
        self.generate_edge_2_SRI_contrib()
        self.generate_self_self_edges()
        #print('alst stuff')
        self.uv_2_E=dict()
        #print('self.E')
        #print(self.E)
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
        #print('done init stuff')


    def generate_self_self_edges(self):

        self.u_2_node_list=dict()
        for u in range(0,self.Nc+2):
            self.u_2_node_list[u]=[]
        
        for n in self.nodes:
            self.u_2_node_list[n[0]].append(n)
        
        for u in range(self.Nc):
            NL = self.u_2_node_list[u]
            len_list = [len(node[1]) for node in NL]  # precompute subset lengths

            for k1 in range(len(NL)):
                sub1 = NL[k1][1]
                len1 = len_list[k1]
                for k2 in range(len(NL)):
                    sub2 = NL[k2][1]
                    if len1 == len_list[k2] - 1 and sub1.issubset(sub2):
                        self.E.append((NL[k1], NL[k2]))
    def get_new_set_and_lost(self,i,v):
        u=i[0]
        Ni=i[1]
        ng_v=self.set_ng_by_v[v]
        ideal=set([u]).union(set(Ni))
        actual=ng_v.intersection(ideal)
        lost_terms=ideal-actual
        new_node=tuple([v,frozenset(actual)])
        return new_node,lost_terms
    
    def is_allowable_transition(self,n,v):
        #print('starting')
        #print('starting')
        #print('starting')
        #p#rint('starting')
        #p#rint('starting')
        #print('starting')
        #print('starting')
        #print('starting')
        w=n[0]
        if v==n[0] or v in n[1]:
            return False
        if len(n[1])==0:
            #print('in here')
            p=tuple([w,frozenset([]),v])
            #print(p)
            
            if p in self.arc_2_orderings and len(self.arc_2_orderings[p])>0 and self.arc_2_orderings[p][0].cost<np.inf:
                return True

        prev_cust_set=set(n[1])
        for u in prev_cust_set:
            #print('u')
            #print(u)
            N_minus_plus=prev_cust_set.copy()
            if N_minus_plus==None:
                N_minus_plus=set([])
            #print('N_minus_plus')
            #print(N_minus_plus)
            N_minus_plus.add(w)
            N_minus_plus.remove(u)
            p=tuple([u,frozenset(N_minus_plus),v])
            #3print('p')
            #print(p)
            if p not in self.arc_2_orderings:
                continue
            for my_ord in self.arc_2_orderings[p]:
                if my_ord.my_order_name[-2]==w and my_ord.cost<np.inf:
                    #input('FOUND ONE')
                    return True
        
        #print('n')
        #print(n)
        #print('v')
        #p3rint(v)
        #i#nput('Lets check')
        return False



    def construct_edge_candidates(self):
        E=[]
        E_2_lost_terms=dict()
        for n in self.node_candidates:
            #if n[0]==self.Nc:
            #    print('looking')
            #    print('self.nc=  '+str(self.Nc))
            for v in np.arange(0,self.Nc+2):
                #if n[0]==self.Nc:
                #    print('v = '+str(v))
                #p=tuple([n[0],n[1],v])
                did_make=False
                if self.is_allowable_transition(n,v):
                    [new_node,lost_terms]= self.get_new_set_and_lost(n,v)           
                    if new_node not in self.node_candidates:
                        input('error here')
                    e=tuple([n,new_node])
                    E.append(e)
                    E_2_lost_terms[e]=lost_terms
                    #if n[0]==self.Nc and v<self.Nc:
                    #    print('GOOD E')
                    #    print(e)
                else:
                    if n[0]==self.Nc and v<self.Nc:
                        input('wrong')
        G = nx.DiGraph()
        source=tuple([self.Nc,frozenset([])])
        G.add_edges_from(E)  # E is a list of (u, v) tuples
        if source not in G:
            return set()  # source not in graph

        reachable = nx.descendants(G, source)
        self.source_node=source
       
        reachable.add(source) 
        unreachable=self.node_candidates-set(reachable)
       # print('unreachable')
       # print(unreachable)
       # input('--')
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
        my_subsets = set()

        for u in range(self.NC):
            neighbors = self.u_2_NG[u]
            items = neighbors + [u]
            items_set = sorted(set(items))  # deduplicate + consistent order

            for k in range(1, self.max_SRI_SET_SIZE + 1):
                for subset in combinations(items_set, k):
                    my_subsets.add(frozenset(subset))

        self.subset_2_make_SRI = my_subsets


    def OLD_generate_subsets_to_consider(self):
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
                    my_SRI.append(new_SRI)
        self.my_SRI=my_SRI
    def make_all_arcs_2_pred(self):
        t1=time.time()
        self.arc_2_pred_arcs = {}

        for (u, bigN_frozen, v) in self.my_arcs:
            bigN = set(bigN_frozen)
            preds = {
                (w, frozenset(bigN - {w}), v)
                for w in bigN
            }
            self.arc_2_pred_arcs[(u, bigN_frozen, v)] = preds
        t1=time.time()-t1
        print(t1)
        #input('time')

    def OLD_make_all_arcs_2_pred(self):
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
    # Group arcs by size of the middle element
        size_to_arcs = {}
        #print('pt1 ')
        for arc in self.my_arcs:
            k = len(arc[1])
            if k not in size_to_arcs:
                size_to_arcs[k] = []
            size_to_arcs[k].append(arc)

        self.arc_2_orderings = {}
        #print('pt2 ')

        # Base case: arcs with empty middle element
        for p in size_to_arcs.get(0, []):
            u, _, v = p
            new_ordering = NEW_order_object([u, v], None, self.my_instance)
            self.arc_2_orderings[p] = [new_ordering] if new_ordering.cost < np.inf else []
        #print('pt3 ')

        # Process remaining arcs in increasing size
        for size in sorted(size_to_arcs):
            #print('pt4 '+str(size))

            if size == 0:
                continue
            for p in size_to_arcs[size]:
                u, _, _ = p
                all_pred_orderings = []
                for pred_arc in self.arc_2_pred_arcs[p]:
                    for ord in self.arc_2_orderings.get(pred_arc, []):
                        new_ord = ord.extend_order(u)
                        if new_ord.cost < np.inf:
                            all_pred_orderings.append(new_ord)
                self.arc_2_orderings[p] = self.compute_efficient_frontier(all_pred_orderings)

        #print('done making base case orderings')

    def OLD_construct_orderings(self):

        #self.my_arcs contains tuples of 3 elements;  the middle element is used to sort them for processing them.  
        #however there are only a small number of consequtive discrete values starting at zero.  So we dont really need to do a sort. 
        # More like producing a list of lists 
        
        sorted_arcs = sorted(self.my_arcs, key=lambda arc: len(arc[1]))
        self.arc_2_orderings=dict()
        
        #we now process the sizs in order
        for p in sorted_arcs:
            if len(p[1])==0:#if we procesed all fo the ones first then we would be able to avoid this if statement 
                new_ordering=NEW_order_object([p[0],p[2]],None,self.my_instance)
                if new_ordering.cost<np.inf:
                    self.arc_2_orderings[p]=[new_ordering]
                else:
                    self.arc_2_orderings[p]=[]


            else:
                #now we can likely speed this up dramatically 
                all_pred_orderings=[]
                u=p[0]
                for my_pred_arc in self.arc_2_pred_arcs[p]:
                    for my_ord in self.arc_2_orderings[my_pred_arc]:
                
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

        # Step 1: Group edges by their lost terms
        lost_terms_to_edges = defaultdict(list)
        for edge, terms in E_2_lost_terms.items():
            key = frozenset(terms)
            lost_terms_to_edges[key].append(edge)

        unique_lost_sets = list(lost_terms_to_edges.keys())

        # Step 2: Build inverted index: customer → lost sets that include them
        cust_to_lostsets = defaultdict(set)
        for lost_set in unique_lost_sets:
            for cust in lost_set:
                cust_to_lostsets[cust].add(lost_set)

        # Step 3: Process each SRI constraint
        for q in tqdm(my_SRI, desc='generating SRI contrib'):
            Nhat = set(q['customers'])
            k = q['my_divisor']
            rhs = q['my_RHS']
            k_inv = 1.0 / k
            q_name = f"{frozenset(Nhat)}_{k}_{rhs}"

            # Precompute RHS of valid inequality
            self.dict_valid_ineq_name_2_rhs[q_name] = -int(len(Nhat) * k_inv)

            tmp_dict = {}

            # Step 4: Only consider lost sets that share customers with Nhat
            relevant_lost_sets = set()
            for cust in Nhat:
                relevant_lost_sets.update(cust_to_lostsets.get(cust, []))

            # Step 5: Compute contributions
            for lost_set in relevant_lost_sets:
                intersection_size = len(lost_set & Nhat)
                #if intersection_size < k:
                #    continue

                coeff = -int(intersection_size * k_inv)
                if coeff >= 0:
                    continue

                for edge in lost_terms_to_edges[lost_set]:
                    tmp_dict[edge] = coeff

            self.dict_valid_ineq_name_edge_2_coeff[q_name] = tmp_dict


    def OLD_generate_edge_2_SRI_contrib(self):
        """Optimized version of generate_edge_2_SRI_contrib"""
        self.dict_valid_ineq_name_2_rhs = {}
        self.dict_valid_ineq_name_edge_2_coeff = {}

        E_2_lost_terms = self.E_2_lost_terms
        my_SRI = self.my_SRI

        # Pre-compute all unique lost term sets and their intersections
        unique_lost_sets = list(set(frozenset(terms) for terms in E_2_lost_terms.values()))
        
        # Group edges by their lost terms to avoid redundant computation
        lost_terms_to_edges = defaultdict(list)
        for edge, terms in E_2_lost_terms.items():
            lost_terms_to_edges[frozenset(terms)].append(edge)


        for q in tqdm(my_SRI,desc='generating SRI contrib'):

                
            Nhat = set(q['customers'])
            Nhat_frozen = frozenset(Nhat)
            k = q['my_divisor']
            rhs = q['my_RHS']
            k_inv = 1.0 / k
            q_name = f"{Nhat_frozen}_{k}_{rhs}"

            # RHS of valid inequality
            self.dict_valid_ineq_name_2_rhs[q_name] = -np.floor(len(Nhat) * k_inv)

            # Build edge contribution dict efficiently
            tmp_dict = {}
            
            for lost_set_frozen in unique_lost_sets:
                lost_set = set(lost_set_frozen)
                intersection_size = len(lost_set & Nhat)
                
                if intersection_size >= k:  # Only process if there's potential contribution
                    coeff = -np.floor(intersection_size * k_inv)
                    if coeff < 0:  # Only keep negative coefficients
                        # Apply this coefficient to all edges with these lost terms
                        for edge in lost_terms_to_edges[lost_set_frozen]:
                            tmp_dict[edge] = coeff

            self.dict_valid_ineq_name_edge_2_coeff[q_name] = tmp_dict

                
    