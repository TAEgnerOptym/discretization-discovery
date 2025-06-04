import gurobipy as gp
from gurobipy import GRB
import time
from collections import defaultdict
import networkx as nx
import numpy as np
# Set desired solver options
import itertools
import gurobipy as gp
from solve_gurobi_lp import solve_gurobi_lp
import sys
from tqdm import tqdm
import json
from itertools import combinations
from New_valid_sep.NEW_order_object_new_sep import NEW_order_object_new_sep
from New_valid_sep.power_set import power_set
from New_valid_sep.compute_efficient_fronteir import compute_efficient_frontier
from New_valid_sep.NEW_order_object_new_sep_backwards import NEW_order_object_new_sep_backwards
class graph_localized_g:

    def __init__(self,MF,my_subset_cust,OPT):
        self.do_external=False

        self.MF=MF
        self.my_instance=self.MF.my_VRP
        self.my_subset_cust=my_subset_cust
        self.OPT=OPT
        Nc = self.MF.my_VRP.num_cust
        self.Nc=Nc

        self.set_ng_by_v=dict()
        for v in range(0,self.Nc+2):
            self.set_ng_by_v[v]=set([])
            if v in self.my_subset_cust:
                self.set_ng_by_v[v]=self.my_subset_cust-set([v])
        self.non_subset_cust=set(range(0,self.Nc))-self.my_subset_cust
        self.non_subset_cust_plus_end=self.non_subset_cust.copy()
        self.non_subset_cust_plus_start=self.non_subset_cust.copy()
        self.non_subset_cust_plus_start_and_end=self.non_subset_cust.copy()

        self.non_subset_cust_plus_end.add(self.Nc+1)
        self.non_subset_cust_plus_start.add(self.Nc)

        self.non_subset_cust_plus_start_and_end.add(self.Nc+1)
        self.non_subset_cust_plus_start_and_end.add(self.Nc)
        my_times=dict()
        t1=time.time()
        self.make_candid_nodes()
        my_times['make_candid_nodes']=time.time()-t1
        print('create_arcs')
        t1=time.time()
        self.create_arcs()
        my_times['create_arcs']=time.time()-t1
        t1=time.time()
        #print('make all arcs pred')
        #self.make_all_arcs_2_pred()
        #my_times['make_all_arcs_2_pred']=time.time()-t1
        print('make all arcs pred back')

        t1=time.time()
        self.make_all_arcs_2_pred_backwards()
        my_times['make_all_arcs_2_pred_backwards']=time.time()-t1
        #input('hold')
        print('construct_orderings_back')
        t1=time.time()
        self.construct_orderings_back()
        my_times['construct_orderings_back']=time.time()-t1
        #t1=time.time()
        #self.construct_orderings()
        #my_times['construct_orderings']=time.time()-t1
        print('construct_edge_candidates')
        t1=time.time()
        self.construct_edge_candidates()
        my_times['construct_edge_candidates']=time.time()-t1
        print('generate_subsets_to_consider')
        t1=time.time()
        self.generate_subsets_to_consider()
        my_times['generate_subsets_to_consider']=time.time()-t1
        t1=time.time()
        #print('generate_SRI')
        self.generate_SRI()
        my_times['generate_SRI']=time.time()-t1
        t1=time.time()
        #print('generate_edge_2_SRI_contrib')
        self.generate_edge_2_SRI_contrib()
        my_times['generate_edge_2_SRI_contrib']=time.time()-t1
        #t1=time.time()
        #self.generate_self_self_edges()
        #my_times['generate_self_self_edges']=time.time()-t1
       
        print('my_times')
        print(my_times)

    def is_allowable_transition(self,n,v):

        w=n[0]
        if v==n[0] or v in n[1]:
            return False
        if len(n[1])==0:
            p=tuple([w,frozenset([]),v])
            
            if p in self.arc_2_orderings and len(self.arc_2_orderings[p])>0 and self.arc_2_orderings[p][0].cost<np.inf:
                return True
        prev_cust_set=set(n[1])
        #if v in self.my_subset_cust and len(n[1])>0 and self.MF.my_VRP.dist_mat_full[w,v]<np.inf for all w in n[1] and self.MF.my_VRP.dist_mat_full[n[0],v]<np.inf:
        #    return True
        if (
            v in self.my_subset_cust
            and len(n[1]) > 0
            and all(self.MF.my_VRP.dist_mat_full[w, v] < np.inf for w in n[1])
            and self.MF.my_VRP.dist_mat_full[n[0], v] < np.inf
        ):
            return True

        if v in self.my_subset_cust:
            return False
        for u in prev_cust_set:

            N_minus_plus=prev_cust_set.copy()
            if N_minus_plus==None:
                N_minus_plus=set([])

            N_minus_plus.add(w)
            N_minus_plus.remove(u)
            p=tuple([u,frozenset(N_minus_plus),v])

            if p not in self.arc_2_orderings:
                continue
            for my_ord in self.arc_2_orderings[p]:
                if my_ord.my_order_name[-2]==w and my_ord.cost<np.inf:
                    return True
        return False        

    def OLD_is_allowable_transition(self,n,v):

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
                    return True
        
        #print('n')
        #print(n)
        #print('v')
        #p3rint(v)
        #i#nput('Lets check')
        return False


    def making_edges_component(self):
        E=[]
        t1=time.time()
        E_2_lost_terms=dict()

        self.big_nodes = [n for n in self.my_candid_nodes if len(n[1]) > 0]
        self.small_nodes = [n for n in self.my_candid_nodes if len(n[1]) == 0]
        
        for n in self.small_nodes:
            for v in np.arange(0,self.Nc+2):
                did_make=False
                if self.is_allowable_transition(n,v):
                    [new_node,lost_terms]= self.get_new_set_and_lost(n,v)           
                    e=tuple([n,new_node])
                    E.append(e)
                    E_2_lost_terms[e]=lost_terms


        reachable_loc_to_v_from_subset = {
            v: {
                w for w in self.my_subset_cust
                if self.MF.my_VRP.dist_mat_full[w, v] < np.inf
            }
            for v in range(self.Nc + 2)
        }

        
        for n in self.big_nodes:
             w_set = n[1]
             common_vs = set.intersection(*(reachable_loc_to_v_from_subset[w] for w in w_set))
             common_vs=common_vs.intersection(self.my_subset_cust)
             for v in common_vs:#self.my_subset_cust:
                if self.is_allowable_transition(n,v):
                    [new_node,lost_terms]= self.get_new_set_and_lost(n,v) 
        
        for n in self.big_nodes:
             w_set = n[1]
             common_vs = set.intersection(*(reachable_loc_to_v_from_subset[w] for w in w_set))
             common_vs=common_vs.intersection(self.non_subset_cust_plus_end)
             for v in common_vs:#self.my_subset_cust:
                #if self.is_allowable_transition(n,v):
                [new_node,lost_terms]= self.get_new_set_and_lost(n,v) 
        time_1=time.time()-t1
        return E,E_2_lost_terms,time_1
    
    def construct_edge_candidates(self):
        E,E_2_lost_terms,time_1=self.making_edges_component()
        print('time making edges')
        print(time_1)
        print('time making edges')
        G = nx.DiGraph()
        source=tuple([self.Nc,frozenset([])])
        G.add_edges_from(E)  # E is a list of (u, v) tuples
        if source not in G:
            return set()  # source not in graph

        reachable = nx.descendants(G, source)
        self.source_node=source
       
        reachable.add(source) 
        unreachable=set(self.my_candid_nodes)-set(reachable)
      
        self.E = [edge for edge in E if edge[0] in reachable and edge[1] in reachable]

        self.E_2_lost_terms=dict()
        for e in self.E:
            self.E_2_lost_terms[e]=E_2_lost_terms[e]

        self.nodes=reachable
        self.sink_node=None
        for s in reachable:
            if s[0]==self.Nc+1:
                self.sink_node=s

        self.non_source_sink_nodes=reachable.copy()
        self.non_source_sink_nodes.remove(self.source_node)
        self.non_source_sink_nodes.remove(self.sink_node)

    
    def get_new_set_and_lost(self,i,v):
        u=i[0]
        Ni=i[1]
        ng_v=self.set_ng_by_v[v]
        ideal=set([u]).union(set(Ni))
        actual=ng_v.intersection(ideal)
        lost_terms=ideal-actual
        new_node=tuple([v,frozenset(actual)])
        return new_node,lost_terms

    def construct_orderings(self):
    # Group arcs by size of the middle element
        size_to_arcs = {}
        print('pt1 ')
        t1=time.time()
        for arc in self.my_arcs:
            k = len(arc[1])
            if k not in size_to_arcs:
                size_to_arcs[k] = []
            size_to_arcs[k].append(arc)
        t1=t1-time.time()
        t2=time.time()
        self.arc_2_orderings = {}
        #print('pt2 ')

        # Base case: arcs with empty middle element
        for p in size_to_arcs.get(0, []):
            u, _, v = p
            new_ordering = NEW_order_object_new_sep([u, v], None, self.my_instance)
            self.arc_2_orderings[p] = [new_ordering] if new_ordering.cost < np.inf else []
        #print('pt3 ')
        t2=time.time()-t2
        t3=time.time()
        # Process remaining arcs in increasing size
        my_sizes=sorted(size_to_arcs.keys())
        my_sizes.remove(0)
        big_poss=0
        for size in my_sizes:
            print('startign +'+str(size))
            print('len(size_to_arcs[size])')
            print(len(size_to_arcs[size]))
            for p in size_to_arcs[size]:
                u, _, _ = p
                all_pred_orderings = []
                this_count=0
                for pred_arc in self.arc_2_pred_arcs[p]:
                    for ord in self.arc_2_orderings.get(pred_arc, []):
                        this_count=this_count+1
                        new_ord = ord.extend_order(u)
                        if new_ord.cost < np.inf:
                            all_pred_orderings.append(new_ord)
                big_poss=max([big_poss,this_count])
                self.arc_2_orderings[p] = compute_efficient_frontier(all_pred_orderings)
        t3=time.time()-t3
        #print('[t1,t2,t3]')
        #print([t1,t2,t3])
        #print('[t1,t2,t3]')
        #print('big_poss')
        #print(big_poss)
        #input('hold')
    
    def construct_orderings_back(self):
    # Group arcs by size of the middle element
        self.arc_2_orderings=dict()
        self.internal_times=dict()
        t1=time.time()
        subset = self.my_subset_cust  # Local alias for speed
        keys = self.arc_2_pred_arcs_backward.keys()
        #print('len(subset)')
        #print(len(subset))
        #input('---')
        set_internal_start_internal_end_with_middle = {key for key in keys if key[2] in subset and key[0] in subset and len(key[1])>0}
        #print('len(set_internal_start_internal_end_with_middle)')
        #print(len(set_internal_start_internal_end_with_middle))
        set_internal_start_internal_end_without_middle = {key for key in keys if key[2] in subset and key[0] in subset and len(key[1])==0}
        #print('len(set_internal_start_internal_end_without_middle)')
        #print(len(set_internal_start_internal_end_without_middle))
        set_internal_start_external_end_with_middle={key for key in keys if key[2] not in subset and key[0] in subset and len(key[1])>0}
        #print('set_internal_start_external_end_with_middle')
        #print(len(set_internal_start_external_end_with_middle))
        set_internal_start_external_end_without_middle={key for key in keys if key[2] not in subset and key[0] in subset and len(key[1])==0}
        #print('set_internal_start_external_end_without_middle')
        #print(len(set_internal_start_external_end_without_middle))

        set_external_start_internal_end = {key for key in keys if key[2] in subset and key[0] not in subset }
        #print('set_external_start_internal_end')
        #print(len(set_external_start_internal_end))

        set_external_start_external_end = {key for key in keys if key[2] not in subset and key[0] not in subset }
        
        #EVIL_set_external_start_external_end = {key for key in keys if key[2] not in subset and key[0] not in subset and len(key[1])>0}
        #if len(EVIL_set_external_start_external_end)>0.5:
        #    for p in EVIL_set_external_start_external_end:
        #        print('p')
        #        print(p)
        #        print('subset')
        #        print(subset)
        #        input('--')
        #    input('er')
        #print('set_external_start_external_end')
        #print(len(set_external_start_external_end))

        self.internal_times['time_init']=time.time()-t1
        t1=time.time()

        set_internal_internal_by_size=dict()
        for k in range(1,len(subset)+1):
            set_internal_internal_by_size[k]=[]
        for p in set_internal_start_internal_end_with_middle:
            set_internal_internal_by_size[len(p[1])].append(p)
        #print('len(set_internal_start_internal_end_with_middle)')
        #print(len(set_internal_start_internal_end_with_middle))
        #for k in range(1,len(subset)+1):
        #    print('size(k)')
        #    print(len(set_internal_internal_by_size[k]))
        #input('---')
        set_internal_external_by_size=dict()
        for k in range(1,len(subset)+1):
            set_internal_external_by_size[k]=[]
        for p in set_internal_start_external_end_with_middle:
            set_internal_external_by_size[len(p[1])].append(p)
        self.internal_times['time_init']=time.time()-t1

        print('starting')
        t1=time.time()
        for p in set_internal_start_internal_end_without_middle:
            self.update_back_given_p_no_mid(p)
        self.internal_times['set_internal_start_internal_end_without_middle']=time.time()-t1
        total_orderings = sum(len(orderings) for orderings in self.arc_2_orderings.values())
        #print('self.arc_2_orderings')
        #print(self.arc_2_orderings)
        print('total_orderings, set_internal_start_internal_end_without_middle')
        print(total_orderings)
       # input('after set_internal_start_internal_end_without_middle ')
        t1=time.time()
        for p in set_internal_start_external_end_without_middle:

            self.update_back_given_p_no_mid(p)
        self.internal_times['set_internal_start_external_end_without_middle']=time.time()-t1
        total_orderings = sum(len(orderings) for orderings in self.arc_2_orderings.values())

        #print('total_orderings, set_internal_start_external_end_without_middle')
        #print(total_orderings)
        #input('after set_internal_start_external_end_without_middle ')

        t1=time.time()
        for p in set_external_start_internal_end:
            self.update_back_given_p_no_mid(p)
        self.internal_times['set_external_start_internal_end']=time.time()-t1
        total_orderings = sum(len(orderings) for orderings in self.arc_2_orderings.values())

        #print('total_orderings set_external_start_internal_end')
        #print(total_orderings)
        #input('---')
        t1=time.time()
        for p in set_external_start_external_end:
            self.update_back_given_p_no_mid(p)
        total_orderings = sum(len(orderings) for orderings in self.arc_2_orderings.values())

        #print('total_orderings set_external_start_external_end')
        #print(total_orderings)
        #input('---')
        self.internal_times['set_external_start_external_end']=time.time()-t1
        #total_orderings = sum(len(orderings) for orderings in self.arc_2_orderings.values())

        #print('total_orderings')
        #print(total_orderings)
        for k in range(1,len(self.my_subset_cust)):
            t1=time.time()
            print('k =  '+str(k))
            print('len(set_internal_internal_by_size[k])')
            print(len(set_internal_internal_by_size[k]))
            #input('---')
            for p in set_internal_internal_by_size[k]:
                self.update_back_given_p(p)
            self.internal_times['set_internal_start_external_end_with_middle'+str(k)]=time.time()-t1
            total_orderings = sum(len(orderings) for orderings in self.arc_2_orderings.values())
        if self.do_external==True:
            print('Starting the external ones')
            t1=time.time()
            for k in range(1,len(self.my_subset_cust)):
                t1=time.time()
                for p in set_internal_external_by_size[k]:
                    self.update_back_given_p(p)
            
    def update_back_given_p_no_mid(self,p):
            u, _, v = p
            new_ordering = NEW_order_object_new_sep_backwards([u, v], None, self.my_instance)
            self.arc_2_orderings[p] = [new_ordering] if new_ordering.cost < np.inf else []
            
            my_dist=self.MF.my_VRP.dist_mat_full[u,v]
            if my_dist<np.inf and new_ordering.cost==np.inf:
                input('error here')
            
            if new_ordering.cost==np.inf and (u not in self.my_subset_cust and v not in self.my_subset_cust ):
                print('flag me')
                print([u,v])
                input('---')
            
    def update_back_given_p(self,p):

        u, _, v = p
        all_pred_orderings = []
        this_count=0
        #print('p')
        for pred_arc in self.arc_2_pred_arcs_backward[p]:
            for ord in self.arc_2_orderings.get(pred_arc, []):
                this_count=this_count+1
                new_ord = ord.extend_order(v)
                if v in ord.my_order_name:
                    input('errror')
                if new_ord.cost < np.inf:
                    all_pred_orderings.append(new_ord)
        self.arc_2_orderings[p] = compute_efficient_frontier(all_pred_orderings)


    def create_arcs(self):
        dist=self.MF.my_VRP.dist_mat_full#[:Nc,:Nc]


        self.my_arcs=[]
        t1=time.time()
        for u in range(0,self.Nc+2):
            for v in range(0,self.Nc+2):
                if u not in self.my_subset_cust or v not in self.my_subset_cust:
                    if dist[u,v]<np.inf:
                        new_arc=tuple([u,frozenset([]),v])
                        self.my_arcs.append(new_arc)
        t1=time.time()-t1
        for p in self.my_arcs:
            if p[0] not in self.my_subset_cust and len(p[1])>0:
                print(p)
                input('error here00')
        t2=time.time()
        for i in self.internal_node_candid:

            this_set=i[1]
            v=i[0]
            for u in i[1]:
                new_intermediate=this_set-set([u])
                new_arc=tuple([u,frozenset(new_intermediate),v])

                self.my_arcs.append(new_arc)
        for p in self.my_arcs:
            if p[0] not in self.my_subset_cust and len(p[1])>0:
                print(p)
                input('error here 1')
        t2=time.time()-t2
        t3=time.time()
        all_poss=power_set(self.my_subset_cust)
        t3=time.time()-t3
        t4=time.time()
        if self.do_external==True:

            for p in all_poss:
                if len(p)>1:
                    
                    set_p=set(p)
                    for u in p:
                        
                        for v in self.non_subset_cust_plus_end:
                            new_intermediate=set_p-set([u])
                            new_arc=tuple([u,frozenset(new_intermediate),v])
                            self.my_arcs.append(new_arc)
        t4=time.time()-t4
        print('before len')
        print(len(self.my_arcs))
        t5=time.time()
    
        self.my_arcs=set(self.my_arcs)
        t5=time.time()-t5
        print('after len')
        print(len(self.my_arcs))
        print('[t1,t2,t3,t4,t5]')
        print([t1,t2,t3,t4,t5])
        print('[t1,t2,t3,t4,t5]')
        for p in self.my_arcs:
            if p[0] not in self.my_subset_cust and len(p[1])>0:
                print(p)
                input('error here')
    def make_all_arcs_2_pred_backwards(self):
        t1 = time.time()

        self.arc_2_pred_arcs_backward = {}

        # First-level cache: bigN_frozen -> {w: frozenset(bigN - {w})}
        bigN_subset_cache = {}

        # Second-level cache: (u, bigN_frozen) -> set of preds
        arc_pattern_cache = {}
        for (u, bigN_frozen, v) in self.my_arcs:
            # First cache layer: subsets of bigN
            if bigN_frozen not in bigN_subset_cache:
                bigN = set(bigN_frozen)
                bigN_subset_cache[bigN_frozen] = {
                    w: frozenset(bigN - {w}) for w in bigN
                }

            subset_map = bigN_subset_cache[bigN_frozen]

            # Second cache layer: full preds set for fixed (u, bigN)
            arc_key = (u, bigN_frozen)
            if arc_key not in arc_pattern_cache:
                preds = {
                    (u, subset_map[w], w) for w in subset_map
                }
                arc_pattern_cache[arc_key] = preds

            # Now reuse the cached preds (they already include u and each w)
            self.arc_2_pred_arcs_backward[(u, bigN_frozen, v)] = arc_pattern_cache[arc_key]

        print(time.time() - t1)


    def make_all_arcs_2_pred(self):
        t1=time.time()
        self.arc_2_pred_arcs = {}

        arc_2_pred_arcs = {}
        bigN_to_preds_cache = {}  # cache: frozenset(bigN) -> set of preds

        for u, bigN_frozen, v in self.my_arcs:
            if bigN_frozen not in bigN_to_preds_cache:
                bigN = set(bigN_frozen)
                frozenset_cache = {
                    w: frozenset(bigN - {w}) for w in bigN
                }
                preds_template = {
                    (w, frozenset_cache[w]) for w in bigN
                }
                bigN_to_preds_cache[bigN_frozen] = preds_template

            # Now just reuse and inject the arc’s v
            preds = {
                (w, subset, v) for (w, subset) in bigN_to_preds_cache[bigN_frozen]
            }
            arc_2_pred_arcs[(u, bigN_frozen, v)] = preds

        self.arc_2_pred_arcs = arc_2_pred_arcs
        t1=time.time()-t1
        print(t1)

    def make_candid_nodes(self):
        my_subset=self.my_subset_cust
        Nc = self.MF.my_VRP.num_cust
        self.Nc=Nc
        my_nodes=[]
        external_nodes=[]
        for u in self.non_subset_cust_plus_start_and_end:
            i=tuple([u,frozenset([])])
            my_nodes.append(i)
            external_nodes.append(i)
        all_poss=power_set(my_subset)
        internal_nodes=[]
        for p in all_poss:
            #print('my_subset')
            #print(my_subset)
            #print('p')
            #print(p)
            poss_start=my_subset-set(p)
            for u in poss_start:
                i=tuple([u,frozenset(p)])
                my_nodes.append(i)
                internal_nodes.append(i)
        self.my_candid_nodes=my_nodes
        self.internal_node_candid=internal_nodes
        self.external_nodes=external_nodes
    
    def generate_subsets_to_consider(self):
        my_subsets = set()
        

        for k in range(1, self.OPT['max_SRI_SET_SIZE'] + 1):
            for subset in combinations(self.my_subset_cust, k):
                my_subsets.add(frozenset(subset))

        self.subset_2_make_SRI = my_subsets


    def  generate_SRI(self):
        my_SRI=[]
        max_SRI_size=self.OPT['max_SRI_SET_SIZE']
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
    
    
    def generate_edge_2_SRI_contrib(self):
        self.dict_valid_ineq_name_2_rhs = {}
        myTIMES_SRI=dict()
        self.dict_valid_ineq_name_edge_2_coeff = {}

        E_2_lost_terms = self.E_2_lost_terms
        my_SRI = self.my_SRI

        t1=time.time()
        # Step 1: Group edges by their lost terms
        lost_terms_to_edges = defaultdict(list)
        for edge, terms in E_2_lost_terms.items():
            key = frozenset(terms)
            lost_terms_to_edges[key].append(edge)
        myTIMES_SRI['pt1']=time.time()-t1
        t1=time.time()
        
        unique_lost_sets = list(lost_terms_to_edges.keys())

        # Step 2: Build inverted index: customer → lost sets that include them
        cust_to_lostsets = defaultdict(set)
        for lost_set in unique_lost_sets:
            for cust in lost_set:
                cust_to_lostsets[cust].add(lost_set)
        myTIMES_SRI['pt2']=time.time()-t1
        t1=time.time()
        self.cust_to_lostsets=cust_to_lostsets
        self.lost_terms_to_edges=lost_terms_to_edges
        #self.fun_SRI_q()
    #def fun_SRI_q(self):
    #    my_SRI = self.my_SRI
     #   cust_to_lostsets=self.cust_to_lostsets
     #   lost_terms_to_edges=self.lost_terms_to_edges
        
        dict_valid_ineq_name_2_rhs = self.dict_valid_ineq_name_2_rhs
        dict_valid_ineq_name_edge_2_coeff = self.dict_valid_ineq_name_edge_2_coeff

        # Step 1: Collect all unique Nhat sets
        unique_nhats = set(q['customers'] for q in my_SRI)

        # Step 2: Precompute contributions for each unique Nhat
        nhat_to_info = {}

        for nhat in unique_nhats:
            nhat_len = len(nhat)

            # Collect relevant lost sets
            relevant_lost_sets = set()
            for cust in nhat:
                relevant_lost_sets.update(cust_to_lostsets.get(cust, []))

            nhat_to_info[nhat] = {
                "len": nhat_len,
                "relevant_lost_sets": relevant_lost_sets,
                # Leave out k-specific coeffs for now — computed per q
            }

        # Step 3: Process each q
        for q in tqdm(my_SRI, desc='generating SRI contrib'):
            nhat = q['customers']  # frozenset
            k = q['my_divisor']
            rhs = q['my_RHS']
            k_inv = 1.0 / k

            q_name = f"{nhat}_{k}_{rhs}"

            nhat_len = nhat_to_info[nhat]["len"]
            relevant_lost_sets = nhat_to_info[nhat]["relevant_lost_sets"]

            # Store RHS
            dict_valid_ineq_name_2_rhs[q_name] = -int(nhat_len * k_inv)

            tmp_dict = {}
            for lost_set in relevant_lost_sets:
                isect_size = len(lost_set & nhat)
                coeff = -int(isect_size * k_inv)
                if coeff >= 0:
                    continue

                for edge in lost_terms_to_edges.get(lost_set, []):
                    tmp_dict[edge] = coeff  # Overwrites if duplicates

            dict_valid_ineq_name_edge_2_coeff[q_name] = tmp_dict

