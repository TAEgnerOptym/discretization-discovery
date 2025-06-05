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
        #self.do_external=OPT['do_external']

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
        #print('create_arcs')
        t1=time.time()
        self.create_arcs()
        my_times['create_arcs']=time.time()-t1
        t1=time.time()
        #print('make all arcs pred back')

        t1=time.time()
        self.make_all_arcs_2_pred_backwards()
        my_times['make_all_arcs_2_pred_backwards']=time.time()-t1
        #input('hold')
        #print('construct_orderings_back')
        t1=time.time()
        self.construct_orderings_back()
        my_times['construct_orderings_back']=time.time()-t1
        #t1=time.time()
        #self.construct_orderings()
        #my_times['construct_orderings']=time.time()-t1
        #print('construct_edge_candidates')

        self.making_edges_component()
        #print('make_uv_2_e')

        self.make_uv_2_e()
        nodes_in_E = {n for edge in self.E for n in edge}
        new_candid_nodes = [n for n in self.my_candid_nodes if n in nodes_in_E]
        self.my_candid_nodes=new_candid_nodes
        t1=time.time()
        #self.construct_edge_candidates()
        #my_times['construct_edge_candidates']=time.time()-t1
        #print('generate_subsets_to_consider')
        t1=time.time()
        self.generate_subsets_to_consider()
        my_times['generate_subsets_to_consider']=time.time()-t1
        t1=time.time()
        #print('generate_SRI')
        self.generate_SRI()
        my_times['generate_SRI']=time.time()-t1
        t1=time.time()
        #print('generate_edge_2_SRI_contrib')
        self.generate_node_slack_2_SRI_contib()
        my_times['generate_edge_2_SRI_contrib']=time.time()-t1
        #t1=time.time()
        #self.generate_self_self_edges()
        #my_times['generate_self_self_edges']=time.time()-t1
        
        #print('my_times')
        #print(my_times)

    def make_uv_2_e(self):

        self.uv_2_E=dict()

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
        return False


    def making_edges_component(self):
        E=[]

        
        for n in self.my_candid_nodes:
             n1_plus_n0=set([n[0]]).union(set(n[1]))
             n1_plus_n0=frozenset(n1_plus_n0)
             for v in self.my_subset_cust-n1_plus_n0:
                if self.OLD_is_allowable_transition(n,v):
                    #[new_node]= self.get_new_set_and_lost(n,v) 
                    new_node=tuple([v,n1_plus_n0])
                    e=tuple([n,new_node])
                    E.append(e)
                    #print(e)
                    #input('---')
        self.E=E
    
   
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
    def construct_orderings_back(self):
    # Group arcs by size of the middle element
        self.arc_2_orderings=dict()

        subset = self.my_subset_cust  # Local alias for speed
        keys = self.arc_2_pred_arcs_backward.keys()
        
        set_internal_internal_by_size=dict()
        for k in range(0,len(subset)+1):
            set_internal_internal_by_size[k]=[]
        for p in keys:
            set_internal_internal_by_size[len(p[1])].append(p)

        for p in set_internal_internal_by_size[0]:
            self.update_back_given_p_no_mid(p)
        
        for k in range(1,len(self.my_subset_cust)):
            for p in set_internal_internal_by_size[k]:
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
        #if len(all_pred_orderings)>0:
        #    print('p')
        #    print(p)
        #    print('all_pred_orderings')
        #    print(all_pred_orderings)
        #    input('- fine but not in nyc 4--')
        self.arc_2_orderings[p] = compute_efficient_frontier(all_pred_orderings)


    def create_arcs(self):


        self.my_arcs=[]

        t2=time.time()
        for i in self.internal_node_candid:

            this_set=i[1]
            v=i[0]
            for u in i[1]:
                new_intermediate=this_set-set([u])
                new_arc=tuple([u,frozenset(new_intermediate),v])

                self.my_arcs.append(new_arc)
        t2=time.time()-t2
        self.my_arcs=set(self.my_arcs)
        
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


    def make_candid_nodes(self):
        my_subset=self.my_subset_cust
        Nc = self.MF.my_VRP.num_cust
        self.Nc=Nc
        my_nodes=[]

        all_poss=power_set(my_subset)
        internal_nodes=[]
        for p in all_poss:
            poss_start=my_subset-set(p)
            for u in poss_start:
                i=tuple([u,frozenset(p)])
                my_nodes.append(i)
                internal_nodes.append(i)
        self.my_candid_nodes=my_nodes
        self.internal_node_candid=internal_nodes
    
    def generate_subsets_to_consider(self):
        my_subsets = set()
        

        for k in range(1, self.OPT['max_SRI_SET_SIZE'] + 1):
            for subset in combinations(self.my_subset_cust, k):
                my_subsets.add(frozenset(subset))

        self.subset_2_make_SRI = my_subsets


    def  generate_SRI(self):
        my_SRI=[]
        max_SRI_size=self.OPT['max_SRI_SET_SIZE']
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

        #print('my_SRI')
        #print(my_SRI)
        #$input('--')
    
    def generate_node_slack_2_SRI_contib(self):

        # Local bindings
        my_candid_nodes = self.my_candid_nodes
        my_SRI = self.my_SRI

        # Precompute n → {n[0]} ∪ n[1]
        dict_n_2_n1_plus_n0 = {
            n: frozenset({n[0]} | n[1]) for n in my_candid_nodes
        }

        self.dict_valid_ineq_name_2_rhs = {}
        self.dict_valid_ineq_name_node_2_coeff = {}

        for q in tqdm(my_SRI, desc='generating SRI contrib'):
            nhat = q['customers']        # frozenset
            k = q['my_divisor']
            rhs = q['my_RHS']
            k_rhs_inv = 1.0 / (k * rhs)
            k_inv=1.0/k
            nhat_len = len(nhat)

            q_name = f"{nhat}_{k}_{rhs}"
            self.dict_valid_ineq_name_2_rhs[q_name] = -int(nhat_len / k)

            # Precompute intersection sizes where meaningful
            intersection_sizes = {
                n: isize
                for n in my_candid_nodes
                if (isize := len(dict_n_2_n1_plus_n0[n] & nhat)) >= k
            }
            ##print(intersection_sizes)
            #input('hold')
            # Only compute floor if above threshold
            tmp_dict = {
                n: -np.floor(isize * k_inv)
                for n, isize in intersection_sizes.items()
            }

            self.dict_valid_ineq_name_node_2_coeff[q_name] = tmp_dict
            #print('self.dict_valid_ineq_name_node_2_coeff[q_name]')
            #print(self.dict_valid_ineq_name_node_2_coeff[q_name])
            #print('self.dict_valid_ineq_name_2_rhs[q_name]')
            #print(self.dict_valid_ineq_name_2_rhs[q_name])
            #print('q')
            #print(q)
            #print('k_rhs_inv')
            #print(k_rhs_inv)
            #input('---')
        #print('self.dict_valid_ineq_name_node_2_coeff[q_name]')
        #print(self.dict_valid_ineq_name_node_2_coeff)
        #print('self.dict_valid_ineq_name_2_rhs')
        #print(self.dict_valid_ineq_name_2_rhs)