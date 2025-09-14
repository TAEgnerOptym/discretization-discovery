import itertools
import numpy as np
import networkx as nx
from collections import defaultdict


class ng_graph_fancy_slow:



    def __init__(self,my_instance,ng_neigh_by_cust):
        #print('making ng-graph p1')
        self.my_instance=my_instance
        self.ng_neigh_by_cust=ng_neigh_by_cust
        self.Nc=len(self.ng_neigh_by_cust)
        #print('self.Nc')
        #print(self.Nc)
        #input('---')
        self.my_instance.my_params['DEBUG_NG_turn_off_CLEAN']=False
        #print('making ng-graph p2')

        self.make_nodes()
        #print('making ng-graph p3')

        #self.make_edges_slow()
        self.make_edges_fast()
        #print('making ng-graph p4')

        self.compute_earliest_by_node()
        #print('making ng-graph p5')

        self.compute_earliest_by_edge()
        #print('making ng-graph p6')

        self.eval_edges_keep()
        #print('making ng-graph p7')

        self.clean_order()
        #print('making ng-graph DONE')



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
                            #self.add_sucessors_slow(my_node)
                            self.add_successors_fast(my_node)
                        if u!=v and np.inf>self.my_instance.dist_mat_full[u,v]:
                            #self.add_non_self_succ_slow(v,my_node)
                            self.add_non_self_succ_fast(v,my_node)

    def add_successors_fast(self, my_node):
        # Unpack & ensure types
        u, Ni_fs = my_node
        if not isinstance(Ni_fs, frozenset):
            Ni_fs = frozenset(Ni_fs)  # normalize once

        # ---- lazy caches ----
        # neighbor sets for O(1) diff
        if not hasattr(self, "_neigh_set") or self._neigh_set is None:
            # ensure every entry is a set
            #self._neigh_set = [set(nb) for nb in self.ng_neigh_by_cust]
            self._neigh_set = {u: set(nb) for u, nb in self.ng_neigh_by_cust.items()}
        neigh_set_u = self._neigh_set[u]

        # cache for frozenset unions: (id(Ni_fs), w) -> frozenset(Ni_fs ∪ {w})
        if not hasattr(self, "_Ni_add_cache") or self._Ni_add_cache is None:
            self._Ni_add_cache = {}
        Ni_add_cache = self._Ni_add_cache
        key_Ni = id(Ni_fs)

        # ---- candidates (neighbors not already in Ni) ----
        candidates = neigh_set_u.difference(Ni_fs)
        if not candidates:
            return

        # ---- build edges in one go ----
        Ni_tuple = tuple(Ni_fs)  # reuse in tuple packing; order irrelevant for frozenset

        # local bindings (micro-opts)
        bit_union = Ni_add_cache.get
        E_extend = self.E.extend

        edges = []
        for w in candidates:
            key = (key_Ni, w)
            new_set = bit_union(key)
            if new_set is None:
                # fast frozenset construction without intermediate set copies
                new_set = frozenset((*Ni_tuple, w))
                Ni_add_cache[key] = new_set
            new_node = (u, new_set)
            edges.append((my_node, new_node))

        E_extend(edges)


    def add_non_self_succ_fast(self, v, my_node):
        u, N_i = my_node

        # ---- quick rejects ----
        if v == u:
            return  # or raise, but don't block with input()
        if v in N_i:
            return

        # ---- one-time caches / normalizations ----
        # neighbors as sets
        if not hasattr(self, "_neigh_set") or self._neigh_set is None:
            # handle list-like OR dict-like ng_neigh_by_cust
            if isinstance(self.ng_neigh_by_cust, dict):
                self._neigh_set = {k: set(nb) for k, nb in self.ng_neigh_by_cust.items()}
            else:
                self._neigh_set = [set(nb) for nb in self.ng_neigh_by_cust]
        neigh_v = self._neigh_set[v]

        # ensure non-neighbors are sets
        if not hasattr(self, "_non_neigh_set") or self._non_neigh_set is None:
            if isinstance(self.non_neigh_cust, dict):
                self._non_neigh_set = {k: set(s) for k, s in self.non_neigh_cust.items()}
            else:
                self._non_neigh_set = [set(s) for s in self.non_neigh_cust]
        N_minus_nv = self._non_neigh_set[v]

        # ensure node_2_ng_allowed entries are sets
        if not hasattr(self, "_node_2_ng_allowed_set") or self._node_2_ng_allowed_set is None:
            self._node_2_ng_allowed_set = {}
        allowed_cache = self._node_2_ng_allowed_set
        try:
            N_u_minus_Ni = allowed_cache[my_node]
        except KeyError:
            N_u_minus_Ni = set(self.node_2_ng_allowed[my_node])
            allowed_cache[my_node] = N_u_minus_Ni

        # ---- compute N_j = (N_i ∩ neigh[v]) ∪ ({u} if u∈neigh[v]) ----
        # Use set ops directly; avoids copies of N_i and repeated conversions.
        base = neigh_v.intersection(N_i)       # set
        if u in neigh_v:
            base.add(u)
        N_j_fs = frozenset(base)

        new_node = (v, N_j_fs)
        new_edge = (my_node, new_node)

        # ---- final feasibility: empty intersection? use isdisjoint (faster) ----
        if N_minus_nv.isdisjoint(N_u_minus_Ni):
            self.E.append(new_edge)


    def add_sucessors_slow(self,my_node):
        u=my_node[0]
        N_i=my_node[1]
        N_i=set(N_i)
        for w in self.ng_neigh_by_cust[u]:
            if w in N_i:
                continue
            new_set=N_i.copy()
            new_set.add(w)
            new_set=frozenset(new_set)
            #new_set=#set(sorted(list(new_set)))
            new_node=tuple([u,new_set])
            #if new_node not in self.my_nodes[u]:
            #    print('self.node_list[u]')
            #    print(self.node_list[u])
            #    print('new_node')
            #    print(new_node)
            #    input('error here')
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
        #N_j=sorted(N_j)
        N_j=set(N_j)
        N_j=N_j.intersection(self.ng_neigh_by_cust[v])
        N_j=frozenset(N_j)
        new_node=tuple([v,N_j])
        
        N_minus_nv=self.non_neigh_cust[v]
        N_u_minus_Ni=self.node_2_ng_allowed[my_node]
        new_edge=tuple([my_node,new_node])

        if len(N_minus_nv.intersection(N_u_minus_Ni))==0:
            self.E.append(new_edge)
        

    def make_nodes(self):
        
        self.my_nodes=[]
        self.my_nodes=dict()
        self.source_node=tuple([self.Nc,frozenset([])])
        self.sink_node=tuple([self.Nc+1,frozenset([])])
        self.my_nodes[self.Nc]=[self.source_node]
        self.my_nodes[self.Nc+1]=[self.sink_node]
        self.ng_neigh_by_cust.append(set([]))
        self.ng_neigh_by_cust.append(set([]))
        ng_neigh_by_cust_dict=dict()
        for u in range(0,self.Nc+2):
            ng_neigh_by_cust_dict[u]=set(self.ng_neigh_by_cust[u])
        self.ng_neigh_by_cust=ng_neigh_by_cust_dict
        self.my_power_set=dict()
        self.non_neigh_cust=dict()
        self.non_neigh_cust[self.Nc]=set(np.arange(0,self.Nc))
        self.non_neigh_cust[self.Nc+1]=set(np.arange(0,self.Nc))
        self.node_2_ng_allowed=dict()
        self.node_2_ng_allowed[self.source_node]=frozenset([])
        self.node_2_ng_allowed[self.sink_node]=frozenset([])
        for u in range(0,self.Nc):
            my_set=set(np.arange(0,self.Nc))
            my_set.remove(u)
            #print('u')
            #print(u)
            #print('self.Nc')
            #print(self.Nc)
            #print('--')
            #if self.Nc!=25:
            #    input('woong')
            #print('my_set')
            #print(my_set)
            #if len(my_set)!=self.Nc-1:
            #    input('HUH')
            #print('self.ng_neigh_by_cust[u]')
            #print(self.ng_neigh_by_cust[u])
            for w in self.ng_neigh_by_cust[u]:
                my_set.remove(w)
            self.non_neigh_cust[u]=my_set
        for u in range(0,self.Nc):
            self.my_power_set[u]=self.power_set(self.ng_neigh_by_cust[u])
            self.my_nodes[u]=[]
            for my_sub in self.my_power_set[u]:
                my_sub=frozenset(my_sub)
                my_new_node=tuple([u,my_sub])
                self.my_nodes[u].append(my_new_node)
                self.node_2_ng_allowed[my_new_node]=set(self.ng_neigh_by_cust[u])-set(my_sub)
        
        self.all_nodes_exc_source_sink=[]
        self.all_nodes_by_sz_exc_source_sink=defaultdict(list)
            
        for u in range(0,self.Nc):
            for i in self.my_nodes[u]:
                
                self.all_nodes_exc_source_sink.append(i)
                sz_N_i=len(i[1])
                self.all_nodes_by_sz_exc_source_sink[sz_N_i].append(i)

    def compute_early_depart_from_v_given_pred_u(self,time_u,u,v):
        dist_u_v=self.dist_serve_add=self.my_instance.dist_mat_full[u,v]

        if dist_u_v==np.inf:
            print('[u,v]')
            print([u,v])
            input('error should not reach here')
        ES_v=self.my_instance.early_start[v]
        LS_v=self.my_instance.late_start[v]
        ES_u=self.my_instance.early_start[u]
        LS_u=self.my_instance.late_start[u]

        arrival_time_v=time_u-dist_u_v
        my_out_time=arrival_time_v
        if arrival_time_v>ES_v:
            
            my_out_time=ES_v
        if my_out_time<LS_v:
            my_out_time=-np.inf
        #print('my_out_time')
        #print(my_out_time)
        #print('[time_u,u,v]')
        #print([time_u,u,v])
        #print('dist_u_v')
        #print(dist_u_v)
        #print('ES_v,LS_v,ES_u,LS_u')
        #print(ES_v,LS_v,ES_u,LS_u)

        #input('--')
        return my_out_time
    
    def compute_earliest_by_node(self):
        import numpy as np
        from collections import defaultdict

        # ---------- locals & fast paths ----------
        d2e = defaultdict(lambda: -np.inf)               # self.dict_node_2_early
        d2e_pred = defaultdict(lambda: -np.inf)          # self.dict_node_early_j_given_pred_i (kept for parity)
        self.dict_node_2_early = d2e
        self.dict_node_early_j_given_pred_i = d2e_pred

        dist = self.my_instance.dist_mat_full            # shape (Nc, Nc)
        ES   = self.my_instance.early_start              # shape (Nc,)
        LS   = self.my_instance.late_start               # shape (Nc,)
        dem  = self.my_instance.dem_full                 # shape (Nc,)
        veh_cap = self.my_instance.vehicle_capacity
        Nc = self.Nc

        # ---------- bitmask helpers (build once) ----------
        # map cust -> bit (0..Nc-1)
        if not hasattr(self, "_bit_for_cust") or self._bit_for_cust is None:
            self._bit_for_cust = np.array([1 << c for c in range(Nc)], dtype=np.uint64)
        bit_for = self._bit_for_cust

        # neigh masks: nm[w] has 1-bits for all neighbors of w
        if not hasattr(self, "_neigh_mask") or self._neigh_mask is None:
            nm = np.zeros(Nc, dtype=np.uint64)
            for w in range(Nc):
                m = 0
                for c in self.ng_neigh_by_cust[w]:
                    m |= bit_for[c]
                nm[w] = m
            self._neigh_mask = nm
        nm = self._neigh_mask

        # fast “isfinite row mask” cache for dist
        if not hasattr(self, "_finite_mask") or self._finite_mask is None:
            self._finite_mask = np.isfinite(dist)
        finite_mask = self._finite_mask

        # valid node existence “my_nodes[w]” membership → set for O(1)
        # Expect my_nodes[w] is an iterable of tuples j = (w, frozenset(Ni_plus_u))
        if not hasattr(self, "_my_nodes_sets") or self._my_nodes_sets is None:
            my_nodes_sets = []
            for w in range(Nc):
                # store as set for O(1) membership
                my_nodes_sets.append(set(self.my_nodes[w]))
            self._my_nodes_sets = my_nodes_sets
        my_nodes_sets = self._my_nodes_sets

        # ---------- initialize source layer ----------
        # all_nodes_by_sz_exc_source_sink[0] contains tuples i=(u, frozenset(Ni))
        for i in self.all_nodes_by_sz_exc_source_sink[0]:
            u = i[0]
            d2e[i] = ES[u]

        max_sz = max(self.all_nodes_by_sz_exc_source_sink.keys())
        ES_arr = ES  # alias

        # small helpers
        def mask_of_set(fs):
            """frozenset of ints -> uint64 bitmask"""
            m = np.uint64(0)
            for c in fs:
                m |= bit_for[c]
            return m

        # Optional cache for demand sums of a frozenset
        if not hasattr(self, "_dem_sum_cache") or self._dem_sum_cache is None:
            self._dem_sum_cache = {}
        dem_sum_cache = self._dem_sum_cache

        # ---------- main DP over size ----------
        for k in range(0, max_sz):
            nodes_k = self.all_nodes_by_sz_exc_source_sink[k]
            for i in nodes_k:
                u, Ni_fs = i[0], i[1]
                time_u = d2e[i]
                if time_u < -0.5:  # your sentinel skip
                    continue

                # demand sum of Ni (cached)
                key_Ni = id(Ni_fs)  # frozenset is immutable; id() is stable in this run
                tot_dem_Ni = dem_sum_cache.get(key_Ni)
                if tot_dem_Ni is None:
                    s = 0.0
                    for w_ in Ni_fs:
                        s += dem[w_]
                    dem_sum_cache[key_Ni] = s
                    tot_dem_Ni = s

                tot_dem_i = dem[u] + tot_dem_Ni
                cap_remaining = veh_cap - tot_dem_i
                if cap_remaining <= 0:
                    continue

                # bitmask for Ni; cached by id as well
                if not hasattr(self, "_Ni_mask_cache"):
                    self._Ni_mask_cache = {}
                Ni_mask_cache = self._Ni_mask_cache
                Ni_mask = Ni_mask_cache.get(key_Ni)
                if Ni_mask is None:
                    Ni_mask = mask_of_set(Ni_fs)
                    Ni_mask_cache[key_Ni] = Ni_mask

                # vector prefilter for candidate w:
                #  1) dist[u,w] finite
                #  2) w not in Ni
                #  3) dem[w] <= cap_remaining
                #  4) u ∈ neigh[w]
                #  5) Ni ⊆ neigh[w]  <=> (nm[w] & Ni_mask) == Ni_mask
                # We’ll build boolean mask then take indices; reduces the Python loop size massively.
                fin_row = finite_mask[u]                              # bool array shape (Nc,)
                not_in_Ni = ( (Ni_mask & bit_for) == 0 )              # vector: for each w, is w ∉ Ni?
                ok_dem = dem <= cap_remaining
                u_bit = bit_for[u]
                has_u = (nm & u_bit) == u_bit
                superset_Ni = (nm & Ni_mask) == Ni_mask

                cand_mask = fin_row & not_in_Ni & ok_dem & has_u & superset_Ni
                if not np.any(cand_mask):
                    continue

                cand_ws = np.nonzero(cand_mask)[0]  # candidate w indices

                # build set_NI_plus_u once (as frozenset)
                # (Note: Ni does NOT include u per your code; we add it.)
                if not hasattr(self, "_Ni_plus_u_cache"):
                    self._Ni_plus_u_cache = {}
                Ni_plus_u_cache = self._Ni_plus_u_cache
                key_plus = (key_Ni, u)
                set_add_u = Ni_plus_u_cache.get(key_plus)
                if set_add_u is None:
                    # frozenset union (reusing fs avoids new set allocs in loop)
                    # Ni_fs | {u}
                    set_add_u = frozenset((*Ni_fs, u))
                    Ni_plus_u_cache[key_plus] = set_add_u

                # Tight locals
                d2e_local = d2e
                dist_row = dist[u]
                ES_local = ES_arr
                LS_local = LS
                time_u_local = time_u
                my_nodes_for_w = my_nodes_sets  # list of sets

                # now iterate only the filtered candidates
                for w in cand_ws:
                    j = (w, set_add_u)
                    if j not in my_nodes_for_w[w]:
                        continue

                    dist_u_w = dist_row[w]
                    arrival_time_w = time_u_local - dist_u_w
                    candid = arrival_time_w
                    if candid > ES_local[w]:
                        candid = ES_local[w]
                    if candid < LS_local[w]:
                        candid = -np.inf

                    # max update
                    if candid > d2e_local[j]:
                        d2e_local[j] = candid

    def compute_earliest_by_node(self):
        import numpy as np
        from collections import defaultdict

        # ---------- locals ----------
        d2e = defaultdict(lambda: -np.inf)
        d2e_pred = defaultdict(lambda: -np.inf)
        self.dict_node_2_early = d2e
        self.dict_node_early_j_given_pred_i = d2e_pred

        
        veh_cap = self.my_instance.vehicle_capacity
        Nc = self.Nc
        dist = self.my_instance.dist_mat_full[:Nc,:Nc]  # (Nc, Nc) float, np.inf for forbidden
        ES   = self.my_instance.early_start[:Nc]    # (Nc,)
        LS   = self.my_instance.late_start[:Nc]     # (Nc,)
        dem  = self.my_instance.dem_full[:Nc]     # (Nc,)

        # ---------- pure-Python bit table (no NumPy dtypes) ----------
        # bit_for[w] = 1 << w (Python int)
        if not hasattr(self, "_bit_for_cust") or self._bit_for_cust is None:
            self._bit_for_cust = [1 << c for c in range(Nc)]
        bit_for = self._bit_for_cust

        # neighbor masks nm[w]: Python int with bits of neighbors of w
        if not hasattr(self, "_neigh_mask") or self._neigh_mask is None:
            nm = [0] * Nc
            for w in range(Nc):
                m = 0
                for c in self.ng_neigh_by_cust[w]:
                    m |= bit_for[c]
                nm[w] = m
            self._neigh_mask = nm
        nm = self._neigh_mask  # list[int], len Nc

        # cache finite distances row-wise
        if not hasattr(self, "_finite_mask") or self._finite_mask is None:
            self._finite_mask = np.isfinite(dist)
        finite_mask = self._finite_mask

        # speed up membership: my_nodes[w] -> set
        if not hasattr(self, "_my_nodes_sets") or self._my_nodes_sets is None:
            self._my_nodes_sets = [set(self.my_nodes[w]) for w in range(Nc)]
        my_nodes_sets = self._my_nodes_sets

        # demand-sum and mask caches for frozensets
        if not hasattr(self, "_dem_sum_cache") or self._dem_sum_cache is None:
            self._dem_sum_cache = {}
        dem_sum_cache = self._dem_sum_cache

        if not hasattr(self, "_Ni_mask_cache") or self._Ni_mask_cache is None:
            self._Ni_mask_cache = {}
        Ni_mask_cache = self._Ni_mask_cache

        if not hasattr(self, "_Ni_plus_u_cache") or self._Ni_plus_u_cache is None:
            self._Ni_plus_u_cache = {}
        Ni_plus_u_cache = self._Ni_plus_u_cache

        # ---------- init layer 0 ----------
        for i in self.all_nodes_by_sz_exc_source_sink[0]:
            u = i[0]
            d2e[i] = ES[u]

        max_sz = max(self.all_nodes_by_sz_exc_source_sink.keys())

        # ---------- main DP ----------
        for k in range(0, max_sz):
            for i in self.all_nodes_by_sz_exc_source_sink[k]:
                u, Ni_fs = i
                time_u = d2e[i]
                if time_u < -0.5:
                    continue

                # demand(Ni) cached by id of frozenset
                key_Ni = id(Ni_fs)
                tot_dem_Ni = dem_sum_cache.get(key_Ni)
                if tot_dem_Ni is None:
                    s = 0.0
                    for w_ in Ni_fs:
                        s += dem[w_]
                    dem_sum_cache[key_Ni] = s
                    tot_dem_Ni = s

                tot_dem_i = dem[u] + tot_dem_Ni
                cap_remaining = veh_cap - tot_dem_i
                if cap_remaining <= 0:
                    continue

                # mask(Ni) cached (Python int bits)
                Ni_mask = Ni_mask_cache.get(key_Ni)
                if Ni_mask is None:
                    m = 0
                    for c in Ni_fs:
                        m |= bit_for[c]
                    Ni_mask = m
                    Ni_mask_cache[key_Ni] = Ni_mask

                # precompute frozenset(Ni ∪ {u})
                key_plus = (key_Ni, u)
                set_add_u = Ni_plus_u_cache.get(key_plus)
                if set_add_u is None:
                    set_add_u = frozenset((*Ni_fs, u))
                    Ni_plus_u_cache[key_plus] = set_add_u

                # --------- bulk candidate prefilter (makes loop small) ----------
                fin_row = finite_mask[u]                      # np.bool_(Nc,)
                ok_dem  = (dem <= cap_remaining)              # np.bool_(Nc,)

                u_bit = bit_for[u]
                # Build boolean arrays from Python-int masks (Nc ~ 100 -> fine)
                not_in_Ni   = np.fromiter(((Ni_mask & bit_for[w]) == 0 for w in range(Nc)), dtype=bool)
                has_u       = np.fromiter((((nm[w] & u_bit) != 0) for w in range(Nc)), dtype=bool)
                superset_Ni = np.fromiter((((nm[w] & Ni_mask) == Ni_mask) for w in range(Nc)), dtype=bool)

                cand_mask = fin_row & not_in_Ni & ok_dem & has_u & superset_Ni
                if not np.any(cand_mask):
                    continue

                cand_ws = np.nonzero(cand_mask)[0]

                # --------- tight inner loop over filtered candidates ----------
                d2e_local = d2e
                dist_row  = dist[u]
                ES_local  = ES
                LS_local  = LS
                my_nodes_for_w = my_nodes_sets
                time_u_local = time_u

                for w in cand_ws:
                    j = (w, set_add_u)
                    if j not in my_nodes_for_w[w]:
                        continue

                    dist_u_w = dist_row[w]
                    arrival_time_w = time_u_local - dist_u_w
                    candid = arrival_time_w if arrival_time_w <= ES_local[w] else ES_local[w]
                    if candid < LS_local[w]:
                        candid = -np.inf

                    if candid > d2e_local[j]:
                        d2e_local[j] = candid

    def compute_earliest_by_edge(self):
        self.dict_node_early_j_given_pred_i=defaultdict(lambda: -np.inf)
        #print('----')
        #print('----')
        #print('----')
        #print('----')
        #print('dict_node_2_early')
        #print(self.dict_node_2_early)
        #input('--')
        dist=self.my_instance.dist_mat_full
        for i in self.all_nodes_exc_source_sink:
            time_u=self.dict_node_2_early[i]
            Ni=i[1]
            u=i[0]
            if time_u<0:
                #if len(Ni)<2 and dist:
                #    print('i')
                #    print(i)
                #    input('definitely wrong')
#
                continue
            for w in range(0,self.Nc):
                #print('*********')
                if w in Ni:
                    continue
                if w==u:
                    continue
                ## pred_cust in Ni:
                if len(Ni-set(self.ng_neigh_by_cust[w]))>0:
                    continue
                if dist[u,w]==np.inf:
                    continue
                #print('i')
                #print(i)
                #print('w')
                #print(w)
                #input('found one ')
                j=[]
                if u in self.ng_neigh_by_cust[w]:
                    #print('pt1')
                    my_tm_set=set(Ni)
                    #print('my_tm_set init')
                    #print(my_tm_set)
                    my_tm_set=my_tm_set.union(set([u]))
                    #print('my_tm_set')
                    #print(my_tm_set)
                    my_tm_set=frozenset(my_tm_set)
                    #print('my_tm_set 1')
                    #print(my_tm_set)
                    j=tuple([w,my_tm_set])
                else:
                    #print('pt2')

                    j=tuple([w,Ni])

                self.dict_node_early_j_given_pred_i[(i,j)]=self.compute_early_depart_from_v_given_pred_u(time_u,u,w)
                #if self.dict_node_early_j_given_pred_i[(i,j)]<0:
#
#                    print('i')
#                    print(i)
#                    print('j')
#                    print(j)
#                    input('-not wrong but debuggin-')

    def eval_edges_keep(self):
        E_after_removal=[]
        self.dict_e_2_i_hat=dict()
        for e in self.E:
            i=e[0]
            j=e[1]
            u=i[0]
            Ni=i[1]
            v=j[0]
            Nj=j[1]
            self.dict_e_2_i_hat[(i,j)]=None
            if u==v or u==self.Nc or v==self.Nc+1 or len(Ni)==0:
                E_after_removal.append(e)
                continue
            if u in self.ng_neigh_by_cust[v]:
                Nhati=set(Nj)-set([u])
                Nhati=frozenset(Nhati)
            else:
                Nhati=Nj
            ihat=tuple([u,Nhati])
            in_tup=tuple([ihat,j])
            
            my_arrival_time=self.dict_node_early_j_given_pred_i[in_tup]
            self.dict_e_2_i_hat[(i,j)]=ihat
            if my_arrival_time>=-0.5:
                E_after_removal.append(e)
            
        self.DEBUG_E_prior_removal=self.E.copy()
        self.E=E_after_removal
    def clean_order(self):
        self.node_list=dict()
        self.DEBUG_ez_lookup_e_2_ihat=dict()
        for u in self.my_nodes:
            new_nodes=[]
            for n in self.my_nodes[u]:
                tmp=sorted(list(n[1]))
                this_node=[n[0],tmp]
                new_nodes.append(this_node)
            self.node_list[u]=new_nodes
        E_2=[]
        self.DEBUG_E_after_removal=self.E.copy()
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
            help1=tuple([e[0][0],frozenset(e[0][1])])
            help2=tuple([e[1][0],frozenset(e[1][1])])
            self.DEBUG_ez_lookup_e_2_ihat[tuple([help1,help2])]=self.dict_e_2_i_hat[e]
            #self.DEBUG_e_to_str_ver[e]=this_new_edge
        self.E=E_2

    def make_edges_fast(self):
        import numpy as np

        E = []
        self.E = E  # keep same attribute

        Nc = self.Nc
        dist = self.my_instance.dist_mat_full

        # ---------- one-time caches (no hasattr checks in hot path) ----------
        # neighbors as sets
        if isinstance(self.ng_neigh_by_cust, dict):
            neigh_set = {k: set(nb) for k, nb in self.ng_neigh_by_cust.items()}
        else:
            neigh_set = [set(nb) for nb in self.ng_neigh_by_cust]

        # non-neighbors as sets
        if isinstance(self.non_neigh_cust, dict):
            non_neigh_set = {k: set(s) for k, s in self.non_neigh_cust.items()}
        else:
            non_neigh_set = [set(s) for s in self.non_neigh_cust]

        # per-u list of v with finite distance and v != u  (vectorized)
        finite = np.isfinite(dist)
        vs_by_u = [np.flatnonzero(finite[u] & (np.arange(dist.shape[1]) != u)) for u in range(dist.shape[0])]

        # cache for unions (Ni ∪ {w}) and (N_i ∩ neigh[v]) ∪ {u?}
        Ni_add_cache = {}
        Ni_inter_neigh_cache = {}

        # ---------- local helpers (no hasattr) ----------
        def add_successors_core(my_node):
            u, Ni_fs = my_node
            if not isinstance(Ni_fs, frozenset):
                Ni_fs = frozenset(Ni_fs)

            candidates = neigh_set[u].difference(Ni_fs)
            if not candidates:
                return

            Ni_tuple = tuple(Ni_fs)
            key_Ni = id(Ni_fs)

            edges_local = []
            get_union = Ni_add_cache.get
            for w in candidates:
                key = (key_Ni, w)
                new_set = get_union(key)
                if new_set is None:
                    new_set = frozenset((*Ni_tuple, w))
                    Ni_add_cache[key] = new_set
                new_node = (u, new_set)
                edges_local.append((my_node, new_node))
            E.extend(edges_local)

        def add_non_self_succ_core(v, my_node, N_u_minus_Ni):
            u, N_i = my_node
            # quick rejects
            if v == u or (v in N_i):
                return

            # N_j = (N_i ∩ neigh[v]) ∪ ({u} if u∈neigh[v])
            key = (id(N_i), v, u)
            N_j_fs = Ni_inter_neigh_cache.get(key)
            if N_j_fs is None:
                base = neigh_set[v].intersection(N_i)
                if u in neigh_set[v]:
                    base.add(u)
                N_j_fs = frozenset(base)
                Ni_inter_neigh_cache[key] = N_j_fs

            if non_neigh_set[v].isdisjoint(N_u_minus_Ni):
                new_node = (v, N_j_fs)
                E.append((my_node, new_node))

        # ---------- main ----------
        # SELF-successors: for u < Nc, once per my_node
        for u in range(Nc):  # only 0..Nc-1
            for my_node in self.my_nodes[u]:
                add_successors_core(my_node)

        # NON-SELF successors: for each u, loop its feasible v-set only
        for u in range(dist.shape[0]):
            vs = vs_by_u[u]
            if vs.size == 0:
                continue
            my_nodes_u = self.my_nodes[u]
            if not my_nodes_u:
                continue

            # pre-bind row invariants
            for my_node in my_nodes_u:
                # cache allowed set for this node once
                # (store back so next time the same node is reused we skip conversion)
                allowed = self.node_2_ng_allowed.get(my_node)
                if not isinstance(allowed, set):
                    allowed = set(allowed)
                    # If you want to persist the set for later calls:
                    self.node_2_ng_allowed[my_node] = allowed

                # iterate only v with finite dist and v != u
                for v in vs:
                    add_non_self_succ_core(v, my_node, allowed)
