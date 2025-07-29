
def iterative_ilp_la(self):


    
        my_VRP=self.full_prob.D['my_VRP']
        D=self.full_prob.D
        print(my_VRP)
        Nc=my_VRP.num_cust
        self.dict_pred_gain=dict()
        [ng_neigh_by_cust_power,junk]=naive_get_LA_neigh(my_VRP,self.full_prob.jy_opt['LAB_MP_neigh_use_power'])
        [ng_neigh_by_cust_all,junk]=naive_get_LA_neigh(my_VRP,self.full_prob.jy_opt['LAB_MP_neigh_use_all'])
        G=set()
        for u in range(0,Nc):

            neighborhood = set(ng_neigh_by_cust_power[u]) | {u}
            for g in power_set(neighborhood):
                if len(g)>1:  # skip empty set and size 1 sets
                    G.add(frozenset(g))
            for i in range(1,len(ng_neigh_by_cust_all[u])):
                g=set(ng_neigh_by_cust_all[u][0:i]).union(set([u]))
                G.add(frozenset(g))
        G.add(frozenset(np.arange(0,Nc)))
        self.G=G

        self.Z_by_group = defaultdict(set)  # g → set of act_u_v
        all_actions=D['action2Cost'].keys()
        u_in_groups = defaultdict(set)
        v_not_in_groups = defaultdict(set)

        for g in G:
            for u in g:
                u_in_groups[u].add(g)
            for v in range(0,Nc+2):  # include depot if needed
                if v not in g:
                    v_not_in_groups[v].add(g)

        # Step 2: Build Z_by_group using set intersection
        num_add=0
        for act in all_actions:
            if act in self.full_prob.delta_name_2_ub and self.full_prob.delta_name_2_ub[act]<0.001:
                #num_already_gone=num_already_gone+1
                continue
            num_add=num_add+1
            _, u_str, v_str = act.split("_")
            u, v = int(u_str), int(v_str)

            relevant_groups = u_in_groups[u] & v_not_in_groups[v]
            for g in relevant_groups:
                self.Z_by_group[g].add(act)

        print('num_add')
        print(num_add)
        input('---')
        costs=D['action2Cost']
        self.cost_val = {
            g: min(costs[act] for act in self.Z_by_group[g]) if self.Z_by_group[g] else float("inf")
            for g in G
        }
        Gall=set([])
        while(True):
            self.filter_constraints()
            self.call_gurobi_milp_solver()
            primal_sol_lp= self.milp_solution
            amount_inside = {
                g: (sum(primal_sol_lp[act] for act in self.Z_by_group[g]))
                for g in G
            }

            frac_amount = {
                g: min(
                    amount_inside[g] - int(amount_inside[g]),
                    np.ceil(amount_inside[g]) - amount_inside[g]
                )
                for g in G
            }
            self.pred_val_gain={
                g: self.cost_val[g]*frac_amount[g]/(amount_inside[g])
                for g in G
            }
        
            pred_val_gain=self.pred_val_gain
            GNew = {
                g for g in GNew
                if pred_val_gain[g] > 0.0
                and amount_inside[g] < 1.999
                and frac_amount[g] > 0.01
                #and len(g)>=2
            }

            num_Keep=4
            GNew = heapq.nlargest(
                num_Keep,
                GNew,
                key=lambda g: pred_val_gain[g]
            )

            for g in GNew:
                if g in Gall:
                    input('error here')
                Gall.add(g)
                con_name_eq='NEW_BOUND_Branch_eq'+str(g)
                my_var_name='fancy_branching_var_'+str(g)
                self.dict_var_name_2_obj[my_var_name]=0
                self.dict_con_name_2_eq[con_name_eq]=0
                self.dict_pred_gain[my_var_name]=1#self.pred_val_gain[g]
                self.dict_var_con_2_lhs_eq[tuple([my_var_name,con_name_eq])]=1#self.delta_con_2_contrib[v_con]
                self.dict_var_name_2_is_integer[my_var_name]=1
                self.full_prob.delta_name_2_lb[my_var_name]=1
                self.full_prob.delta_name_2_ub[my_var_name]=np.inf
                for my_act in self.Z_by_group[g]:
                    self.dict_var_con_2_lhs_eq[tuple([my_act,con_name_eq])]=-1