
class New_local_graph_based_separ:

    #def __init__(self,E,uv_2_E,Nodes,non_source_sink_nodes,dict_valid_ineq_name_2_rhs,dict_valid_ineq_name_edge_2_coeff,source_node,sink_node,my_NG):
    def __init__(self,my_NG):
        self.my_NG=my_NG

        self.E=self.my_NG.E
        self.uv_2_E=self.my_NG.uv_2_E
        self.Nodes=self.my_NG.nodes
        self.non_source_sink_nodes=self.my_NG.non_source_sink_nodes
        self.dict_valid_ineq_name_2_rhs=self.my_NG.dict_valid_ineq_name_2_rhs
        self.dict_valid_ineq_name_edge_2_coeff=self.my_NG.dict_valid_ineq_name_edge_2_coeff
        self.source_node=self.my_NG.source_node
        self.sink_node=self.my_NG.sink_node
        #self.E=E
        #self.uv_2_E=uv_2_E
        #self.Nodes=Nodes
        #self.source_node=source_node
        #self.sink_node=sink_node
        #self.dict_valid_ineq_name_2_rhs=dict_valid_ineq_name_2_rhs
        #self.dict_valid_ineq_name_edge_2_coeff=dict_valid_ineq_name_edge_2_coeff
        #self.non_source_sink_nodes=non_source_sink_nodes
        self.dict_var_name_2_obj=dict()
        self.dict_var_con_2_lhs_exog=dict()
        self.dict_var_con_2_lhs_eq=dict()
        self.dict_con_name_2_LB=dict()
        self.dict_con_name_2_eq=dict()
        #print('making vars')
        self.make_vars()
        #print('making make_flow_in_out')

        self.make_flow_in_out()
        #print('making match')
        self.make_match()
        #print('making valid')
        self.make_valid_ineq()
        #print('dont graph part')
    def make_vars(self):
        #edge_vars
        self.dict_var_2_obj=dict()
        for e in self.E:
            var_name='EDGE_VAR_'+str(e)
            self.dict_var_2_obj[var_name]=0
        
       #self.non_source_sink_nodes=self.Nodes-set([self.source_node,self.sink_node])
        for n in self.non_source_sink_nodes:
            var_name='SLACK_FLOW_'+str(n)

            self.dict_var_2_obj[var_name]=1
        
        for q in self.dict_valid_ineq_name_2_rhs:
            var_name='SLACK_VALID_'+str(q)
            self.dict_var_2_obj[var_name]=1

    def make_valid_ineq(self):
        # Precompute edge variable names once
        if not hasattr(self, 'edge_to_varname'):
            self.edge_to_varname = {
                edge: 'EDGE_VAR_' + str(edge)
                for edge in self.my_NG.E_2_lost_terms
            }

        d_lhs = self.dict_var_con_2_lhs_exog
        d_lb = self.dict_con_name_2_LB
        d_rhs = self.dict_valid_ineq_name_2_rhs
        d_edge_coeff = self.dict_valid_ineq_name_edge_2_coeff

        edge_to_varname = self.edge_to_varname  # local binding
        update_lhs = d_lhs.update
        append_entry = lambda v, c, coeff: ((v, c), coeff)

        for q_name in tqdm(d_rhs, desc='generating VALID INEQ'):
            con_name = f'Valid_ineq_{q_name}'
            slack_var_name = f'SLACK_VALID_{q_name}'

            d_lb[con_name] = d_rhs[q_name] - 0.0001

            entries = [append_entry(slack_var_name, con_name, 1)]
            
            edge_coeffs = d_edge_coeff[q_name]
            entries += [
                append_entry(edge_to_varname[edge], con_name, coeff)
                for edge, coeff in edge_coeffs.items()
            ]

            update_lhs(entries)


    def OLD_make_valid_ineq(self):
        for q in self.dict_valid_ineq_name_2_rhs:
            con_name='Valid_ineq_'+str(q)
            self.dict_con_name_2_LB[con_name]=self.dict_valid_ineq_name_2_rhs[q]-0.0001
            slack_var_name='SLACK_VALID_'+str(q)
            self.dict_var_con_2_lhs_exog[tuple([slack_var_name,con_name])]=1
            for e in self.dict_valid_ineq_name_edge_2_coeff[q]:
                var_name='EDGE_VAR_'+str(e)
                coeff=self.dict_valid_ineq_name_edge_2_coeff[q][e]
                if coeff>0:
                    input('error')
                self.dict_var_con_2_lhs_exog[tuple([var_name,con_name])]=coeff
    def make_match(self):
        print('making match constraints')
        for uv in self.uv_2_E:
            if uv[0]==uv[1]:
                continue
            con_name='match_EQ_'+str(uv)
            self.dict_con_name_2_eq[con_name]=0

            for e in self.uv_2_E[uv]:
                var_name='EDGE_VAR_'+str(e)
                my_tup_1=tuple([var_name,con_name])
                self.dict_var_con_2_lhs_eq[my_tup_1]=1



    def make_flow_in_out(self):
        for i in self.non_source_sink_nodes:
            con_name='flow_in_out_'+str(i)
            self.dict_con_name_2_LB[con_name]=0
            var_name='SLACK_FLOW_'+str(i)
            my_tup_slack=tuple([var_name,con_name])
            self.dict_var_con_2_lhs_exog[my_tup_slack]=1
        
        for e in self.E:
            i=e[0]
            j=e[1]
            var_name='EDGE_VAR_'+str(e)

            if i!=self.source_node:
                con_name='flow_in_out_'+str(i)
                my_tup_1=tuple([var_name,con_name])
                self.dict_var_con_2_lhs_exog[my_tup_1]=1
            if j!=self.sink_node:
                con_name='flow_in_out_'+str(j)
                my_tup_2=tuple([var_name,con_name])
                self.dict_var_con_2_lhs_exog[my_tup_2]=-1
