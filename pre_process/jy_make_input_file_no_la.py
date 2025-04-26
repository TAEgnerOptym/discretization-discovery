import numpy as np

class jy_make_input_file_no_la:

    def __init__(self,my_instance,my_dem_graph,my_time_graph,num_terms_per_bin,ngGraph):
        
        self.my_instance=my_instance
        self.jy_opt=my_instance.my_params
        
        #self.my_instance.num_cust=int(self.my_instance.num_cust)
        self.my_dem_graph=my_dem_graph
        self.my_time_graph=my_time_graph
        if self.jy_opt['use_ng']>0.5:
            self.ngGraph=ngGraph

        self.out_dict=dict()
        self.num_terms_per_bin=num_terms_per_bin
        self.make_all_delta()
        self.make_graph_names()
        self.make_h_2_source_sink_id()
        self.make_nodes()
        self.make_allActions()
        self.make_exog_name_RHS()
        self.make_all_integ_con()
        self.make_hIJP()
        self.make_contrib()
    

    def make_hIJP(self):
        self.hijP=dict()
        self.hijP['timeGraph']=dict()
        self.hijP['capGraph']=dict()
        if self.jy_opt['use_ng']>0.5:
            self.hijP['ngGraph']=dict()

        for ij in  self.my_dem_graph.E:
            i=ij[0]
            j=ij[1]
            u=i[0]
            v=j[0]
            x_name=[]
            if u!=v:
                x_name='act_'+str(u)+'_'+str(v)
            else:
                x_name='null_action'
            self.hijP['capGraph'][str(ij)]=[x_name]
        for ij in  self.my_time_graph.E:
            i=ij[0]
            j=ij[1]
            u=i[0]
            v=j[0]
            x_name=[]
            if u!=v:
                x_name='act_'+str(u)+'_'+str(v)
            else:
                x_name='null_action'
            self.hijP['timeGraph'][str(ij)]=[x_name]
        self.out_dict['hij2P']=self.hijP
        if self.jy_opt['use_ng']>0.5:
            for ij in  self.ngGraph.E:
                i=ij[0]
                j=ij[1]
                u=i[0]
                v=j[0]
                x_name=[]
                if u!=v:
                    x_name='act_'+str(u)+'_'+str(v)
                    #print(u)
                    #print(v)
                    #input('---')
                else:
                    x_name='null_action'
                if x_name not in self.all_actions and x_name!=self.null_action:
                    print('x_name')
                    print(x_name)
                    input('err') 
                self.hijP['ngGraph'][str(ij)]=[x_name]

    def make_all_integ_con(self):
        #print('hi')
        self.all_integ_con=[]
        self.primIntegCon2Contrib=dict()
        self.actionIntegCon2Contrib=dict()
        for uv in self.all_uv_edges:
            u=uv[0]
            v=uv[1]
            #if u<self.my_instance.num_cust and v<self.my_instance.num_cust:
            con_name='con_psi'+str(u)+'_'+str(v)
            x_name='act_'+str(u)+'_'+str(v)
            psi_name='prim_act_'+str(u)+'_'+str(v)
            tup_1=tuple([x_name,con_name])
            tup_2=tuple([psi_name,con_name])
            tup_1=str(tup_1)
            tup_2=str(tup_2)
            self.actionIntegCon2Contrib[tup_1]=1
            self.primIntegCon2Contrib[tup_2]=-1
            self.all_integ_con.append(con_name)
        self.out_dict['actionIntegCon2Contrib']=self.actionIntegCon2Contrib
        self.out_dict['primIntegCon2Contrib']=self.primIntegCon2Contrib
        self.out_dict['allIntegCon']=self.all_integ_con
    def make_all_delta(self):
        self.all_delta=[]
        for u in range(0,self.my_instance.num_cust):
            dem_name='delta_capRem_'+str(u)
            time_name='delta_timeRem_'+str(u)
            self.all_delta.append(dem_name)
            self.all_delta.append(time_name)
        self.out_dict['allDelta']=self.all_delta
    
    def make_contrib(self):
        print('hi making contrib')
        self.deltaCon2Contrib=dict()
        self.actionCon2Contrib=dict()
        for u in range(0,int(self.my_instance.num_cust)):
            name_time_ub='time_ub_'+str(u)
            name_time_lb='time_lb_'+str(u)
            name_cap_lb='cap_lb_'+str(u)
            name_cap_ub='cap_ub_'+str(u)
            dem_var_name='delta_capRem_'+str(u)
            time_var_name='delta_timeRem_'+str(u)
            tup_time_ub=str(tuple([time_var_name,name_time_ub]))
            tup_time_lb=str(tuple([time_var_name,name_time_lb]))
            tup_cap_ub=str(tuple([dem_var_name,name_cap_ub]))
            tup_cap_lb=str(tuple([dem_var_name,name_cap_lb]))
            self.deltaCon2Contrib[tup_time_ub]=-1
            self.deltaCon2Contrib[tup_time_lb]=1
            self.deltaCon2Contrib[tup_cap_ub]=-1
            self.deltaCon2Contrib[tup_cap_lb]=1
        self.all_actions_not_source_sink_connected=[]
        for uv in self.all_uv_edges:
            u=uv[0]
            v=uv[1]
            new_action='act_'+str(u)+'_'+str(v)
            if u==self.my_instance.num_cust:
                name_min_veh='exog_min_veh_'
                tup_action_minVeh=str(tuple([new_action,name_min_veh]))
                self.actionCon2Contrib[tup_action_minVeh]=1
                
            if u<self.my_instance.num_cust:
                name_cover='exog_cover_'+str(u)
                tup_action=str(tuple([new_action,name_cover]))
                self.actionCon2Contrib[tup_action]=1

                name_flow_in_out_pos='exog_flow_pos'+str(u)
                name_flow_in_out_neg='exog_flow_neg'+str(u)
                tup_action_2=str(tuple([new_action,name_flow_in_out_pos]))
                self.actionCon2Contrib[tup_action_2]=1

                tup_action_3=str(tuple([new_action,name_flow_in_out_neg]))
                self.actionCon2Contrib[tup_action_3]=-1
            if v<self.my_instance.num_cust:
                name_flow_in_out_pos='exog_flow_pos'+str(v)
                name_flow_in_out_neg='exog_flow_neg'+str(v)
                tup_action_2=str(tuple([new_action,name_flow_in_out_pos]))
                self.actionCon2Contrib[tup_action_2]=-1

                tup_action_3=str(tuple([new_action,name_flow_in_out_neg]))
                self.actionCon2Contrib[tup_action_3]=1
            if u<self.my_instance.num_cust and v<self.my_instance.num_cust:
                name_uv_cap='cap_uv_'+str(u)+'_'+str(v)
                name_uv_time='time_uv_'+str(u)+'_'+str(v)
                dem_name_u='delta_capRem_'+str(u)
                time_name_u='delta_timeRem_'+str(u)
                dem_name_v='delta_capRem_'+str(v)
                time_name_v='delta_timeRem_'+str(v)
                self.all_actions_not_source_sink_connected.append(new_action)

                tup_time_action=str(tuple([new_action,name_uv_time]))
                tup_dem_action=str(tuple([new_action,name_uv_cap]))

                tup_time_u=str(tuple([time_name_u,name_uv_time]))
                tup_dem_u=str(tuple([dem_name_u,name_uv_cap]))

                tup_time_v=str(tuple([time_name_v,name_uv_time]))
                tup_dem_v=str(tuple([dem_name_v,name_uv_cap]))

                val_time_action=-self.my_instance.early_start[v]+self.my_instance.late_start[u]-self.my_instance.dist_mat_full[u,v]
                val_dem_action=-self.my_instance.vehicle_capacity-self.my_instance.dem[u]#+.01
                val_time_u=1
                val_dem_u=1
                val_time_v=-1
                val_dem_v=-1
                self.actionCon2Contrib[tup_time_action]=val_time_action
                self.actionCon2Contrib[tup_dem_action]=val_dem_action
                self.deltaCon2Contrib[tup_time_u]=val_time_u
                self.deltaCon2Contrib[tup_dem_u]=val_dem_u
                self.deltaCon2Contrib[tup_time_v]=val_time_v
                self.deltaCon2Contrib[tup_dem_v]=val_dem_v

                
                #input('hihi')

        self.out_dict['deltaCon2Contrib']=self.deltaCon2Contrib
        self.out_dict['actionCon2Contrib']=self.actionCon2Contrib
        self.out_dict['all_actions_not_source_sink_connected']=self.all_actions_not_source_sink_connected
    def make_exog_name_RHS(self):
        self.allExogNames=[]
        self.exogName2Rhs=dict()
        tot_dem=0
        for u in range(0,int(self.my_instance.num_cust)):
            tot_dem=tot_dem+self.my_instance.dem[u]
            name_cover='exog_cover_'+str(u)
            self.allExogNames.append(name_cover)
            self.exogName2Rhs[name_cover]=1

            name_flow_in_out_pos='exog_flow_pos'+str(u)
            name_flow_in_out_neg='exog_flow_neg'+str(u)
            self.allExogNames.append(name_flow_in_out_pos)
            self.allExogNames.append(name_flow_in_out_neg)
            self.exogName2Rhs[name_flow_in_out_pos]=0
            self.exogName2Rhs[name_flow_in_out_neg]=0

            name_time_lb='time_lb_'+str(u)
            self.allExogNames.append(name_time_lb)
            self.exogName2Rhs[name_time_lb]=self.my_instance.late_start[u]

            name_time_ub='time_ub_'+str(u)
            self.allExogNames.append(name_time_ub)
            self.exogName2Rhs[name_time_ub]=-self.my_instance.early_start[u]

            name_cap_lb='cap_lb_'+str(u)
            self.allExogNames.append(name_cap_lb)
            self.exogName2Rhs[name_cap_lb]=self.my_instance.dem[u]

            name_cap_ub='cap_ub_'+str(u)
            self.allExogNames.append(name_cap_ub)
            self.exogName2Rhs[name_cap_ub]=-self.my_instance.vehicle_capacity
        
        min_veh=np.ceil(tot_dem/self.my_instance.vehicle_capacity)
        self.exogName2Rhs['exog_min_veh_']=min_veh
        self.allExogNames.append('exog_min_veh_')
        
        for uv in self.all_uv_edges:
            u=uv[0]
            v=uv[1]
            if u<self.my_instance.num_cust and v<self.my_instance.num_cust:
                name_uv_cap='cap_uv_'+str(u)+'_'+str(v)
                name_uv_time='time_uv_'+str(u)+'_'+str(v)
                self.allExogNames.append(name_uv_cap)
                self.allExogNames.append(name_uv_time)
                self.exogName2Rhs[name_uv_cap] =-self.my_instance.vehicle_capacity
                self.exogName2Rhs[name_uv_time]=-round(self.my_instance.early_start[v]-self.my_instance.late_start[u],1)

        self.out_dict['allExogNames']=self.allExogNames
        self.out_dict['exogName2Rhs']=self.exogName2Rhs

    def make_allActions(self):
        print('all action')
        self.all_actions=[]
        self.null_action='null_action'
        self.out_dict['nullAction']=self.null_action

        self.all_uv_edges=[]
        self.all_non_null_actions=[]
        self.allPrimitiveVars=[]
        self.action_2_cost=dict()
        for u in range(0,int(self.my_instance.num_cust)+2):
            for v in range(0,int(self.my_instance.num_cust)+2):
                if self.my_instance.dist_mat_full[u,v]<np.inf:
                    new_action='act_'+str(u)+'_'+str(v)
                    self.all_uv_edges.append(tuple([u,v]))
                    self.all_actions.append(new_action)
                    self.all_non_null_actions.append(new_action)
                    self.action_2_cost[new_action]=round(self.my_instance.orig_dist_mat_full[u,v],1)
                    new_prim_var='prim_act_'+str(u)+'_'+str(v)

                    self.allPrimitiveVars.append(new_prim_var)
        self.out_dict['allActions']=self.all_actions
        self.out_dict['allNonNullAction']=self.all_non_null_actions
        self.out_dict['allPrimitiveVars']=self.allPrimitiveVars
        self.out_dict['action2Cost']=self.action_2_cost
    def make_graph_names(self):
        self.allGraphNames=[]
        self.allGraphNames.append('timeGraph')
        self.allGraphNames.append('capGraph')
        if self.jy_opt['use_ng']>0.5:
            self.allGraphNames.append('ngGraph')
        self.out_dict['allGraphNames']=self.allGraphNames
    
    def make_h_2_source_sink_id(self):
        self.h2SourceId=dict()
        self.h2SinkId=dict()

        #self.h2SourceId['timeGraph']='(25, 1236.0, 1236.0)'
        #self.h2SourceId['capGraph']='(25, 20.0, 20.0)'
        #self.h2SinkId['timeGraph']='(26, 0, 1236.0)'
        #self.h2SinkId['capGraph']='(26, 0, 20.0)'
        #sink_node=max(self.my_dem_graph.node_list.keys())
        #source_node=sink_node-1

        sink_cust=max(self.my_dem_graph.node_list.keys())
        source_cust=sink_cust-1
        
        self.h2SourceId['timeGraph']=str(self.my_time_graph.node_list[source_cust][0])
        self.h2SourceId['capGraph']=str(self.my_dem_graph.node_list[source_cust][0])
        self.h2SinkId['timeGraph']=str(self.my_time_graph.node_list[sink_cust][0])
        self.h2SinkId['capGraph']=str(self.my_dem_graph.node_list[sink_cust][0])
        
        if self.jy_opt['use_ng']>0.5:
            print('self.ngGraph')
            print(self.ngGraph)

            self.h2SourceId['ngGraph']=str(self.ngGraph.node_list[source_cust][0])
            self.h2SinkId['ngGraph']=str(self.ngGraph.node_list[sink_cust][0])
            
        
        #print(self.h2SourceId['timeGraph'])
        #print(self.h2SourceId['capGraph'])
        #print('self.h2SinkId['timeGraph']')
        #self.h2SourceId['timeGraph']='('+str(source_node)+', 1236.0, 1236.0)'
        #self.h2SourceId['capGraph']='('+str(source_node)+', 20.0, 20.0)'
        #self.h2SinkId['timeGraph']='('+str(sink_node)+', 0, 1236.0)'
        #self.h2SinkId['capGraph']='('+str(sink_node)+', 0, 20.0)'

        self.out_dict['h2SourceId']=self.h2SourceId
        self.out_dict['h2sinkid']=self.h2SinkId
    

    def make_nodes(self):
        self.graphName2Nodes=dict()
        self.graphName2Nodes['capGraph']=[self.h2SourceId['capGraph'],self.h2SinkId['capGraph']]
        self.graphName2Nodes['timeGraph']=[self.h2SourceId['timeGraph'],self.h2SinkId['timeGraph']]

        self.initGraphNode2AggNode=dict()
        self.initGraphNode2AggNode['capGraph']=dict()
        self.initGraphNode2AggNode['timeGraph']=dict()
        self.initGraphNode2AggNode['capGraph'][self.h2SourceId['capGraph']]='cap_-1'
        self.initGraphNode2AggNode['capGraph'][self.h2SinkId['capGraph']]='cap_-2'

        self.initGraphNode2AggNode['timeGraph'][self.h2SourceId['timeGraph']]='time_-1'
        self.initGraphNode2AggNode['timeGraph'][self.h2SinkId['timeGraph']]='time_-2'
        if self.jy_opt['use_ng']>0.5:
            self.graphName2Nodes['ngGraph']=[self.h2SourceId['ngGraph'],self.h2SinkId['ngGraph']]

            self.initGraphNode2AggNode['ngGraph']=dict()
            self.initGraphNode2AggNode['ngGraph'][self.h2SourceId['ngGraph']]='ng_-1'#str(self.ngGraph.node_list[source_cust][0])
            self.initGraphNode2AggNode['ngGraph'][self.h2SinkId['ngGraph']]='ng_-2'#str(self.ngGraph.node_list[sink_cust][0])
            
        for u in range(0,self.my_instance.num_cust):
            my_count=0
            u_use=u
            if self.jy_opt['allOneBig_init']>0.5:
                u_use=2
            for i in self.my_dem_graph.node_list[u]:
                self.graphName2Nodes['capGraph'].append(str(i))
                bin_num=int(my_count/self.num_terms_per_bin)
                if self.jy_opt['allOneBig_init']>0.5:
                    bin_num=50
                    #input('--')
                self.initGraphNode2AggNode['capGraph'][str(i)]='cap_'+str(u_use)+'_'+str(bin_num)
                my_count=my_count+1
            my_count=0
            for i in self.my_time_graph.node_list[u]:
                self.graphName2Nodes['timeGraph'].append(str(i))
                bin_num=int(my_count/self.num_terms_per_bin)
                if self.jy_opt['allOneBig_init']>0.5:
                    bin_num=50
                    #input('--')
                self.initGraphNode2AggNode['timeGraph'][str(i)]='time'+str(u_use)+'_'+str(bin_num)
                my_count=my_count+1
            if self.jy_opt['use_ng']>0.5:
                my_count=0
                for i in self.ngGraph.node_list[u]:
                    self.graphName2Nodes['ngGraph'].append(str(i))
                    bin_num=int(my_count/self.num_terms_per_bin)
                    if self.jy_opt['allOneBig_init']>0.5:
                        bin_num=50
                        #input('--')
                    self.initGraphNode2AggNode['ngGraph'][str(i)]='ng_'+str(u_use)+'_'+str(bin_num)
                    my_count=my_count+1



        self.out_dict['graphName2Nodes']=self.graphName2Nodes
        self.out_dict['initGraphNode2AggNode']=self.initGraphNode2AggNode
    