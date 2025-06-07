from collections import defaultdict
from solve_gurobi_lp import solve_gurobi_lp
import json
import re
import numpy as np
class my_LP_sep_structure:

    def __init__(self,my_graph,MF,OPT):
        self.my_graph=my_graph
        self.MF=MF
        self.OPT=OPT
        self.primal_sol=self.MF.my_lower_bound_LP.lp_primal_solution
        self.try_slack_match=True
        if 1<0:
            print('SOLUTION')
            for my_key in self.primal_sol:
                if my_key.startswith('act') and self.primal_sol[my_key]>0.01:
            #        print('my_key')
                    
                    u, y1 = my_key[4:].split('_')
                    if int(u) in self.my_graph.my_subset_cust or int(y1) in self.my_graph.my_subset_cust:
                        print(my_key+'  = '+str(self.primal_sol[my_key]))                   
                        
        #        print(self.primal_sol[my_key])
        #self.my_graph.E
        #print('----')
        #input('--')
        self.E=self.my_graph.E
        self.Nodes=self.my_graph.my_candid_nodes
        self.uv_2_E=self.my_graph.uv_2_E

        self.non_source_sink_nodes=[]
        self.Nc=self.MF.my_VRP.num_cust
        self.my_subset_cust=self.my_graph.my_subset_cust
        
        self.nodes_need_pos_slack=[]
        for n in self.Nodes:
            if len(n[1])==0:
                self.nodes_need_pos_slack.append(n)
        #for 
        ##self.Nodes=self.my_NG.nodes
        #self.non_source_sink_nodes=self.my_NG.non_source_sink_nodes
        self.dict_valid_ineq_name_2_rhs=self.my_graph.dict_valid_ineq_name_2_rhs
        self.dict_valid_ineq_name_node_2_coeff=self.my_graph.dict_valid_ineq_name_node_2_coeff
        #self.source_node=self.my_NG.source_node
        #self.sink_node=self.my_NG.sink_node

        self.dict_var_2_obj=dict()
        self.dict_var_con_2_lhs_exog=dict()
        self.dict_var_con_2_lhs_eq=dict()
        self.dict_con_name_2_LB=dict()
        self.dict_con_name_2_eq=dict()
        self.con_name_xuv_2_benders_contrib=dict()
        self.con_name_2_benders_OBJ=dict()
        #print('making vars')
        self.make_vars()
        
        #print('making match')
        self.make_match()
        #print('making flow match')
        self.make_flow_match()
        self.make_entry_rule()
        print('make_valid_ineq')
        self.make_valid_ineq()
        print('ready to call lp')
        #rint('Slack_validfrozenset({8, 10, 12})_2_1.0' in self.dict_var_2_obj)
        #input('just before final check')

        #for (v, c) in self.dict_var_con_2_lhs_exog:
        #    assert v in self.dict_var_2_obj, f"Variable {v} not in dict_var_2_obj"
        #    assert c in self.dict_con_name_2_LB, f"Constraint {c} not in dict_con_name_2_LB"

        
        epsilon = 1e-6  # Tolerance for "close to zero"

        #near_zero_LB = {
        #   con: val
        #    for con, val in self.dict_con_name_2_LB.items()
        #    if abs(val) < epsilon
        #}

        #near_zero_eq = {
        #    con: val
        #    for con, val in self.dict_con_name_2_eq.items()
        #    if abs(val) < epsilon
        #}
        
        #if len(near_zero_LB)>0 or len(near_zero_eq)>0:
        #    print('near_zero_eq')
        #    print(near_zero_eq)
        #    print('near_zero_LB')
        #    print(near_zero_LB)
        #    input()


        self.out_solution=solve_gurobi_lp(self.dict_var_2_obj,
                    self.dict_var_con_2_lhs_exog,
                    self.dict_con_name_2_LB,
                    self.dict_var_con_2_lhs_eq,
                    self.dict_con_name_2_eq)
        if 1<0:
            print('done lp solve')
            #input('puase')
            x=self.out_solution['primal_solution']
            print('slacks used')
            for p in x:
                if x[p]>0.001 and p.startswith('SLACK_Neg_'):
                    print(p+'  '+str(x[p]))
            print('---')
        if self.out_solution['objective']>0.0001:

            self.make_benders_cut()
            self.add_ineq_to_MF()
    def make_vars(self):

        self.dict_var_2_obj=dict()
        self.dict_e_2_var_name=dict()
        #self.REV_dict_e_2_var_name=dict()

        self.dict_node_2_slack_pos_name=dict()
        self.dict_node_2_slack_neg_name=dict()
        self.dict_node_2_slack_neg_flow_name=dict()
        #self.REV_dict_node_2_slack_name=dict()
        
        for e in self.E:
            var_name='EDGE_VAR_'+str(e)
            #print(e)
            self.dict_e_2_var_name[e]=var_name
            self.dict_var_2_obj[var_name]=0
        
        for n in self.Nodes:
            var_name='SLACK_Pos_'+str(n)
            self.dict_node_2_slack_pos_name[n]=var_name
            if n in self.nodes_need_pos_slack:
                self.dict_var_2_obj[var_name]=1
            else:
                self.dict_var_2_obj[var_name]=1
        for n in self.Nodes:
            var_name='SLACK_Neg_flow_'+str(n)
            self.dict_node_2_slack_neg_flow_name[n]=var_name
            self.dict_var_2_obj[var_name]=1
        
        for n in self.Nodes:
            var_name='SLACK_Neg_'+str(n)
            self.dict_node_2_slack_neg_name[n]=var_name
            self.dict_var_2_obj[var_name]=0

        for q in self.dict_valid_ineq_name_2_rhs:
            var_name='Slack_valid'+q

            self.dict_var_2_obj[var_name]=1
        if self.try_slack_match==True:
            for u in self.my_subset_cust:
                for v in self.my_subset_cust:
                    var_name_primal='act_'+str(u)+'_'+str(v)
                    
                    if var_name_primal in self.primal_sol:
                        var_slack_pos='VarPosMatch_'+str(u)+'_'+str(v)
                        var_slack_neg='VarNegMatch_'+str(u)+'_'+str(v)
                        self.dict_var_2_obj[var_slack_pos]=1
                        self.dict_var_2_obj[var_slack_neg]=1
    def make_flow_match(self):

        print('make_flow_match')
        self.dict_node_2_flow_con_name=dict()
        self.REV_dict_node_2_flow_con_name=dict()
        
        for n in self.Nodes:
            ##print('n')
            #print(n)
            con_name='flow_'+str(n)
            self.dict_node_2_flow_con_name[n]=con_name
            self.REV_dict_node_2_flow_con_name[con_name]=n
            self.dict_con_name_2_eq[con_name]=0
            
            if n in self.nodes_need_pos_slack:
                
                v=n[0]
                #input('in here')
                for u in range(0,self.Nc+1):
                    if u not in self.my_graph.my_subset_cust:
                        var_name='act_'+str(u)+'_'+str(v)
                        if var_name in self.primal_sol:
                            self.dict_con_name_2_eq[con_name]-=(self.primal_sol[var_name]+self.OPT['my_magnanti'])
                            self.con_name_xuv_2_benders_contrib[tuple([con_name,var_name])]=1
                
            my_pos_slack=self.dict_node_2_slack_pos_name[n]
            self.dict_var_con_2_lhs_eq[tuple([my_pos_slack,con_name])]=1
            my_neg_slack=self.dict_node_2_slack_neg_name[n]
            self.dict_var_con_2_lhs_eq[tuple([my_neg_slack,con_name])]=-1

            my_neg_slack_flow=self.dict_node_2_slack_neg_flow_name[n]
            self.dict_var_con_2_lhs_eq[tuple([my_neg_slack_flow,con_name])]=-1
            #input('---')

        for e in self.E:
            var_name=self.dict_e_2_var_name[e]
            n1=e[0]
            n2=e[1]
            name_con_1=self.dict_node_2_flow_con_name[n1]
            name_con_2=self.dict_node_2_flow_con_name[n2]
            self.dict_var_con_2_lhs_eq[tuple([var_name,name_con_1])]=-1
            self.dict_var_con_2_lhs_eq[tuple([var_name,name_con_2])]=1


    def make_match(self):
        my_primal=self.primal_sol
        for u in self.my_subset_cust:
            for v in self.my_subset_cust:
                var_name='act_'+str(u)+'_'+str(v)
                
                if var_name in my_primal:
                    con_name='match_'+str(u)+'_'+str(v)
                    self.dict_con_name_2_LB[con_name]=-(my_primal[var_name]+self.OPT['my_magnanti'])
                    self.con_name_xuv_2_benders_contrib[tuple([con_name,var_name])]=1
                    for e in self.uv_2_E[tuple([u,v])]:
                        e_var_name=self.dict_e_2_var_name[e]
                        self.dict_var_con_2_lhs_exog[tuple([e_var_name,con_name])]=-1
                    
                    if self.try_slack_match:
                        var_slack_pos='VarPosMatch_'+str(u)+'_'+str(v)
                        self.dict_var_con_2_lhs_exog[tuple([var_slack_pos,con_name])]=1
                    #self.dict_var_2_obj[var_slack_pos]=1
                    #self.dict_var_2_obj[var_slack_neg]=1

    def make_entry_rule(self):
        use_ineq=True
        my_primal=self.primal_sol
        for v in self.my_subset_cust:
            con_name='Conshould'+str(v)
            slack_var_pos='VarshouldPos'+str(v)
            self.dict_var_2_obj[slack_var_pos]=1
            #slack_var_neg='VarshouldNeg'+str(v)
            #self.dict_var_2_obj[slack_var_neg]=-1
            self.con_name_2_benders_OBJ[con_name]=1

            if use_ineq==True:
                self.dict_con_name_2_LB[con_name]=1
                self.dict_var_con_2_lhs_exog[tuple([slack_var_pos,con_name])]=1

            else:
                self.dict_con_name_2_eq[con_name]=1
                slack_var_neg='VarshouldNeg'+str(v)
                self.dict_var_2_obj[slack_var_neg]=1
                self.dict_var_con_2_lhs_eq[tuple([slack_var_pos,con_name])]=1
                self.dict_var_con_2_lhs_eq[tuple([slack_var_pos,con_name])]=-1

            for  u in range(0,self.Nc+1):
                var_name='act_'+str(u)+'_'+str(v)
                if u in self.my_subset_cust:
                    continue
                if var_name in my_primal:
                    self.con_name_xuv_2_benders_contrib[tuple([con_name,var_name])]=1
                    if use_ineq==True:
                        self.dict_con_name_2_LB[con_name]-=(self.primal_sol[var_name]+self.OPT['my_magnanti'])
                    else:
                        self.dict_con_name_2_eq[con_name]-=(self.primal_sol[var_name]+self.OPT['my_magnanti'])

                    #self.dict_con_name_2_LB[con_name]-=(self.primal_sol[var_name]+self.OPT['my_magnanti'])
            #if self.dict_con_name_2_LB[con_name]<-.001:
            #    print('self.dict_con_name_2_LB[con_name]')
            #    print(self.dict_con_name_2_LB[con_name])
            #    input('not expecting this')
        for e in self.E:
            v=e[1][0]
            con_name='Conshould'+str(v)
            var_name=self.dict_e_2_var_name[e]
            if use_ineq==True:
                self.dict_var_con_2_lhs_exog[tuple([var_name,con_name])]=1
            else:
                self.dict_var_con_2_lhs_eq[tuple([var_name,con_name])]=1


    def make_valid_ineq(self):
        for q in self.dict_valid_ineq_name_2_rhs:
            con_name='Valid_ineq_'+q
            self.dict_con_name_2_LB[con_name]=self.dict_valid_ineq_name_2_rhs[q]
            if self.dict_valid_ineq_name_2_rhs[q]>-.999:
                input('error2 ')
            self.con_name_2_benders_OBJ[con_name]=self.dict_valid_ineq_name_2_rhs[q]
            slack_valid_var_name='Slack_valid'+q
            if slack_valid_var_name not in self.dict_var_2_obj:
                input('error here 3')
                
            self.dict_var_con_2_lhs_exog[tuple([slack_valid_var_name,con_name])]=1
            
            for n in self.dict_valid_ineq_name_node_2_coeff[q]:
                coeff=self.dict_valid_ineq_name_node_2_coeff[q][n]
                slack_neg_name=self.dict_node_2_slack_neg_name[n]
                self.dict_var_con_2_lhs_exog[tuple([slack_neg_name,con_name])]=coeff
                if coeff>-.999:
                    input('error bad')
                #print('tuple([slack_neg_name,con_name,coeff])')
                #print(tuple([slack_neg_name,con_name,coeff,self.dict_valid_ineq_name_2_rhs[q]]))
            #print('len(self.dict_valid_ineq_name_2_rhs)')
            #print(len(self.dict_valid_ineq_name_2_rhs))
            #input('--')
    def make_benders_cut(self):
        epsilon=.00001
        benders_obj=-epsilon
        self.lp_dual_solution=self.out_solution['dual_solution']
        benders_dict=dict()
        print('making cut objective')
        for con_name in self.con_name_2_benders_OBJ:
            dual_val=self.lp_dual_solution[con_name]
            tmp=self.con_name_2_benders_OBJ[con_name]*dual_val
            benders_obj+=tmp
            #if abs(self.con_name_2_benders_OBJ[con_name]*dual_val)>0.001:
            #    print('con_name = '+con_name)
            #    print('self.con_name_2_benders_OBJ[con_name] = '+str(self.con_name_2_benders_OBJ[con_name]))
            #    print('dual_val = '+str(dual_val))
            #    print('tmp = '+ str(tmp))
        
        for n in self.nodes_need_pos_slack:
            con_name=self.dict_node_2_flow_con_name[n]
            dual_val=self.lp_dual_solution[con_name]
            if dual_val<-0.001:
                input('error here')
        for (con_name,var_name) in self.con_name_xuv_2_benders_contrib:
            dual_val=self.lp_dual_solution[con_name]
            if var_name not in benders_dict:
                benders_dict[var_name]=0
            
            tmp=self.con_name_xuv_2_benders_contrib[(con_name,var_name)]*dual_val
           
            benders_dict[var_name]+=tmp
        self.benders_dict=benders_dict
        self.benders_obj=benders_obj
        #print('my_subset_cust')
        #print(self.my_subset_cust)
        #input('look at the cut')
    def add_ineq_to_MF(self):
        cur_count_cutting_planes=self.MF.count_cutting_planes
        self.new_CP_name='my_valid_ineq_'+str(cur_count_cutting_planes)
        self.MF.all_exog.append(self.new_CP_name)
        self.MF.count_cutting_planes+=1
        self.MF.exog_name_2_rhs[self.new_CP_name]=self.benders_obj#my_separ.new_cut_RHS-self.epsilon_slack_valid*2
        DEBUG_RHS=self.benders_obj
        DEBUG_LHS=0
        all_rows=['OFFSET',self.benders_obj]
        all_rows.append(['MyCust in cut ',str(self.my_subset_cust)])
        all_rows.append(['objective ',str(self.out_solution['objective'])])
        all_rows.append(['iteration Number ',len(self.MF.history_dict['lblp_lower'])])
        all_rows.append(['current lower bound ',self.MF.history_dict['lblp_lower'][-1]])
        for primal_var in self.benders_dict:
            val=self.benders_dict[primal_var]
            if abs(val)>0.0001:# or self.primal_sol[primal_var]>0.0001:
                new_row=[primal_var,val,self.primal_sol[primal_var]]
                all_rows.append(new_row)

            if abs(val)>0.000001:
                #print('val')
                #print(val)
                #print('primal_var')
                #print(primal_var)
                self.MF.action_con_2_contrib[tuple([primal_var,self.new_CP_name])]=val
                DEBUG_LHS+=(self.primal_sol[primal_var]+self.OPT['my_magnanti'])*val
        #print('self.benders_obj')
        #print(self.benders_obj)
        #print('DEBUG_RHS')
        #print(DEBUG_RHS)
        #print('DEBUG_LHS')
        #print(DEBUG_LHS)
        gap=abs((DEBUG_RHS-DEBUG_LHS)-self.out_solution['objective'])
        if gap>0.01:
            print('DEBUG_RHS-DEBUG_LHS)')
            print(DEBUG_RHS-DEBUG_LHS)
            print('self.out_solution[objective]')
            print(self.out_solution['objective'])
            input('-ERROR HERE --')
        with open('../ALL_JSON_BIG/NewCuts/CUT_FILE_'+str(self.MF.count_cutting_planes)+'.json', 'w') as file:
            json.dump(all_rows, file)