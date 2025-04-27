#Hier_flow_in_out_pos_h=timeGraph_ell=rand_34610169__10
from collections import defaultdict
#from src.common.route import route
from typing import Dict, DefaultDict, Set, List
import numpy as np
import pulp as pl
from pulp import LpProblem, LpVariable, LpMaximize, PULP_CBC_CMD
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xpress as xp
import networkx as nx
import time
from scipy.sparse import csr_matrix
import pulp
from scipy.cluster.hierarchy import linkage
import xpress as xp
from warm_start_lp import warm_start_lp_using_class


class compressor:
    
    def __init__(self,my_full_solver):
        t1=time.time()
        self.MF=my_full_solver
        self.jy_opt=self.MF.jy_opt
        self.graph_node_2_agg_node=self.MF.graph_node_2_agg_node
        self.myLBObj=self.MF.my_lower_bound_LP
        self.graph_names=self.MF.graph_names
        self.actions_ignore=self.MF.actions_ignore

        #input('about to resolve')
        #self.myLBObj.lp_prob.solve()
        #input('done resolve')
        self.time_compressor=dict()
        self.time_compressor['prior']=time.time()-t1
        t1=time.time()
        self.get_dual_dict()
        self.time_compressor['get_dual_dict']=time.time()-t1
        t1=time.time()
        self.remove_flow_in_flow_out_terms()
        self.time_compressor['remove_flow_in_flow_out_terms']=time.time()-t1
        t1=time.time()
        self.add_new_RHS_terms()
        self.time_compressor['add_new_RHS_terms']=time.time()-t1
        t1=time.time()
        self.add_new_ineq_terms()
        self.time_compressor['add_new_ineq_terms']=time.time()-t1

        if my_full_solver.jy_opt['use_Xpress']==False:
            self.make_LP()
        else:
            self.make_xpress_LP()
            #self.make_xpress_LP_Small()

        t1=time.time()

        self.get_pi_by_h_node()
        self.time_compressor['get_pi_by_h_node']=time.time()-t1
        t1=time.time()
        self.make_f_2_new_f()
        self.time_compressor['make_f_2_new_f']=time.time()-t1
        t1=time.time()
        self.make_i_2_new_f()
        self.time_compressor['make_i_2_new_f']=time.time()-t1
        
        total = sum(self.time_compressor.values())
        self.time_percentage_compressor = {key: (val / total if total != 0 else 0) for key, val in self.time_compressor.items()}
        if self.MF.jy_opt['verbose']==True:
            print('self.time_compressor')
            print(self.time_compressor)
            print('time_percentage_compressor')
            print(self.time_percentage_compressor)
            print('----')
    def check_agg_nodes(self):
        for  h in self.graph_names:
            count_find=dict()

            for i in self.graph_node_2_agg_node[h]:
                count_find[i]=0
            for f in self.myLBObj.agg_node_2_nodes[h]:
                for i in self.myLBObj.agg_node_2_nodes[h][f]:
                    count_find[i]=count_find[i]+1
            for i in self.graph_node_2_agg_node[h]:
                if count_find[i]!=1:
                    print('count_find[i]')
                    print(count_find[i])
                    input('error here')
    def get_dual_dict(self):

        self.check_agg_nodes()

        dual_sol=self.MF.my_lower_bound_LP.lp_dual_solution
        h_node_2_dual_val=dict()
        self.H_ell_2_list_leaf=dict()
        self.H_leaf_2_list_ell=dict()
        for h in self.graph_names:
            h_node_2_dual_val[h]=dict()
            self.H_ell_2_list_leaf[h]=dict()
            self.H_leaf_2_list_ell[h]=dict()
            this_sink=self.graph_node_2_agg_node[h][self.MF.h_2_sink_id[h]]
            this_source=self.graph_node_2_agg_node[h][self.MF.h_2_source_id[h]]
            nodes_use=set(self.myLBObj.agg_node_2_nodes[h])-set([this_sink,this_source])
            for n in nodes_use:
                con_name='flow_in_out_h='+h+"_n="+n
                h_node_2_dual_val[h][n]=dual_sol[con_name]

            self.get_agglomerative_dictionary(h_node_2_dual_val[h],h)
    def get_agglomerative_dictionary(self,X,h):
        
        keys = list(X.keys())
        n = len(keys)

        # Create a 2-D array of scalar values (each value is a 1-D point).
        data = np.array([X[k] for k in keys]).reshape(-1, 1)

        if n == 1:
            # Create the single node clustering information.
            cluster_id_2_member = {0: [keys[0]]}
            member_2_cluster_id = {keys[0]: [0]}
            
            # The output dictionaries use tuples (h, cluster_id)
            out_cluster_id_2_member = { (h, 0): [keys[0]] }
            out_member_2_cluster_id = { keys[0]: [(h, 0)] }
            
            # Save to the corresponding attributes and exit.
            self.H_ell_2_list_leaf[h] = out_cluster_id_2_member
            self.H_leaf_2_list_ell[h] = out_member_2_cluster_id
            return

        # Compute the linkage matrix using SciPy's hierarchical clustering.
        Z = linkage(data, method="average")
        
        cluster_id_2_member = {}
        for i in range(n):
            cluster_id_2_member[i] = [keys[i]]
            
    
        # Process each merge from the linkage matrix.
        # Each row in Z is of the form: [idx1, idx2, distance, sample_count]
        # New cluster ids are assigned as n, n+1, ... (as is standard in SciPy)
        for i, row in enumerate(Z):
            idx1, idx2, distance, count = row
            idx1, idx2 = int(idx1), int(idx2)
            new_cluster_id = n + i  # new cluster id for this merge
            # The new cluster contains all leaves from the clusters idx1 and idx2.
            cluster_id_2_member[new_cluster_id] = cluster_id_2_member[idx1] + cluster_id_2_member[idx2]
        
        member_2_cluster_id=dict()
        for my_node in X:
            member_2_cluster_id[my_node]=[]
        for cluster_id in cluster_id_2_member:
            for q in cluster_id_2_member[cluster_id]:
                member_2_cluster_id[q].append(cluster_id)
        
        out_cluster_id_2_member=dict()
        for i in cluster_id_2_member:
            i_out=tuple([h,i])
            out_cluster_id_2_member[i_out]=cluster_id_2_member[i]

        out_member_2_cluster_id=dict()
        for i in member_2_cluster_id:
            out_member_2_cluster_id[i]=[]
            for j in member_2_cluster_id[i]:
                out_member_2_cluster_id[i].append(tuple([h,j]))


        self.H_ell_2_list_leaf[h]=out_cluster_id_2_member
        self.H_leaf_2_list_ell[h]=out_member_2_cluster_id



    def remove_flow_in_flow_out_terms(self):
        my_prefix='flow_in_out_h'
        
        lp_prob=self.myLBObj.lp_prob
        # Iterate over all constraints, filtering those whose name starts with my_prefix.
        constraints_to_remove = [con for con in lp_prob.getConstraint() if con.name.startswith(my_prefix)]

        # Remove each filtered constraint.
        for con in constraints_to_remove:
            lp_prob.delConstraint(con)




    def add_new_RHS_terms(self):
        weight_compress=self.MF.jy_opt['weight_compress']
        mult_figs=100
        counter=0
        self.H_ell_2_pos=dict()
        self.H_ell_2_neg=dict()
        self.ell_2_pos_neg=dict()
        self.NEW_RHS=dict()
        self.NEW_Con_name_2_dict=dict()
        for h in self.graph_names:
            self.H_ell_2_pos[h]=dict()
            self.H_ell_2_neg[h]=dict()
            for ell in self.H_ell_2_list_leaf[h]:
                component_leaves=self.H_ell_2_list_leaf[h][ell]
                num_terms=len(component_leaves)
                weight_compress
                my_weight_pre_round=(1/num_terms)
                my_weight=-(weight_compress/mult_figs)*np.ceil(my_weight_pre_round)
                con_1='Hier_flow_in_out_pos_h='+h+"_ell="+str(ell)
                con_2='Hier_flow_in_out_neg_h='+h+"_ell="+str(ell)
                con_1= con_1.replace(" ", "_")
                con_2= con_2.replace(" ", "_")
                con_1= con_1.replace("(", "_")
                con_2= con_2.replace("(", "_")
                con_1= con_1.replace(")", "_")
                con_2= con_2.replace(")", "_")
                self.NEW_RHS[con_1]=my_weight
                self.NEW_RHS[con_2]=my_weight
                self.NEW_Con_name_2_dict[con_1]=defaultdict(float)
                self.NEW_Con_name_2_dict[con_2]=defaultdict(float)
                self.H_ell_2_neg[h][con_1]=ell
                self.H_ell_2_pos[h][con_2]=ell
                self.ell_2_pos_neg[con_1]=ell
                self.ell_2_pos_neg[con_2]=ell
                
                counter=counter+1
                
    def add_new_ineq_terms(self):
        counterMe=0
        
        for h in self.graph_names:
            agg_source=self.graph_node_2_agg_node[h][self.MF.h_2_source_id[h]]
            agg_sink=self.graph_node_2_agg_node[h][self.MF.h_2_sink_id[h]]
            for fg in self.myLBObj.h_fg_2_q[h]:
                f=fg[0]
                g=fg[1]
                var_name='EDGE_h='+h+'_f='+f+'_g='+g

                f=fg[0]
                g=fg[1]
                
                if f==g: 
                    continue
                if f!=agg_source:
                    for ell in self.H_leaf_2_list_ell[h][f]:
                        con_1='Hier_flow_in_out_pos_h='+h+"_ell="+str(ell)
                        con_2='Hier_flow_in_out_neg_h='+h+"_ell="+str(ell)
                        con_1= con_1.replace(" ", "_")
                        con_2= con_2.replace(" ", "_")
                        con_1= con_1.replace("(", "_")
                        con_2= con_2.replace("(", "_")
                        con_1= con_1.replace(")", "_")
                        con_2= con_2.replace(")", "_")
                        tup_1=tuple([var_name,con_1])
                        tup_2=tuple([var_name,con_2])

                        #self.dict_var_con_2_lhs_exog[tup_1]+=1
                        #self.dict_var_con_2_lhs_exog[tup_2]+=-1
                        self.NEW_Con_name_2_dict[con_1][var_name]+=1
                        self.NEW_Con_name_2_dict[con_2][var_name]+=-1
                        
                        counterMe=counterMe+1
                if g!=agg_sink:
                    for ell in self.H_leaf_2_list_ell[h][g]:
                        con_1='Hier_flow_in_out_pos_h='+h+"_ell="+str(ell)
                        con_2='Hier_flow_in_out_neg_h='+h+"_ell="+str(ell)
                        con_1= con_1.replace(" ", "_")
                        con_2= con_2.replace(" ", "_")
                        con_1= con_1.replace("(", "_")
                        con_2= con_2.replace("(", "_")
                        con_1= con_1.replace(")", "_")
                        con_2= con_2.replace(")", "_")
                        #tup_1=tuple([var_name,con_1])
                        #tup_2=tuple([var_name,con_2])
                        #if tup_1 not in self.dict_var_con_2_lhs_exog:
                        #    self.dict_var_con_2_lhs_exog[tup_1]=0
                        #if tup_2 not in self.dict_var_con_2_lhs_exog:
                        #    self.dict_var_con_2_lhs_exog[tup_2]=0
                        
                        #self.dict_var_con_2_lhs_exog[tup_1]+=-1
                        #self.dict_var_con_2_lhs_exog[tup_2]+=1
                        self.NEW_Con_name_2_dict[con_1][var_name]+=-1
                        self.NEW_Con_name_2_dict[con_2][var_name]+=1
                        counterMe=counterMe+1

    def make_xpress_LP(self):
        t2=time.time()
        lp_prob=self.myLBObj.lp_prob

        
        NEW_Con_name_2_dict=self.NEW_Con_name_2_dict
        NEW_RHS=self.NEW_RHS
        var_dict=self.myLBObj.var_dict
        for con_name, rhs in NEW_RHS.items():
            # Build the linear expression from the coefficient dictionary.
            # NEW_Con_name_2_dict[con_name] is assumed to be a defaultdict mapping variable names to coefficients.
            lhs = xp.Sum(NEW_Con_name_2_dict[con_name][var_name] * var_dict[var_name]
                        for var_name in NEW_Con_name_2_dict[con_name])
            
            # Create a constraint of the form: lhs >= rhs.
            constraint_obj = xp.constraint(lhs >= rhs, name=con_name)
            
            # Add the constraint to your lp_prob.
            lp_prob.addConstraint(constraint_obj)
        
        lp_prob.controls.defaultalg = self.MF.jy_opt['compress_solver']

        #('starting')
        self.time_compressor['pre_XLP']=time.time()-t2

        if self.MF.jy_opt['use_julians_custom_lp_solver']<0.5:
            start_time = time.time()

            lp_prob.solve()
            end_time = time.time()
            self.lp_time = end_time - start_time
        else: #lp_prob, var_dict, zero_names
            lp_prob,time_lp_1=warm_start_lp_using_class(lp_prob,var_dict,self.MF.all_actions_not_source_sink_connected,self.actions_ignore)
            self.lp_time=time_lp_1
        #input('stopping')

        #self.lp_time = end_time - start_time
        self.time_compressor['lp_time']=self.lp_time
        t3=time.time()
        self.lp_prob = lp_prob
        self.lp_primal_solution = dict()
        #for var_name, var in var_dict.items():
        #    self.lp_primal_solution[var_name] = lp_prob.getSolution(var_name)

        self.lp_status = lp_prob.getProbStatus()
        self.lp_objective = lp_prob.getObjVal()
        self.lp_dual_solution = dict()

        lp_sol = lp_prob.getSolution()
        self.lp_primal_solution = {var_obj.name: lp_sol[i]
                                for i, var_obj in enumerate(lp_prob.getVariable())}

        lp_dual = lp_prob.getDuals()
        self.lp_dual_solution = {con.name: lp_dual[i]
                                for i, con in enumerate(lp_prob.getConstraint())}

        if self.lp_status == 'Infeasible':
            input('HOLD')
        self.time_compressor['post_XLP']=time.time()-t3

    def get_pi_by_h_node(self):
        self.h_f_2_dual=dict()
        self.h_f_2_dual_sig_fig=dict()
        self.h_val_2_id=dict()
        for h in self.graph_names:
            self.h_f_2_dual[h]=dict()
            self.h_f_2_dual_sig_fig[h]=dict()
            self.h_val_2_id[h]=dict()
            counter_h=0
            for f in self.H_leaf_2_list_ell[h]:
                self.h_f_2_dual[h][f]=0
                for ell in self.H_leaf_2_list_ell[h][f]:
                    con_1='Hier_flow_in_out_pos_h='+str(h)+"_ell="+str(ell)
                    con_2='Hier_flow_in_out_neg_h='+str(h)+"_ell="+str(ell)
                    con_1= con_1.replace(" ", "_")
                    con_2= con_2.replace(" ", "_")
                    con_1= con_1.replace("(", "_")
                    con_2= con_2.replace("(", "_")
                    con_1= con_1.replace(")", "_")
                    con_2= con_2.replace(")", "_")
                    
                    dual_1=self.lp_dual_solution[con_1]
                    dual_2=self.lp_dual_solution[con_2]
                    #print('dual_1')
                    #print(dual_1)
                    #print('dual_2')
                    #print(dual_2)
                    #print('con_1')
                    #print(con_1)
                    #print('con_2')
                    #print(con_1)
                    self.h_f_2_dual[h][f]=self.h_f_2_dual[h][f]+dual_1-dual_2
                new_val=round(self.h_f_2_dual[h][f],3)
                self.h_f_2_dual_sig_fig[h][f]=new_val
                if tuple([h,new_val]) not in self.h_val_2_id[h]:
                    self.h_val_2_id[h][tuple([h,new_val])]=counter_h
                    counter_h=counter_h+1
                
    def make_f_2_new_f(self):
        myLBObj=self.myLBObj
        self.H_f_2_new_f=dict()
        for h in self.graph_names:
            self.H_f_2_new_f[h]=dict()
            this_fg_sink=myLBObj.graph_node_2_agg_node[h][myLBObj.h_2_sink_id[h]]
            this_fg_source=myLBObj.graph_node_2_agg_node[h][myLBObj.h_2_source_id[h]]
            self.H_f_2_new_f[h][this_fg_sink]=tuple([h,-2])
            self.H_f_2_new_f[h][this_fg_source]=tuple([h,-1])
            for f in self.h_f_2_dual_sig_fig[h]:
                my_dual_val=self.h_f_2_dual_sig_fig[h][f]
                my_dual_id=self.h_val_2_id[h][tuple([h,my_dual_val])]
                my_key=tuple([h,my_dual_id])
                self.H_f_2_new_f[h][f]=my_key
    def make_i_2_new_f(self):
        self.NEW_graph_node_2_agg_node=dict()
        count_orig=dict()
        count_new=dict()
        for h in self.graph_names:
            self.NEW_graph_node_2_agg_node[h]=dict()
            count_orig[h]=len(set(self.myLBObj.graph_node_2_agg_node[h].values()))
            count_new[h]=len(set(self.H_f_2_new_f[h].values()))
            
            for i in self.myLBObj.graph_node_2_agg_node[h]:
                f=self.myLBObj.graph_node_2_agg_node[h][i]
                
                
                if f not in self.H_f_2_new_f[h]:
                    print('not fuond')
                    input('error here ')
                #print('f')
                #print(f)
                my_new_name=str(self.H_f_2_new_f[h][f])
                my_new_name=my_new_name.replace(" ", "_")
                self.NEW_graph_node_2_agg_node[h][i]=my_new_name
       