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


class compressor:
    
    def __init__(self,my_full_solver):
        t1=time.time()
        self.MF=my_full_solver
        self.jy_opt=self.MF.jy_opt
        self.graph_node_2_agg_node=self.MF.graph_node_2_agg_node
        self.myLBObj=self.MF.my_lower_bound_LP
        self.graph_names=self.MF.graph_names
        self.time_compressor=dict()
        self.time_compressor['prior']=time.time()-t1
        t1=time.time()
        self.get_dual_dict()
        self.time_compressor['get_dual_dict']=time.time()-t1
        t1=time.time()
        self.copy_lp_terms()
        self.time_compressor['copy_lp_terms']=time.time()-t1
        t1=time.time()
        self.remove_flow_in_flow_out_terms()
        self.time_compressor['remove_flow_in_flow_out_terms']=time.time()-t1
        t1=time.time()
        self.add_new_RHS_terms()
        self.time_compressor['add_new_RHS_terms']=time.time()-t1
        t1=time.time()
        self.add_new_ineq_terms()
        self.time_compressor['add_new_ineq_terms']=time.time()-t1
        t1=time.time()
        self.check_solution_feasibility()
        self.time_compressor['check_solution_feasibility']=time.time()-t1
        if my_full_solver.jy_opt['use_Xpress']==False:
            self.make_LP()
        else:
            self.make_xpress_LP()
            
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


    def copy_lp_terms(self):
        
        
        myLBObj=self.myLBObj

        self.dict_var_name_2_obj=myLBObj.dict_var_name_2_obj.copy()
        self.dict_var_con_2_lhs_exog=myLBObj.dict_var_con_2_lhs_exog.copy()
        self.dict_con_name_2_LB=myLBObj.dict_con_name_2_LB.copy()
        self.dict_var_con_2_lhs_eq=myLBObj.dict_var_con_2_lhs_eq.copy()
        self.dict_con_name_2_eq=myLBObj.dict_con_name_2_eq.copy()

    def remove_flow_in_flow_out_terms(self):
        my_prefix='flow_in_out_h'
        len_prefix=len(my_prefix)
        terms_remove=[]
        for con_name in self.dict_con_name_2_eq:
            if len(con_name)>=len_prefix:
                if con_name[0:len_prefix]==my_prefix:
                    terms_remove.append(con_name)
        for con_name in terms_remove:#self.dict_con_name_2_eq:
            del self.dict_con_name_2_eq[con_name]
        self.dict_var_con_2_lhs_eq = {k: v for k, v in self.dict_var_con_2_lhs_eq.items() if k[1] not in terms_remove}
    
    def add_new_RHS_terms(self):
        weight_compress=self.MF.jy_opt['weight_compress']
        mult_figs=100
        counter=0
        self.H_ell_2_pos=dict()
        self.H_ell_2_neg=dict()
        self.ell_2_pos_neg=dict()
        for h in self.graph_names:
            self.H_ell_2_pos[h]=dict()
            self.H_ell_2_neg[h]=dict()
            #print('h')
            #print(h)
            #input('---')
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
                self.dict_con_name_2_LB[con_1]=my_weight
                self.dict_con_name_2_LB[con_2]=my_weight
                self.H_ell_2_neg[h][con_1]=ell
                self.H_ell_2_pos[h][con_2]=ell
                self.ell_2_pos_neg[con_1]=ell
                self.ell_2_pos_neg[con_2]=ell
                #print('my_weight')
                #print(my_weight)
                #input('---')
                counter=counter+1
                #print('con_2')
                #print(con_2)
                #print('con_1')
                #print(con_1)
        #input(counter)
        #print('self.graph_names')
        #print(self.graph_names)
        #input('counter')
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
                        if tup_1 not in self.dict_var_con_2_lhs_exog:
                            self.dict_var_con_2_lhs_exog[tup_1]=0
                        if tup_2 not in self.dict_var_con_2_lhs_exog:
                            self.dict_var_con_2_lhs_exog[tup_2]=0
                        self.dict_var_con_2_lhs_exog[tup_1]+=1
                        self.dict_var_con_2_lhs_exog[tup_2]+=-1
                        if con_1 not in self.dict_con_name_2_LB:
                            print('con_1')
                            print(con_1)
                            input('issue here ')
                        if con_2 not in self.dict_con_name_2_LB:
                            print('con_2')
                            print(con_2)
                            input('issue here 2 ')
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
                        tup_1=tuple([var_name,con_1])
                        tup_2=tuple([var_name,con_2])
                        if tup_1 not in self.dict_var_con_2_lhs_exog:
                            self.dict_var_con_2_lhs_exog[tup_1]=0
                        if tup_2 not in self.dict_var_con_2_lhs_exog:
                            self.dict_var_con_2_lhs_exog[tup_2]=0
                        self.dict_var_con_2_lhs_exog[tup_1]+=-1
                        self.dict_var_con_2_lhs_exog[tup_2]+=1
                        counterMe=counterMe+1
                        if con_1 not in self.dict_con_name_2_LB:
                            input('issue here 3')
                        if con_2 not in self.dict_con_name_2_LB:
                            input('issue here 4 ')
        #print('counterMe')
        #print(counterMe)
        #input('counterMe')
    def make_LP(self):
        t2=time.time()

        dict_var_name_2_obj=self.dict_var_name_2_obj
        dict_var_con_2_lhs_exog=self.dict_var_con_2_lhs_exog
        dict_con_name_2_LB=self.dict_con_name_2_LB
        dict_var_con_2_lhs_eq=self.dict_var_con_2_lhs_eq
        dict_con_name_2_eq=self.dict_con_name_2_eq
        ##print('dict_var_con_2_lhs_exog')
        #print(dict_var_con_2_lhs_exog)
        #input('---')
        #print('self.dict_con_name_2_LB')
        #print(self.dict_con_name_2_LB)
        #input('---')
        # --- Build the LP model ---
        lp_prob = pulp.LpProblem("MyLP", pulp.LpMinimize)

        # Create decision variables (all non-negative)
        var_dict = {}
        for var_name, coeff in dict_var_name_2_obj.items():
            var_dict[var_name] = pulp.LpVariable(var_name, lowBound=0)

        # Define the objective function (minimize sum(obj_coeff * var))
        lp_prob += pulp.lpSum(dict_var_name_2_obj[var_name] * var_dict[var_name]
                            for var_name in dict_var_name_2_obj), "Objective"

        # --- Add inequality constraints (>=) ---
        # Group terms for each inequality constraint.
        ineq_expressions = {}
        for (var_name, con_name), coeff in dict_var_con_2_lhs_exog.items():
            #if con_name[0]=='H':
            #    print('con_name')
            #    print(con_name)
            #    input('found one here ')
            ineq_expressions.setdefault(con_name, 0)
            ineq_expressions[con_name] += coeff * var_dict[var_name]

        # Add each inequality constraint to the model.
        for con_name, expr in ineq_expressions.items():
            if con_name in dict_con_name_2_LB:
                lp_prob += expr >= dict_con_name_2_LB[con_name], con_name #+ "_ineq"
            else:
                print('con_name')
                print(con_name)
                input('error here')
        # --- Add equality constraints ---
        # Group terms for each equality constraint.
        eq_expressions = {}
        for (var_name, con_name), coeff in dict_var_con_2_lhs_eq.items():
            eq_expressions.setdefault(con_name, 0)
            eq_expressions[con_name] += coeff * var_dict[var_name]

        # Add each equality constraint to the model.
        for con_name, expr in eq_expressions.items():
            if con_name in dict_con_name_2_eq:
                lp_prob += expr == dict_con_name_2_eq[con_name], con_name #+ "_eq"

        # --- Solve the LP ---
        # Using the default CBC solver here.
        self.time_compressor['lp_prior']=time.time()-t2

        start_time=time.time()
        solver = pulp.PULP_CBC_CMD(msg=False)
        lp_prob.solve(solver)
        end_time=time.time()
        self.time_compressor['lp_time']=end_time-start_time

        t3=time.time()
        self.lp_time=end_time-start_time
        self.lp_prob=lp_prob
        self.lp_primal_solution=dict()
        for var_name, var in var_dict.items():
            self.lp_primal_solution[var_name]=var.varValue
        self.lp_status=pulp.LpStatus[lp_prob.status]
        if self.lp_status!='Optimal':
            print('self.lp_status')
            print(self.lp_status)
            input('error inn')
        self.lp_objective= pulp.value(lp_prob.objective)
        self.lp_dual_solution=dict()
        #print('self.lp_objective')
        #print(self.lp_objective)
        #print('self.lp_status')
        #print(self.lp_status)
        #print('looking here ')
        #input('---')
        for con_name, constraint in lp_prob.constraints.items():
            #print('con_name')
            #print(con_name)
            #if con_name[0]=='h':
            #    input('confused')
            #if len(con_name)>=len('Hier_flow_in_out') and con_name[0:len('Hier_flow_in_out')]=='Hier_flow_in_out':
            #    print('FOUND ONE')
            #    print('con_name')
            #    print(con_name)
            #    print('constraint.pi')
            #    print(constraint.pi)
            self.lp_dual_solution[con_name]=constraint.pi
        #print('LP obejctive of comrpessor self.lp_objective')
        #print(self.lp_objective)
        #input('---')
        self.time_compressor['lp_post']=time.time()-t3

    def make_xpress_LP(self):
        xp.init(self.MF.jy_opt['xpress_file_loc'])

        t2=time.time()
        dict_var_name_2_obj = self.dict_var_name_2_obj
        dict_var_con_2_lhs_exog = self.dict_var_con_2_lhs_exog
        dict_con_name_2_LB = self.dict_con_name_2_LB
        dict_var_con_2_lhs_eq = self.dict_var_con_2_lhs_eq
        dict_con_name_2_eq = self.dict_con_name_2_eq
# --- Build the LP model ---
        lp_prob = xp.problem("MyLP")
        lp_prob.setOutputEnabled(self.jy_opt['verbose']>0.5)

        # Create decision variables (all non-negative) and store them in a list.
        var_dict = {}
        vars_list = []  # list of variables to add to the model
        for var_name, coeff in dict_var_name_2_obj.items():
            # Create a variable with lower bound 0.
            v = lp_prob.addVariable(name=var_name, lb=0)
            var_dict[var_name] = v
            vars_list.append(v)

        # Define the objective function (minimize sum of coeff * variable).
        # Converting the generator to a list to be safe.
        objective = xp.Sum([dict_var_name_2_obj[var_name] * var_dict[var_name]
                            for var_name in dict_var_name_2_obj])
        lp_prob.setObjective(objective, sense=xp.minimize)
        # --- Add inequality constraints (>=) ---
        # Group terms for each inequality constraint.
        ineq_expressions = {}
        #did_find_2 = False
        for (var_name, con_name), coeff in dict_var_con_2_lhs_exog.items():
            if con_name not in ineq_expressions:
                ineq_expressions[con_name] = 0
            ineq_expressions[con_name] += coeff * var_dict[var_name]
            #if con_name == 'exog_min_veh_':
            #    did_find_2 = True

        #did_find = False
        cons = []
        for con_name, expr in ineq_expressions.items():
            #if con_name in dict_con_name_2_LB:
                #if con_name == 'exog_min_veh_':
                #    did_find = True
            con = xp.constraint(expr >= dict_con_name_2_LB[con_name], name=con_name)#+ "_ineq")
            cons.append(con)
        lp_prob.addConstraint(cons)

        #if did_find == False:
        #    input('this is odd')

        # --- Add equality constraints ---
        # Group terms for each equality constraint.
        eq_expressions = {}
        for (var_name, con_name), coeff in dict_var_con_2_lhs_eq.items():
            if con_name not in eq_expressions:
                eq_expressions[con_name] = 0
            eq_expressions[con_name] += coeff * var_dict[var_name]

        # Add each equality constraint to the model.
        for con_name, expr in eq_expressions.items():
            if con_name in dict_con_name_2_eq:
                con_eq = xp.constraint(expr == dict_con_name_2_eq[con_name], name=con_name)
                lp_prob.addConstraint(con_eq)

        # --- Solve the LP ---
        self.time_compressor['pre_XP_lp']=time.time()-t2
        start_time = time.time()
        lp_prob.solve()
        end_time = time.time()

        self.lp_time = end_time - start_time
        self.time_compressor['lp_time']=self.lp_time
        t3=time.time()
        self.lp_prob = lp_prob
        self.lp_primal_solution = dict()
        #for var_name, var in var_dict.items():
        #    self.lp_primal_solution[var_name] = lp_prob.getSolution(var_name)

        self.lp_status = lp_prob.getProbStatus()
        self.lp_objective = lp_prob.getObjVal()
        self.lp_dual_solution = dict()

        #for con in lp_prob.getConstraint():
        #    self.lp_dual_solution[con.name] = lp_prob.getDual(con.name)[0]

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
                new_val=self.h_f_2_dual[h][f]
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
            #print('[h,count_orig[h],count_new[h]]')
            #print([h,count_orig[h],count_new[h]])
            #print('---')
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
        #print('count_orig')
        #p#rint(count_orig)
        #print('count_new')
        #print(count_new)
        #input('--at the end of the compresssion this is our size-')

    def check_solution_feasibility(self):
        #print('----')
        #print('----')
        #print('----')
        #print('----')
        #print('----')

        #print('CHEKCING SOLUTION')
        X=self.myLBObj.lp_primal_solution
        #dict_var_name_2_obj=self.dict_PROJ_var_name_2_obj
        dict_var_con_2_lhs_exog=self.dict_var_con_2_lhs_exog
        dict_con_name_2_LB=self.dict_con_name_2_LB
        dict_var_con_2_lhs_eq=self.dict_var_con_2_lhs_eq
        dict_con_name_2_eq=self.dict_con_name_2_eq
        """
        Check if the solution provided in dictionary X is feasible for the LP.
        
        Parameters:
        X : dict
            Dictionary mapping variable names to their proposed values.
        dict_var_con_2_lhs_exog : dict
            Dictionary with keys (var_name, con_name) mapping to the coefficient
            for inequality (exogenous) constraints.
        dict_con_name_2_LB : dict
            Dictionary mapping constraint names to their lower bound (right-hand side) 
            for inequality constraints.
        dict_var_con_2_lhs_eq : dict
            Dictionary with keys (var_name, con_name) mapping to the coefficient
            for equality constraints.
        dict_con_name_2_eq : dict
            Dictionary mapping constraint names to their fixed value for equality constraints.
        tol : float, optional
            Tolerance for checking equality constraint feasibility.
        
        Returns:
        bool: True if the candidate solution is feasible, False otherwise.
        """
        
        feasible = True
        tol=.0001

        # --- Check inequality constraints ---
        ineq_expressions = {}
        for (var_name, con_name), coeff in dict_var_con_2_lhs_exog.items():
            ineq_expressions.setdefault(con_name, 0)
            ineq_expressions[con_name] += coeff * X.get(var_name, 0)

        for con_name, lhs_value in ineq_expressions.items():
            # For each inequality constraint, check if LHS >= lower bound.
            if con_name in dict_con_name_2_LB:
                lb = dict_con_name_2_LB[con_name]
                if lhs_value < lb - tol:
                    
                    for (var, c_name), coeff in dict_var_con_2_lhs_exog.items():
                        if c_name == con_name:
                            var_value = X.get(var, 0)
                            term_value = coeff * var_value
                            if var_value>.0001:
                                print(f"  Variable '{var}': coefficient = {coeff}, value = {var_value}, product = {term_value}")
                
                    print('con_name')
                    print(con_name)
                    #print('h')
                   # print(h)
                    print('self.H_ell_2_pos_neg[h][con_name]')
                    print(self.ell_2_pos_neg[con_name])
                    ell=self.ell_2_pos_neg[con_name]
                    #print('self.H_ell_2_list_leaf[timeGraph][ell]')
                    #print(self.H_ell_2_list_leaf['timeGraph'][ell])
                    print(f"Inequality constraint '{con_name}' is violated: LHS = {lhs_value}, required >= {lb}")
                    feasible = False
                    input('error here ')

        # --- Check equality constraints ---
        eq_expressions = {}
        for (var_name, con_name), coeff in dict_var_con_2_lhs_eq.items():
            eq_expressions.setdefault(con_name, 0)
            eq_expressions[con_name] += coeff * X.get(var_name, 0)
            
        for con_name, lhs_value in eq_expressions.items():
            if con_name in dict_con_name_2_eq:
                target = dict_con_name_2_eq[con_name]
                if abs(lhs_value - target) > tol:
                    print(f"Equality constraint '{con_name}' is violated: LHS = {lhs_value}, required = {target}")
                    feasible = False
                    for (var, c_name), coeff in dict_var_con_2_lhs_eq.items():
                        if c_name == con_name:
                            var_value = X.get(var, 0)
                            term_value = coeff * var_value
                            print(f"  Variable '{var}': coefficient = {coeff}, value = {var_value}, product = {term_value}")
                
                    input('error here')

        if False==feasible:
      #      print("The solution is feasible!")
      #      #input('---')
      #  else:
            print("The solution is NOT feasible.")
            input('--')
            
        return feasible
               