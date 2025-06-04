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

sys.path.append("pre_process")
from naive_pre import *
#from valid_ineq_helper_ng_la import ng_help_valid_ineq
from backup_valid_ineq_helper_ng_la import ng_help_valid_ineq
verbose=False
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

class graph_based_separ:

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
        DEBUG_SAVE_ME=[]
        for i in self.non_source_sink_nodes:
            

            con_name='flow_in_out_'+str(i)
            self.dict_con_name_2_LB[con_name]=0
            var_name='SLACK_FLOW_'+str(i)
            my_tup_slack=tuple([var_name,con_name])
            self.dict_var_con_2_lhs_exog[my_tup_slack]=1
            if i==(5, frozenset({8, 1, 3, 0})):
                print('con_name')
                print(con_name)
                print('FOUND ME')
                DEBUG_SAVE_ME=con_name
        if 1<0:
            i=(5, frozenset({8, 1, 3, 0}))
            con_name='flow_in_out_'+str(i)
            print('i in self.non_source_sink_nodes')
            print(i in self.non_source_sink_nodes)
            if con_name not in self.dict_con_name_2_LB:
                print('con_name')
                print(con_name)
                print('DEBUG_SAVE_ME')
                print(DEBUG_SAVE_ME)
                input('wrotng')
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

class Separ_object:



    def __init__(self,my_graph_based_separ,x,x_mag,allow_slack_on_nodes):
        self.my_graph_based_separ=my_graph_based_separ
        self.x=x
        self.allow_slack_on_nodes=allow_slack_on_nodes
        self.non_source_sink_nodes=self.my_graph_based_separ.non_source_sink_nodes#et(self.nodes)-set([self.my_graph_based_separ.source_node,self.my_graph_based_separ.sink_node])
        
        
        self.x_mag=x_mag
        self.magnanti_epsilon=.00001
        self.get_vars_cons_keep()
        if 1<0:
            i=(5, frozenset({8, 1, 3, 0}))
            print('i in self.non_source_sink_nodes')
            print(i in self.non_source_sink_nodes)
            con_name='flow_in_out_'+str(i)
            print('con_name not in self.my_graph_based_separ.dict_con_name_2_LB')
            print(con_name in self.my_graph_based_separ.dict_con_name_2_LB)
            print('con_name not in self.cons_keep')
            print(con_name  in self.cons_keep)
            print('con_name in G.dict_con_name_2_LB')
            print(con_name in self.my_graph_based_separ.dict_con_name_2_LB)
            print('con_name in self.COMP_dict_con_name_2_LB')
            print( con_name in self.COMP_dict_con_name_2_LB)
            input('---')
        #print(self.cons_keep)
        #input('---')
        self.out_solution=solve_gurobi_lp(self.COMP_dict_var_name_2_obj,
                    self.COMP_dict_var_con_2_lhs_exog,
                    self.COMP_dict_con_name_2_LB,
                    self.COMP_dict_var_con_2_lhs_eq,
                    self.COMP_dict_con_name_2_eq)
        self.dual_solution=self.out_solution['dual_solution']
       
        #print('objective')
        #print(self.out_solution['objective'])
        #input('--')
        self.lp_primal_new=self.out_solution['primal_solution']
        self.generate_cut()
        do_add_extra_cuts=False
        self.aux_cuts=[]
        if do_add_extra_cuts==True:
            active_q_terms=[]
            for q_full in self.valid_cons_keep:
                dual_val=self.dual_solution[q_full]
                q=q_full[11:]
                if abs(dual_val)>0.001:
                    active_q_terms.append(q_full)
                    print('q  '+str(q)+' val'+ str(dual_val))
            #if len(active_q_terms)>1:
            for q in active_q_terms: 
                self.make_aux_cut(active_q_terms,q)
        #print('printing solution')
        if verbose:
            my_con='Valid_ineq_frozenset({0, 1, 2})_2_1.0'
            for vc in self.my_graph_based_separ.dict_var_con_2_lhs_exog:
                v=vc[0]
                c=vc[1]
                if c==my_con:
                    val=self.my_graph_based_separ.dict_var_con_2_lhs_exog[vc]
                    print(str(vc)+" -> "+str(val))
            #input('---')
            orig_rhs=self.my_graph_based_separ.dict_con_name_2_LB[my_con]
            exog=self.my_graph_based_separ.dict_var_con_2_lhs_exog
            tot_contrib=0
            for p in self.lp_primal_new:
                if self.lp_primal_new[p]>0.01:
                    my_tup=tuple([p,my_con])
                    weight=0
                    x_val=self.lp_primal_new[p]
                    if my_tup in exog:
                        weight=float(exog[my_tup])
                    contrib=weight*x_val
                    print(p+': val '+str(x_val)+ '     weight       ' +str(weight)+'      contrib     '+str(contrib) )
                    if p.startswith('SLACK')==False:
                        tot_contrib=tot_contrib+contrib
            #print('tot_contrib')
            #print(tot_contrib)
            #print('orig_rhs')
            #print(orig_rhs)
            #print('objective')
            #print(self.out_solution['objective'])
            #input('---')

    def make_aux_cut(self,active_q_terms,q_this):

        con_remove=set(active_q_terms)-set([q_this])

        NEW_COMP_dict_var_con_2_lhs_exog = {
            (var, con): val
            for (var, con), val in self.COMP_dict_var_con_2_lhs_exog.items()
            if con not in con_remove
        }

        NEW_COMP_dict_con_name_2_LB = {
            con: lb
            for con, lb in self.COMP_dict_con_name_2_LB.items()
            if con not in con_remove
        }
        
        out_solution=solve_gurobi_lp(self.COMP_dict_var_name_2_obj,
                    NEW_COMP_dict_var_con_2_lhs_exog,
                    NEW_COMP_dict_con_name_2_LB,
                    self.COMP_dict_var_con_2_lhs_eq,
                    self.COMP_dict_con_name_2_eq)
        if out_solution['objective']>0.0001:
            #print('creating a cut from objective ')
            #print(out_solution['objective'])
            #print('q_this')
            #print(q_this)
            dual_solution=self.out_solution['dual_solution']

            lp_primal_new=self.out_solution['primal_solution']
            [new_rhs,new_cut_vec]=self.generate_cut_aux(dual_solution,q_this)
            self.aux_cuts.append([new_rhs,new_cut_vec,q_this])
        #else:
        #    print('NO CUT from  '+str(q_this) )
    def generate_cut_aux(self,dual_solution,q_full):
        
        new_cut_RHS=0
        new_cut_x_uv_2_coeff=dict()
        G=self.my_graph_based_separ

        E_weight=dict()
        for e in G.E:
            i=e[0]
            j=e[1]
            i_name='flow_in_out_'+str(i)
            j_name='flow_in_out_'+str(j)      
            dual_i=0
            dual_j=0
            if i!=G.source_node and i_name in dual_solution:
                dual_i=dual_solution[i_name]
            if j!=G.sink_node and j_name in dual_solution:
                dual_j=dual_solution[j_name]
            E_weight[str(e)]=-dual_i+dual_j
    
        new_cut_RHS=0
        dual_val=dual_solution[q_full]
        q=q_full[11:]

        new_cut_RHS=G.dict_valid_ineq_name_2_rhs[q]*dual_val#+self.e
        
        for e in G.dict_valid_ineq_name_edge_2_coeff[q]:
            coeff=G.dict_valid_ineq_name_edge_2_coeff[q][e]
            E_weight[str(e)]-=(dual_val*coeff)
                
        new_cut_x_uv_2_coeff=dict()
        for uv in G.uv_2_E:
            if uv[0]!=uv[1]:
                new_cut_x_uv_2_coeff[uv] = -min(E_weight[str(e)] for e in G.uv_2_E[uv])

        return [new_cut_RHS,new_cut_x_uv_2_coeff]


    def get_vars_cons_keep(self):
        G=self.my_graph_based_separ

        self.active_x=dict()
        for uv in self.my_graph_based_separ.uv_2_E:
            u=uv[0]
            v=uv[1]
            var_name='act_'+str(u)+'_'+str(v)
            #if u!=v and self.x[var_name]+self.x_mag[var_name]>0:
            if u!=v:
                self.active_x[uv]=self.x[var_name]+(self.magnanti_epsilon*self.x_mag[var_name])
                #print('self.active_x[uv]')
                #print(self.active_x[uv])
                #print('uv')
                #print(uv)
                #input('ADDING HERE')

        #print('len')
        self.vars_keep=[]
        self.cons_keep=[]
        self.cons_keep_eq=[]
        self.valid_cons_keep=[]
        
        for i in self.non_source_sink_nodes:
            if self.allow_slack_on_nodes==True:
                var_name='SLACK_FLOW_'+str(i)
                self.vars_keep.append(var_name)
            con_name='flow_in_out_'+str(i)
            #print('con_name')
            #print(con_name)
            #input('keeping')
            self.cons_keep.append(con_name)
            #self.dict_var_2_obj[var_name]=1
        
        for uv in self.active_x:
            con_name='match_EQ_'+str(uv)
            self.cons_keep_eq.append(con_name)
            for e in self.my_graph_based_separ.uv_2_E[uv]:
                var_name='EDGE_VAR_'+str(e)
                self.vars_keep.append(var_name)
        num_ineq_found=0
        set_2={v.replace("EDGE_VAR_", "") for v in self.vars_keep}

        # Step 1: Build a shared key-to-string map
        all_keys = set()
        for dict1a in G.dict_valid_ineq_name_edge_2_coeff.values():
            all_keys.update(dict1a.keys())

        key_to_str = {k: str(k) for k in all_keys}

        # Step 2: Use it during your loop
        for q in tqdm(G.dict_valid_ineq_name_2_rhs, "Getting Vars Cons"):
            
            dict1a = G.dict_valid_ineq_name_edge_2_coeff[q]
            
            # Use precomputed str-mapping
            dict1 = {key_to_str[k]: v for k, v in dict1a.items()}
            set1 = set(dict1.keys())

            cardinality = len(set1 & set_2)

            if cardinality > 0:
                var_name = 'SLACK_VALID_' + str(q)
                con_name = 'Valid_ineq_' + str(q)
                self.vars_keep.append(var_name)
                self.cons_keep.append(con_name)
                self.valid_cons_keep.append(con_name)
                num_ineq_found += 1


        self.COMP_dict_var_name_2_obj= {k: G.dict_var_2_obj[k] for k in self.vars_keep}

        self.COMP_dict_con_name_2_LB={k: G.dict_con_name_2_LB[k] for k in self.cons_keep }
        self.COMP_dict_con_name_2_eq={k: G.dict_con_name_2_eq[k] for k in self.cons_keep_eq }

        self.COMP_dict_var_con_2_lhs_exog = {
            (var, con): val
            for (var, con), val in G.dict_var_con_2_lhs_exog.items()
            if var in self.COMP_dict_var_name_2_obj and con in self.COMP_dict_con_name_2_LB
        }

        self.COMP_dict_var_con_2_lhs_eq = {
            (var, con): val
            for (var, con), val in G.dict_var_con_2_lhs_eq.items()
            if var in self.COMP_dict_var_name_2_obj and con in self.COMP_dict_con_name_2_eq
        }

        for uv in self.active_x:
            con_name='match_EQ_'+str(uv)
            u=uv[0]
            v=uv[1]
            var_name='act_'+str(u)+'_'+str(v)
            self.COMP_dict_con_name_2_eq[con_name]=self.active_x[uv]#self.x[var_name]+self.x_mag[var_name]

    def generate_cut(self):
        
        self.new_cut_RHS=0
        self.new_cut_x_uv_2_coeff=dict()
        G=self.my_graph_based_separ

        self.E_weight=dict()
        for e in G.E:
            i=e[0]
            j=e[1]
            i_name='flow_in_out_'+str(i)
            j_name='flow_in_out_'+str(j)      
            var_edge_name='EDGE_VAR_'+str(e)
            if var_edge_name not in self.lp_primal_new :
                print('i')
                print(i)
                print('j')
                print(j)
                input('error here')
                continue
            dual_i=0
            dual_j=0

            if i!=G.source_node:# and i_name in self.dual_solution:
                if i not in self.non_source_sink_nodes:
                    print('i')
                    print(i)
                    input('ok this is not ok')
                if i_name not in self.dual_solution:
                    print('i')
                    print(i)
                    print('i_name')
                    print(i_name)
                    print('i not in self.non_source_sink_nodes')
                    print()
                    input('ok this is not ok either')
                dual_i=self.dual_solution[i_name]
                if i_name not in self.dual_solution:
                    print('error')
                    print('i_name')
                    print(i_name)
                        
                    input('error  2')
            if j!=G.sink_node:# and j_name in self.dual_solution:
                if j_name not in self.dual_solution:
                    print('error')
                    print('j_name')
                    print(j_name)
                        
                    input('error ')
                dual_j=self.dual_solution[j_name]
            if dual_i<-.0001 or dual_j<-0.0001:
                input('not sensical')
            self.E_weight[str(e)]=-dual_i+dual_j
    
        self.new_cut_RHS=0
        #print('G.E')
        #print(G.E)
        #print('G.E[3]')
        #print(G.E[3])
        for q_full in self.valid_cons_keep:
            dual_val=self.dual_solution[q_full]
            q=q_full[11:]
            if G.dict_valid_ineq_name_2_rhs[q]>0:
                input('ok wrong i think')
            self.new_cut_RHS+=G.dict_valid_ineq_name_2_rhs[q]*dual_val
            if abs(dual_val)>0.001:
                print(str(q)+' = ' +str(dual_val))
            for e in G.dict_valid_ineq_name_edge_2_coeff[q]:
                coeff=G.dict_valid_ineq_name_edge_2_coeff[q][e]
                self.E_weight[str(e)]-=(dual_val*coeff)
                if dual_val<-0.0001:
                    print('dual_val')
                    print(dual_val)
                    input('error')
        self.new_cut_x_uv_2_coeff=dict()
        num_errors=0
        num_correct=0
        for uv in G.uv_2_E:
            if uv[0]!=uv[1]:
                pi_uv = min(self.E_weight[str(e)] for e in G.uv_2_E[uv])
                self.new_cut_x_uv_2_coeff[uv]=-pi_uv
                con_name='match_EQ_'+str(uv)
                if con_name in self.dual_solution:
                    val=self.dual_solution[con_name]
                    if abs(pi_uv-val)>0.01:
                        print('val')
                        print(val)
                        print('self.new_cut_x_uv_2_coeff[uv]')
                        print(self.new_cut_x_uv_2_coeff[uv])
                        print('uv')
                        print(uv)
                        print('len(G.uv_2_E[uv])')
                        print(len(G.uv_2_E[uv]))
                        input('error here')
                        num_errors=num_errors+1
                    else:
                        num_correct=num_correct+1
        if num_errors>0:
            print([num_errors,num_correct])
            input('num_errors,num_correct')
    def return_cut(self):
        return self.new_cut_x_uv_2_coeff,self.new_cut_RHS


    
class complete_separater_end_to_end:

    def update_given_solution(self,running_average_sol):

        filtered_duals_aux = {
            k: v
            for k, v in self.MF.my_lower_bound_LP.lp_dual_solution.items()
            if k.startswith('my_valid_ineqAUX_') and  abs(v) > 1e-8 #.contains("") # # or v != 0 if exact zeros are fine
        }        
    
        filtered_duals_reg = {
            k: v
            for k, v in self.MF.my_lower_bound_LP.lp_dual_solution.items()
            if  k.startswith('my_valid_ineq_') and abs(v) > 1e-8  # or v != 0 if exact zeros are fine
        }        
        print('filtered_duals_aux')
        print(filtered_duals_aux)
        print('filtered_duals_reg')
        print(filtered_duals_reg)
        #input('my dict')
        self.x_act=self.MF.my_lower_bound_LP.lp_primal_solution
        if len(running_average_sol)==0:
            for uv in self.my_graph_based_separ.uv_2_E:
                u=uv[0]
                v=uv[1]
                
                if u==v:
                    continue
                var_name='act_'+str(u)+'_'+str(v)
                val=self.x_act[var_name]
                self.running_avg[var_name]=self.running_avg[var_name]*.95
                self.running_avg[var_name]+=val*0.05

                if verbose and val>0:
                    print(var_name+'  '+str(val))
        else:
            self.running_avg=running_average_sol
        self.x_mag=self.running_avg

        if verbose:
            print('self.running_avg')
            print(self.running_avg)
            input('sol above from input')
        self.OPT['allow_slack_on_nodes']=True
        #if len(self.MF.history_dict['sum_lp_value_project'])>0 and self.MF.history_dict['sum_lp_value_project'][-1]<0.001:
        #    self.OPT['allow_slack_on_nodes']=False
        #for uv in self.MF.all_actions_ever_seen:
        #    self.x_mag[uv]=1
        self.my_separ=Separ_object(self.my_graph_based_separ,self.x_act,self.x_mag,self.OPT['allow_slack_on_nodes'])
        self.objective_cut=self.my_separ.out_solution['objective']
        print('self.OPT[allow_slack_on_nodes]')
        print(self.OPT['allow_slack_on_nodes'])
        if self.my_separ.out_solution['objective']>.0001:
            
            self.add_ineq_to_MF()
            self.add_aux_cuts_to_MF()
        print('objective')
        print(self.my_separ.out_solution['objective'])
        print('ALL DONE')

    def add_aux_cuts_to_MF(self):
        #print('aux cuts')
        for k in range(0,len(self.my_separ.aux_cuts)):
            #print('k:  '+str(k))
            new_cut_x_uv_2_coeff=self.my_separ.aux_cuts[k][1]
            new_cut_RHS=self.my_separ.aux_cuts[k][0]
            q_this=self.my_separ.aux_cuts[k][2]
            cur_count_cutting_planes=self.MF.count_cutting_planes
            new_CP_name='my_valid_ineqAUX_'+str(cur_count_cutting_planes)+'__'+str(q_this)
            self.MF.all_exog.append(new_CP_name)
            self.MF.count_cutting_planes+=1
            self.MF.exog_name_2_rhs[new_CP_name]=new_cut_RHS-self.epsilon_slack_valid/10
            print('new_CP_name')
            print(new_CP_name)
            for uv in new_cut_x_uv_2_coeff:
                u=uv[0]
                v=uv[1]
                primal_var='act_'+str(u)+'_'+str(v)
                val=new_cut_x_uv_2_coeff[uv]
                if abs(val)>0.000001:
                    self.MF.action_con_2_contrib[tuple([primal_var,new_CP_name])]=val
        
        #input('doner ')

    def make_custom_NG(self, K=6):
        x = self.MF.my_lower_bound_LP.lp_primal_solution
        Nc = self.MF.my_VRP.num_cust

        # Step 1: Build a directed graph in NetworkX
        G = nx.Graph()

       #for u in range(Nc+2):
       #     for v in range(Nc+2):
       #         var_name = f'act_{u}_{v}'
       #         if var_name in x and x[var_name]>0.0001:
       #             print(var_name+"   "+str(x[var_name]))
        for u in range(Nc):
            for v in range(Nc):
                if u == v:
                    continue
                var_name = f'act_{u}_{v}'
                if var_name in x and x[var_name]>0.0001:
                    
                    weight = 1/(.001+x[var_name])
                    G.add_edge(u, v, weight=weight)
                    #print(var_name+"   "+str(x[var_name]))
        # Step 2: Compute all-pairs shortest path lengths
        all_pairs_dist = dict(nx.all_pairs_dijkstra_path_length(G))

        # Step 3: For each node, find K nearest neighbors by shortest path distance
        nearest_neighbors = []
        for u in range(0,Nc):
            dist_u = all_pairs_dist.get(u, {})
            neighbors = [(v, d) for v, d in dist_u.items() if v != u]
            neighbors.sort(key=lambda item: item[1])
            tmp = [v for v, _ in neighbors[:K]]
            nearest_neighbors.append(tmp)
        # Add dummy entries for depot-like nodes if needed
        #print('nearest_neighbors')
        #print(nearest_neighbors)
        #input('---')
        self.custom_NG = nearest_neighbors
        #self.custom_NG[Nc] = []
        #self.custom_NG[Nc+1] = []
        self.u_2_NG=self.custom_NG
    
    def __init__(self,MF,do_custom_NG=False,num_LA_cutting_plane=8,max_SRI_Divisor=3,max_SRI_SET_SIZE=5):
        self.MF=MF
        self.OPT=dict()
        self.OPT['num_LA_cutting_plane']=num_LA_cutting_plane
        self.OPT['max_SRI_Divisor']=max_SRI_Divisor
        self.OPT['max_SRI_SET_SIZE']=max_SRI_SET_SIZE
        self.OPT['allow_slack_on_nodes']=True
        self.OPT['do_custom_NG']=do_custom_NG
        self.epsilon_slack_valid=.00001
        #self.x_act=MF.my_lower_bound_LP.lp_primal_solution
        #self.x_mag=defaultdict(float)
        #for uv in self.MF.all_actions_ever_seen:
        #    self.x_mag[uv]=1
        if self.OPT['do_custom_NG']==True:
            self.ORIG_u_2_NG=naive_get_LA_neigh(self.MF.my_VRP,self.OPT['num_LA_cutting_plane'])
            self.ORIG_u_2_NG=self.ORIG_u_2_NG[0]
            self.make_custom_NG(self.OPT['num_LA_cutting_plane'])
           ## for u in range(0,len(self.ORIG_u_2_NG)):
           #     print('u'+str(u))
           #     print('new ')
           #     print(sorted(self.u_2_NG[u]))
           ##     print('orig')
           #     print(sorted(self.ORIG_u_2_NG[u]))
            #    print('---')
                #input('--')
        else:
            #input('i dont reall want to be here')
            self.u_2_NG=naive_get_LA_neigh(self.MF.my_VRP,self.OPT['num_LA_cutting_plane'])
            self.u_2_NG=self.u_2_NG[0]
        #self.my_NG=novel_ng_graph_basic(self.u_2_NG,self.MF.my_VRP,self.OPT['max_SRI_Divisor'],self.OPT['max_SRI_SET_SIZE'])
        self.my_NG=ng_help_valid_ineq(self.MF.my_VRP,self.u_2_NG,self.OPT['max_SRI_Divisor'],self.OPT['max_SRI_SET_SIZE'])
        
        self.my_graph_based_separ=graph_based_separ(self.my_NG)
        self.running_avg=dict()
        for uv in self.my_graph_based_separ.uv_2_E:
            u=uv[0]
            v=uv[1]
            if u==v:
                continue
            var_name='act_'+str(u)+'_'+str(v)

            self.running_avg[var_name]=0
            
        if 0>1:
            self.my_separ=Separ_object(self.my_graph_based_separ,self.x_act,self.x_mag)
            self.objective_cut=self.my_separ.out_solution['objective']
            print('self.objective_cut')
            print(self.objective_cut)
            print('----')
            if self.my_separ.out_solution['objective']>.0001:
                self.add_ineq_to_MF()
       

    #def make_custom_NG(self):
    #    input('make this oine next')

    def print_cut(self):

        print('printinig component dual')
        valid_ineq_comp_def=[]

        for q_full in self.my_separ.valid_cons_keep:
            dual_val=self.my_separ.dual_solution[q_full]
            q=q_full[11:]
            if abs(dual_val)>0:
                print(str(q)+'   '+str(dual_val))
                valid_ineq_comp_def.append([float(dual_val),str(q)])
                #self.new_cut_RHS+=G.dict_valid_ineq_name_2_rhs[q]
        #input('input coeff')
        #print('PRINTINIG CUT')
        cut_desc=[]
        for uv in self.my_separ.new_cut_x_uv_2_coeff:
            u=uv[0]
            v=uv[1]
            primal_var='act_'+str(u)+'_'+str(v)
            val=self.my_separ.new_cut_x_uv_2_coeff[uv]
            primal_val=self.MF.my_lower_bound_LP.lp_primal_solution[primal_var]
            if self.running_avg[primal_var]+primal_val+abs(val)>0.000001:
                #print('u,v,cut_coeff,primal_val,runAvgVal')
                print([u,v,val,primal_val,self.running_avg[primal_var]])
                cut_desc.append([int(u),int(v),float(val),float(primal_val),float(self.running_avg[primal_var])])
        #print('self.MF.exog_name_2_rhs[self.new_CP_name]')
        #print(self.MF.exog_name_2_rhs[self.new_CP_name])
        #print('self.new_CP_name')
        #print(self.new_CP_name)
        #print('printed_cut_above')
        out_terrms=dict()
        out_terrms['cut_desc']=cut_desc
        out_terrms['valid_ineq_comp_def']=valid_ineq_comp_def
        with open('../ALL_JSON_BIG/R101_stuff/CUT_FILE_'+str(self.MF.count_cutting_planes)+'.json', 'w') as file:
            json.dump(out_terrms, file)
        #input('---')
    def add_ineq_to_MF(self):
        cur_count_cutting_planes=self.MF.count_cutting_planes
        self.new_CP_name='my_valid_ineq_'+str(cur_count_cutting_planes)
        self.MF.all_exog.append(self.new_CP_name)
        self.MF.count_cutting_planes+=1
        self.MF.exog_name_2_rhs[self.new_CP_name]=self.my_separ.new_cut_RHS-self.epsilon_slack_valid*2
        DEBUG_RHS=self.my_separ.new_cut_RHS
        DEBUG_LHS=0
        for uv in self.my_separ.new_cut_x_uv_2_coeff:
            u=uv[0]
            v=uv[1]
            primal_var='act_'+str(u)+'_'+str(v)
            val=self.my_separ.new_cut_x_uv_2_coeff[uv]
            if abs(val)>0.000001:
                self.MF.action_con_2_contrib[tuple([primal_var,self.new_CP_name])]=val
            if u!=v:
                xuv=self.MF.my_lower_bound_LP.lp_primal_solution[primal_var]
            #print('xuv  '+str(xuv)+'   val '+str(val)+'   xuv*val:  '+str(xuv*val))
                DEBUG_LHS+=xuv*val
        
        if abs((DEBUG_RHS-DEBUG_LHS)-self.objective_cut)>.01:
            print('DEBUG_RHS')
            print(DEBUG_RHS)
            print('DEBUG_LHS')
            print(DEBUG_LHS)
            print('self.objective_cut')
            print(self.objective_cut)
            input('cut coefficients not written correctly')
        #else:
        #    print('GREAT JOB')
        #    print('DEBUG_RHS')
        ##    print(DEBUG_RHS)
         #   print('DEBUG_LHS')
         #   print(DEBUG_LHS)
            #input('FOUND')
        #if verbose:
        #print('DEBUG_RHS')
        #print(DEBUG_RHS)
        #print('DEBUG_LHS')
        #print(DEBUG_LHS)
        #print('self.objective_cut')
        #print(self.objective_cut)
        #input('--')
        #self.print_cut()