import gurobipy as gp
from gurobipy import GRB
import time
from collections import defaultdict

import numpy as np
# Set desired solver options
import itertools
import gurobipy as gp
from solve_gurobi_lp import solve_gurobi_lp
import sys
sys.path.append("pre_process")
from naive_pre import *

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

    def __init__(self,E,E_2_uv,uv_2_E,Nodes,dict_valid_ineq_name_2_rhs,dict_valid_ineq_name_edge_2_coeff,source_node,sink_node):

        self.E=E
        self.E_2_uv=E_2_uv
        self.uv_2_E=uv_2_E
        self.Nodes=Nodes
        self.source_node=source_node
        self.sink_node=sink_node
        self.dict_valid_ineq_name_2_rhs=dict_valid_ineq_name_2_rhs
        self.dict_valid_ineq_name_edge_2_coeff=dict_valid_ineq_name_edge_2_coeff

        self.dict_var_name_2_obj=dict()
        self.dict_var_con_2_lhs_exog=dict()
        self.dict_var_con_2_lhs_eq=dict()
        self.dict_con_name_2_LB=dict()
        self.dict_con_name_2_eq=dict()
        self.make_vars()
        self.make_flow_in_out()
        self.make_match()
        self.make_valid_ineq()

    def make_vars(self):
        #edge_vars
        self.dict_var_2_obj=dict()
        for e in self.E:
            var_name='EDGE_VAR_'+str(e)
            self.dict_var_2_obj[var_name]=0
        
        self.non_source_sink_nodes=set(self.nodes)-set([self.source_node,self.sink_node])
        for n in self.non_source_sink_nodes:
            var_name='SLACK_FLOW_'+str(n)

            self.dict_var_2_obj[var_name]=1
        
        for q in self.dict_valid_ineq_name_2_rhs:
            var_name='SLACK_VALID_'+str(q)
            self.dict_var_2_obj[var_name]=1


    def make_valid_ineq(self):
        for q in self.dict_valid_ineq_name_edge_2_coeff:
            con_name='Valid_ineq_'+str(q)
            self.dict_con_name_2_LB[con_name]=self.dict_valid_ineq_name_2_rhs[q]
            for e in self.dict_valid_ineq_name_edge_2_coeff[q]:
                var_name='EDGE_VAR_'+str(e)
                coeff=self.dict_valid_ineq_name_edge_2_coeff[q][e]
                self.dict_valid_ineq_name_edge_2_coeff[tuple([var_name,con_name])]=coeff
    def make_match(self):
        
        for uv in self.uv_2_E:
            con_name='match_EQ_'+str(uv)
            self.dict_con_name_2_eq[con_name]=0

            for e in self.E_2_uv:
                var_name='EDGE_VAR_'+str(e)
                my_tup_1=tuple([var_name,con_name])
                self.dict_var_con_2_lhs_eq[my_tup_1]=1

    def make_flow_in_out(self):
        for i in self.non_source_sink_nodes:
            con_name='flow_in_out_'+str(i)
            self.dict_con_name_2_LB[con_name]=0
            var_name='SLACK_FLOW_'+str(i)
            my_tup_slack=tuple([var_name,con_name])
            self.dict_var_con_2_lhs_exog[my_tup_slack]=0
        
        for e in self.E:
            i=e[0]
            j=e[1]
            var_name='EDGE_VAR_'+str(e)

            if i!=self.source_node:
                con_name='flow_in_out_'+str(i)
                my_tup_1=tuple([var_name,con_name])
                self.dict_var_con_2_lhs_exog[my_tup_1]=-1
            if j!=self.sink_node:
                con_name='flow_in_out_'+str(j)
                my_tup_2=tuple([var_name,con_name])
                self.dict_var_con_2_lhs_exog[my_tup_2]=1

class Separ_object:

    def __init__(self,my_graph_based_separ,x,x_mag):
        self.my_graph_based_separ=my_graph_based_separ
        self.x=x
        self.x_mag=x_mag
        self.magnanti_epsilon=.0001
        self.get_vars_cons_keep()
        self.out_solution=solve_gurobi_lp(self.COMP_dict_var_name_2_obj,
                    self.COMP_dict_var_con_2_lhs_exog,
                    self.COMP_dict_con_name_2_LB,
                    self.COMP_dict_var_con_2_lhs_eq,
                    self.COMP_dict_con_name_2_eq)
        self.dual_solution=self.out_solution['dual_solution']
        self.generate_cut()


        
    def get_vars_cons_keep(self):

        self.active_x=dict()
        for uv in self.my_graph_based_separ.uv_2_E:
            u=uv[0]
            v=uv[1]
            var_name='act_'+str(u)+'_'+str(v)
            if self.x[var_name]+self.x_mag[var_name]>0:
                self.active_x[uv]=self.x[var_name]+(self.magnanti_epsilon*self.x_mag[var_name])

        self.vars_keep=[]
        self.cons_keep=[]
        self.cons_keep_eq=[]
        self.valid_cons_keep=[]
        
        self.non_source_sink_nodes=set(self.nodes)-set([self.my_graph_based_separ.source_node,self.my_graph_based_separ.sink_node])
        for n in self.non_source_sink_nodes:
            var_name='SLACK_FLOW_'+str(n)
            self.vars_keep.append(var_name)
            con_name='flow_in_out_'+str(i)
            self.cons_keep.append(con_name)
            #self.dict_var_2_obj[var_name]=1
        
        for uv in self.active_x:
            con_name='match_EQ_'+str(uv)
            self.cons_keep_eq.append(con_name)
            for e in self.my_graph_based_separ.uv_2_e[uv]:
                var_name='EDGE_VAR_'+str(e)
                self.vars_keep.append(var_name)
        for q in self.dict_valid_ineq_name_2_rhs:
            #for e in self.dict_valid_ineq_name_edge_2_coeff[q]:
            dict1=self.dict_valid_ineq_name_edge_2_coeff[q]
            dict2=self.vars_keep
            cardinality = len(dict1.keys() & dict2.keys())
            if cardinality>0:
                var_name='SLACK_VALID_'+str(q)
                con_name='Valid_ineq_'+str(q)
                self.vars_keep.append(var_name)
                self.cons_keep.append(con_name)
                self.valid_cons_keep.append(con_name)

        G=self.my_graph_based_separ
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
            for (var, con), val in G.dict_var_con_2_lhs_exog.items()
            if var in self.COMP_dict_var_name_2_obj and con in self.COMP_dict_con_name_2_eq
        }

        for uv in self.active_x[uv]:
            con_name='match_EQ_'+str(uv)
            u=uv[0]
            v=uv[1]
            var_name='act_'+str(u)+'_'+str(v)
            self.COMP_dict_con_name_2_eq[con_name]=self.x[var_name]+self.x_mag[var_name]

        
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
            dual_i=0
            dual_j=0
            if i!=G.source_node:
                dual_i=self.dual_solution[i_name]
            if j!=G.sink_node:
                dual_j=self.dual_solution[j_name]
            self.E_weight[e]=dual_i-dual_j
    
        self.new_cut_RHS=0
        for q in self.valid_cons_keep:
            dual_val=self.dual_solution[q]
            self.new_cut_RHS+=G.dict_valid_ineq_name_2_rhs*dual_val

            for e in G.dict_valid_ineq_name_edge_2_coeff[q]:
                coeff=G.dict_valid_ineq_name_edge_2_coeff[q]
                self.E_weight[e]-=(dual_val*coeff)
                
        self.new_cut_x_uv_2_coeff=dict()
        for uv in G.uv_2_E:
            self.new_cut_x_uv_2_coeff[uv] = min(self.E_weight[e] for e in G.uv_2_E[uv])

    def return_cut(self):
        return self.new_cut_x_uv_2_coeff,self.new_cut_RHS


class novel_ng_graph_basic:


    def generate_subsets_to_consider(self):
        #generate all subsets 
        my_subsets=set([])
        for u in range(0,self.NC):
            print('u')
            print(u)
            print('self.u_2_NG[u]')
            print(self.u_2_NG[u])


            tmp=self.u_2_NG[u]+[u]
            my_power_set=power_set(tmp)
            for p in my_power_set:
                if len(p)<=self.max_SRI_SET_SIZE:
                    print('p')
                    print(p)
                    tmp=frozenset(p)
                    my_subsets.add(tmp)
        self.subset_2_make_SRI=my_subsets


    def  generate_SRI(self):
        my_SRI=[]
        max_SRI_size=self.max_SRI_SET_SIZE
        for p in self.subset_2_make_SRI:
            max_sz=np.ceil(len(p)/2)
            #print('max_sz')
            #print(max_sz)
            ##print('max_SRI_size')
            #print(max_SRI_size)
            #input('---')
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
    


    def __init__(self,u_2_NG,my_VRP):
        self.my_VRP=my_VRP
        self.u_2_NG=u_2_NG
        self.NC=len(self.u_2_NG)
        self.max_SRI_Divisor=2
        self.max_SRI_SET_SIZE=3
        self.source_node=tuple([self.NC,set([])])
        self.sink_node=tuple([self.NC+1,set([])])
        self.generate_nodes()
        self.generate_edges()
        self.generate_subsets_to_consider()
        self.generate_SRI()
        self.generate_edge_2_SRI_contrib()
        self.non_source_sink_cust=np.arange(0,self.NC)#set(u_2_NG.keys())-set([self.my_VRP.NC,self.my_VRP.NC+1])

    def generate_edge_2_SRI_contrib(self):

        self.dict_valid_ineq_name_2_rhs=dict()
        self.dict_valid_ineq_name_edge_2_coeff=dict()
        my_SRI=self.my_SRI
        for q in my_SRI:
            Nhat=set(q['customers'])
            k=q['my_divisor']
            rhs=q['my_RHS']
            q_name=str(Nhat)+'_'+str(k)+'_'+str(rhs)
            self.dict_valid_ineq_name_2_rhs[q_name]=-np.floor(len(Nhat)/k)
            tmp_dict=dict()
            for e in self.E_2_lost_terms:
                my_lost_terms=self.E_2_lost_terms
                coeff=np.floor(len(my_lost_terms)/k)
                if coeff>0:
                    tmp_dict[e]=-coeff

            self.dict_valid_ineq_name_edge_2_coeff[q_name]=tmp_dict

    def generate_nodes(self):
        self.nodes=[]
        VRP=self.my_VRP
        NC=self.NC

        sink_node=tuple([NC+1,set([])])
        source_node=tuple([NC,set([])])
        self.nodes.append(sink_node)
        self.nodes.append(source_node)
        self.non_source_sink_cust=np.arange(0,NC)
        #self.node_2_ng_allowed
        for u in self.non_source_sink_cust:
            #print('self.u_2_NG[u]')
            #print(self.u_2_NG[u])
            #input('---')
            my_power_set=power_set(self.u_2_NG[u])
            #self.nodes[u]=[]
            for my_sub in my_power_set:
                my_sub=set(sorted(list(my_sub)))
                my_new_node=tuple([u,my_sub])
                self.nodes.append(my_new_node)

    def generate_edges(self):

        VRP=self.my_VRP
        Dist=VRP.dist_mat_full
        self.E_2_uv=dict()
        self.uv_2_E=dict()
        self.E_2_lost_terms=dict()
        self.E=[]
        NC=self.NC
        #print('self.nodes')
        #print(self.nodes)
        for n in self.nodes:
            print('n')
            print(n)
            u=n[0]
            N_n=n[1]
            for w in range(0,NC+2):
                if u!=w and w!=NC and Dist[u,w]<np.inf and w not in N_n :
                    self.make_new_edge(n,w)

    def make_new_edge(self,i,w):
        orig_terms=i[1].union(set([i[0]]))
        this_ng_set=set([])
        if w<self.my_VRP.num_cust:
            this_ng_set=self.u_2_NG[w]
        new_set=set(orig_terms).intersection()
        lost_terms=set(orig_terms)-new_set
        new_set=sorted(list(new_set))
        j=tuple([w,new_set])
        ifroz=tuple([i[0],frozenset(i[1])])
        jfroz=tuple([j[0],frozenset(j[1])])
        e=tuple([ifroz,jfroz])
        print('e')
        print(e)
        self.E.append(e)
        uv=tuple([i[0],w])
        if uv not in self.uv_2_E:
            self.uv_2_E[uv]=[]
        self.uv_2_E[uv].append(e)
        self.E_2_uv[e]=uv
        self.E_2_lost_terms[e]=lost_terms

class complete_separater_end_to_end:

    def __init__(self,MF):
        print('hello')
        self.MF=MF
        self.OPT=dict()
        self.OPT['num_LA_cutting_plane']=4
        self.OPT['do_custom_NG']=False

        self.x_act=MF.my_lower_bound_LP.lp_primal_solution
        self.x_mag=defaultdict(float)
        for uv in self.MF.all_actions_ever_seen:
            self.x_mag[uv]=1
        if self.OPT['do_custom_NG']==True:
            self.make_custom_NG()
        else:
            self.u_2_NG=naive_get_LA_neigh(self.MF.my_VRP,self.OPT['num_LA_cutting_plane'])
            self.u_2_NG=self.u_2_NG[0]
        self.my_NG=novel_ng_graph_basic(self.u_2_NG,self.MF.my_VRP)
        self.my_graph_based_separ=graph_based_separ(self.my_NG.E,self.my_NG.E_2_uv,self.my_NG.uv_2_E,self.my_NG.nodes,self.my_NG.dict_valid_ineq_name_2_rhs,self.my_NG.dict_valid_ineq_name_edge_2_coeff,self.my_NG.source_node,self.my_NG.sink_node)
        self.my_separ=Separ_object(self.my_graph_based_separ,self.x_act,self.x_mag)
        self.objective_cut=self.my_separ.out_solution['objective']
        if self.my_separ.out_solution['objective']>.0001:

            self.add_ineq_to_MF()
       

    def make_custom_NG(self):
        input('make this oine next')

    def add_ineq_to_MF(self):
        cur_count_cutting_planes=self.MF.count_cutting_planes
        new_CP_name='my_valid_ineq_'+str(cur_count_cutting_planes)
        self.MF.all_exog.append(new_CP_name)
        self.MF.exog_name_2_rhs[new_CP_name]=self.my_sep.new_cutting_plane_rhs
        for uv in self.my_sep.new_cuttin_plane_coef:
            u=uv[0]
            v=uv[1]
            primal_var='act_'+str(u)+'_'+str(v)
            val=self.my_sep.new_cutting_plane_coeff
            if abs(val)>0.000001:
                self.MF.action_con_2_contrib[tuple([primal_var,new_CP_name])]=val
        self.MF.exog_name_2_rhs[new_CP_name]=self.my_sep.new_cut_x_uv_2_coeff
        