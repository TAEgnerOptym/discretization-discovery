

EPSILON=0.0001
class projector:

    def __init__(self,full_solver):
        self.MF=full_solver
        self.P_star=set()
        self.P_star=self.P_star.union(self.MF.actions_incumbant)
        x_lb=self.MF.my_lower_bound.lp_primal_sol
        for p in self.MF.actions_2_cost:
            if x_lb[p]>EPSILON:
                self.P_star.add(p)
        
        if False== hasattr('my_orig_lp_projector',self.MF.my_LP):
            self.make_lp_object_structure()
        
        self.read_and_modify()
        

    def make_lp_object_structure(self):
        self.make_uv_2_keep()
        self.make_lp()
    def read_and_modify(self):

        my_lp_object=self.MF.my_orig_lp_projector.lp.copy()

        for p in set(self.MF.actions_2_cost)-self.P_star:
            kill_vars=kill_vars-my_lp_object.uv_2_vars_keep[p]
        
        self.dict_2_ub=self.MF.projector_dict_2_ub.copy()
        for var in kill_vars:
            self.dict_2_ub[var]=0

    def make_LP(self): 
        self.make_obj()
        self.make_A_ineq()
        self.make_flow_in_out()
        self.make_lb_ub()
    
    def make_uv_2_keep(self):
        self.uv_2_var_keep=dict()
        self.uv_2_con_keep=dict()
        self.all_killable_vars=[]
        self.all_killable_cons=[]
        for u in self.MF.my_vrp.Nc+2:
            for v in self.MF.my_vrp.Nc+2:
                my_act='act'+str(u)+'_'+str(v)
                if my_act in self.MF.act_to_cost:
                    self.uv_2_var_keep[(u,v)]=[my_act]
                    name_uv_cap='cap_uv_'+str(u)+'_'+str(v)
                    name_uv_time='time_uv_'+str(u)+'_'+str(v)
                    self.uv_2_con_keep[(u,v)]=[name_uv_cap,name_uv_time]
        for h in self.MF.h_ij_2_p:
            for ij in self.MF.h_ij_2_p:
                p=self.MF.h_ij_2_p[h][ij]
                if p!=self.MF.null_action:
                    var_name='var_edge_h='+str(h)+'_i=_'+str(i)+' j '+str(j)
                    self.uv_2_var_keep[p]=var_name
        
        
        all_killable_vars=[]
        for uv in self.uv_2_var_keep:
            all_killable_vars=all_killable_vars+self.uv_2_var_keep[uv]
            all_killable_cons=all_killable_cons+self.uv_2_con_keep[uv]

        
