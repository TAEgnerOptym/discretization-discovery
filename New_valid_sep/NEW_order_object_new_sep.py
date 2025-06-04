import numpy as np
class NEW_order_object_new_sep:
    def __init__(self,my_order_name,pred_order,my_instance):
        self.my_order_name=my_order_name
        self.my_instance=my_instance

        self.pred_order=pred_order
        self.u=my_order_name[0]
        self.w=my_order_name[1]
        self.dist_serve_add=self.my_instance.dist_mat_full[self.u,self.w]

        self.compute_cost()#dict()
        self.compute_early_depart_wo_wait()
        self.compute_latest_depart()
        #if self.my_instance.my_params['DEBUG_NG_turn_off_CLEAN']==False:
        if self.lateDepart>self.my_instance.early_start_full[self.u]:
            self.cost=np.inf

        self.dem_in_arc=0
        #or u in self.my_order_name:
        #    self.dem_in_arc+=self.my_instance.dem_full[u]
        if pred_order==None:
            self.dem_in_arc = sum(self.my_instance.dem_full[u] for u in self.my_order_name)
        else:
            self.dem_in_arc=self.my_instance.dem_full[self.u]+pred_order.dem_in_arc
        if self.dem_in_arc>self.my_instance.vehicle_capacity:
            self.cost=np.inf

    def extend_order(self,new_u):
        
        trm1=[new_u]
        NEW_my_order_name=trm1+self.my_order_name
        my_new_order=NEW_order_object_new_sep(NEW_my_order_name,self,self.my_instance)

        return my_new_order

    def compute_cost(self):
        self.cost=0#self.my_instance.dist_mat_full[u,v]
        u=self.u
        w=self.w

        if self.pred_order!=None:
            self.cost=self.pred_order.cost+self.dist_serve_add#self.my_instance.dist_mat_full[u,w]
        else:
            self.cost=self.dist_serve_add#self.my_instance.dist_mat_full[u,w]
            
    def compute_early_depart_wo_wait(self):
        
        early_depart_prev=np.inf#self.my_instance.early_start[self.w]
        
        if self.pred_order!=None:
            early_depart_prev=self.dist_serve_add+self.pred_order.early_depart_wo_wait
        self.early_depart_prev=early_depart_prev
        trm1=self.dist_serve_add+early_depart_prev
        trm2=self.my_instance.early_start_full[self.u]
        self.early_start_u=trm2
        self.early_depart_wo_wait=min([trm1,trm2])
        self.earlyArrival=self.early_depart_wo_wait-self.dist_serve_add
        
    def compute_latest_depart(self):
        late_depart_prev=self.my_instance.late_start_full[self.w]
        if self.pred_order!=None:
            late_depart_prev=self.pred_order.lateDepart
        self.late_depart_prev=late_depart_prev
        trm1=self.my_instance.late_start_full[self.u]
        trm2=self.dist_serve_add+late_depart_prev
        self.lateDepart=max([trm1,trm2])
        self.late_start_u=trm1
