import numpy as np
class NEW_order_object_new_sep_backwards:
    def __init__(self,my_order_name,pred_order,my_instance):
        self.my_order_name=my_order_name
        self.my_instance=my_instance

        self.pred_order=pred_order
        self.v=my_order_name[-1]
        self.w=my_order_name[-2]
        self.u=my_order_name[0]

        self.dist_serve_add=self.my_instance.dist_mat_full[self.w,self.v]
        #print('dist_serve_add')
        #print(self.dist_serve_add)
        self.compute_cost()#dict()
        self.compute_early_arrival()
        self.compute_latest_depart()
        ##print(self.cost)
        #if self.my_instance.my_params['DEBUG_NG_turn_off_CLEAN']==False:
        if self.lateDepart>self.my_instance.early_start_full[self.u]:#[self.v]:
            self.cost=np.inf
        #print('cost2')
        #print(self.cost )
        self.dem_in_arc=0
        #or u in self.my_order_name:
        #    self.dem_in_arc+=self.my_instance.dem_full[u]
        if pred_order==None:
            self.dem_in_arc = sum(self.my_instance.dem_full[u] for u in self.my_order_name)
        else:
            self.dem_in_arc=self.my_instance.dem_full[self.v]+pred_order.dem_in_arc
        if self.dem_in_arc>self.my_instance.vehicle_capacity:
            self.cost=np.inf
        #print('cost3')
        #print('self.my_order_name')
        #print(self.my_order_name)
        #print('self.cost ')
        #print(self.cost )
    def extend_order(self,new_final):
        
        trm1=[new_final]
        NEW_my_order_name=self.my_order_name+trm1
        my_new_order=NEW_order_object_new_sep_backwards(NEW_my_order_name,self,self.my_instance)

        return my_new_order

    def compute_cost(self):
        self.cost=0#self.my_instance.dist_mat_full[u,v]
        

        if self.pred_order!=None:
            self.cost=self.pred_order.cost+self.dist_serve_add#self.my_instance.dist_mat_full[u,w]
        else:
            self.cost=self.dist_serve_add#self.my_instance.dist_mat_full[u,w]
            
    def compute_early_arrival(self):
        
        earlyArrival=self.my_instance.early_start_full[self.w]
        
        if self.pred_order!=None:
            earlyArrival=self.pred_order.earlyArrival
        trm1=-self.dist_serve_add+earlyArrival
        trm2=self.my_instance.early_start_full[self.v]
        self.earlyArrival=min([trm1,trm2])
        
    def compute_latest_depart(self):
        late_depart_prev=self.my_instance.late_start_full[self.w]
        if self.pred_order!=None:
            late_depart_prev=self.pred_order.lateDepart

        trm1=self.my_instance.late_start_full[self.v]+self.cost
        trm2=late_depart_prev
        self.lateDepart=max([trm1,trm2])

        #print(self.my_order_name)
        #print('self.lateDepart')
        #print(self.lateDepart)
        #print('self.cost')
        #print(self.cost)
        #print('self.my_instance.late_start_full[self.v]')
        #print(self.my_instance.late_start_full[self.v])
        #print('trm1,trm2')
        #print([trm1,trm2])
        #print('self.u')
        #print(self.u)
        #print('self.v')
        #print(self.v)
        
