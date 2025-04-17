class LA_arc_time_maker():

    def __init__(input_dict,num_LA_neigh):
        self.cust_2_dem=input_dict['cust_2_dem']
        self.loc_loc_2_travel_plus_service=input_dict['loc_loc_2_travel_plus_service']
        self.loc_loc_2_travel_wo_service=input_dict['loc_loc_2_travel_wo_service']
        self.time_init=input_dict['time_init']
        self.vehicle_capacity=input_dict['vehicle_capacity']
        self.num_LA_neigh=num_LA_neigh
        self.gen_LA_neigh()
    
    def gen_LA_neigh(self):