
    def THINK_get_p_h_mapping(self):
        num_dec_keep=2
        self.non_null_act_2_primal=dict()
        self.graph_non_null_act_2_dual=dict()
        lp_primal=self.my_lower_bound_LP.lp_primal_solution
        lp_dual=self.my_lower_bound_LP.lp_dual_solution
        
        for p in self.all_non_null_action:
            self.non_null_act_2_primal[p]=lp_primal[p]
            
        special_tup_list=dict()
        for h in self.graph_names:
            self.graph_non_null_act_2_dual[h]=dict()
            special_tup_list[h]=dict()
            for p in self.all_non_null_action:

                this_term='action_match_h='+h+"_p="+p

                self.graph_non_null_act_2_dual[h][p]=lp_dual[this_term]
                prim_round=round(lp_primal[p],num_dec_keep)
                dual_round=round(lp_dual[this_term],num_dec_keep)
                my_tup=tuple([prim_round,dual_round])
                if my_tup not in special_tup_list[h]:
                    special_tup_list[h][my_tup]=[]
                special_tup_list[h][my_tup].append(p)
        print(special_tup_list)
        input('---')






for con in self.dict_con_name_2_eq:
    if con.startswith('action_match_h'):
        pattern = r'action_match_h=(\w+)_p=act_(\d+)_(\d+)'
        match = re.search(pattern, con)
        pattern_2 = r'action_match_h=(\w+)_p=(\S+)'
        match_2 = re.search(pattern_2, con)
        x=[]
        y=[]
        z=[]
        x1=[]
        f=[]
        if match:
            x, y, z = match.groups()
            x1, f = match_2.groups()

        else:
            print('con')
            print(con)
            input('error')
        if match_2:
            x1, f = match_2.groups()
        else:
            print('con')
            print(con)
            input('error2')
        print('con')
        print(con)
        print('x')
        print(x)
        print('y')
        print(y)
        print('z')
        print(z)
        print('x1')
        print(x1)
        print('f')
        print(f)
        input('--')
