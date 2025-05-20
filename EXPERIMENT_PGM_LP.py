import gurobipy as gp

class PGM_LP_SOLVER:

    def __init__(self,full_prob,graph_node_2_agg_node,X_):
                    
        self.full_prob=full_prob
        self.graph_node_2_agg_node=self.full_prob.graph_node_2_agg_node
        self.actions_ignore=self.full_prob.actions_ignore
        
        self.setup_RMP()
        self.graph_lp_list=dict()
        for h in self.full_prob.graph_names:
            self.edges_2_activities[h]=self.get_graph_lp_h_mapping(h)
        self.run_alg()

    def run_alg(self):

        lp_last=np.inf
        while(True):
            self.run_RMP_given_X()
            if self.lp_last>self.LP_current+self.jy_opt[['min_inc_compress']]:
                self.all_X_forbidden=self.allX-set(self.primal_X_current_active)
                self.forget_unused_X()
            tot_objective=0
            for h in self.full_prob.graph_names:
                #self.set_get_objective(h)
                objective,xTermsInPaths=self.solve_pricing_shortest_path_h_lp(h)
                tot_objective=tot_objective+objective
                if objective<-self.full_prob.jy_opt['epsilon']:
                    self.all_X_forbidden=self.all_X_forbidden-set([xTermsInPaths])

            if tot_objective>-self.full_prob.jy_opt['epsilon']:
                break
    def 