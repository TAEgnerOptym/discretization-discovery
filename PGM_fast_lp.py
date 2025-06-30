import networkx as nx
from typing import Dict, DefaultDict, Set, List


def compute_pricing(self):
    self.all_actions=self.my_lower_bound_lp
    self.edge_weight_pricing_graph=dict()
    H=self.self.nov_h_uv_2_fg.keys()
    phi_by_h_p=dict()
    E_by_h=dict()
    for h in H:
        phi_by_h_p[h]=dict()
        E_by_h[h]=dict()
    for my_act in self.all_actions:
        trm1=self.dict_var_name_2_obj[my_act]
        trm2=self.dual_con_contrib[my_act]
        trm3=sum(self.dual_big_edge_contrib_by_h[my_act])
        
        for h in self.dual_big_edge_contrib_by_h[my_act]:
            phi_by_h_p[h][my_act]=(-trm1-trm2)*(self.dual_big_edge_contrib_by_h[my_act]/trm3)
    L=self.my_lower_bound_object
    did_find_path=dict()
    p_to_cost=DefaultDict(float)
    all_p_found=set([])
    for h in H:
        source=L.graph_node_2_agg_node[h][self.h_2_source_id[h]]
        sink=L.graph_node_2_agg_node[h][self.h_2_sink_id[h]]
        E_list=[]
        for fg in self.h_fg_2_q[h]:
            f=fg[0]
            g=fg[1]
            q=self.h_fg_2_q[h][fg]
            p=q[0]
            weight=phi_by_h_p[h][p]+0.00001
            E_list.append([f,g,weight,p])
        out_dict_h=analyze_path_or_cycle(E_list, source, sink)
        if out_dict_h['cost']<-0.001 or out_dict_h['type']=='negative_cycle':
            #for p in out_dict_h['p_terms']:
            #    if p_to_cost[p]=min(p_to_cost[p])
            all_p_found=all_p_found+set(out_dict_h['p_terms'])
    for act_u_v_term in all_p_found:
        act_u_v=act_u_v_term[0]
        var_name_compress=self.var_name_map[act_u_v]
        var=self.var_dict[var_name_compress]
        self.var_add_novel.append(var)
    self.var_add_novel_2=[]

def analyze_path_or_cycle(E_list, source, sink):
    G = nx.DiGraph()
    for f, g, weight, p in E_list:
        G.add_edge(f, g, weight=weight, p=p)

    try:
        # Attempt shortest path using Bellman-Ford (handles negative weights)
        path = nx.bellman_ford_path(G, source, sink, weight='weight')
        cost = nx.path_weight(G, path, weight='weight')
        p_terms = {G[u][v]['p'] for u, v in zip(path, path[1:])}
        return {
            "type": "shortest_path",
            "cost": cost,
            "path": [(u, v, G[u][v]['weight'], G[u][v]['p']) for u, v in zip(path, path[1:])],
            "p_terms": p_terms
        }
    except nx.NetworkXUnbounded:
        # Negative cycle detected
        for cycle in nx.simple_cycles(G):
            # Check if it's truly negative
            cycle_edges = list(zip(cycle, cycle[1:] + [cycle[0]]))
            weight_sum = sum(G[u][v]['weight'] for u, v in cycle_edges)
            if weight_sum < 0:
                return {
                    "type": "negative_cycle",
                    "cost": weight_sum,
                    "cycle": [(u, v, G[u][v]['weight'], G[u][v]['p']) for u, v in cycle_edges],
                    "p_terms": {G[u][v]['p'] for u, v in cycle_edges}
                }
        return {
            "type": "negative_cycle_detected_but_not_extracted",
            "cost": None,
            "cycle": [],
            "p_terms": set()
        }
    except nx.NetworkXNoPath:
        return {
            "type": "no_path",
            "cost": None,
            "path": [],
            "p_terms": set()
        }
