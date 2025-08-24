import math
import networkx as nx
from collections import defaultdict

def compute_path_based_rc(self) -> None:
    """
    Computes self.path_based_rc[act] = reduced_cost(act) + sum_h best_by_h[h][act],
    where best_by_h[h][act] = min_{(i,j): hij_2_P[h][(i,j)]=act} [ d_src[i] + d_sink[j] + w_ij ],
    with w_ij = graph_act_2_weight[h][act] on edge (i,j).
    Assumes no negative cycles.
    """

    # --- Guards ----------------------------------------------------------------
    if self.null_action in self.all_non_null_action:
        raise ValueError("null_action must not be in all_non_null_action")

    # --- 1) Base reduced costs per action -------------------------------------
    act_2_red_cost = {}
    for act in self.all_non_null_action:
        if act in self.full_prob.delta_name_2_ub and self.full_prob.delta_name_2_ub[act] < 1e-3:
            act_2_red_cost[act] = math.inf
        else:
            act_2_red_cost[act] = self.action_2_cost[act]

    # Subtract dual contributions: (act, con) -> coeff
    # NOTE: self.action_con_2_contrib is a dict, not a function.
    for (act, con), coeff in self.action_con_2_contrib.items():
        # If act is infinite (tight UB), keep it infinite.
        if act_2_red_cost.get(act, math.inf) is not math.inf:
            act_2_red_cost[act] -= self.lp_dual_solution[con] * coeff

    # --- 2) Build per-graph action weights ------------------------------------
    # For each h: weight for 'null_action' is 0; non-null actions map from duals.
    BIGPOS = 9.9999999999e11
    EPS = 1e-5

    graph_act_2_weight = {}
    for h in self.graph_name_2_nodes:
        w = {self.null_action: 0.0}
        graph_act_2_weight[h] = w

    for act in self.all_non_null_action:
        # If reduced cost is inf (disallowed/very tight), give huge edge weights.
        if act_2_red_cost[act] == math.inf:
            for h in self.graph_name_2_nodes:
                graph_act_2_weight[h][act] = BIGPOS
        else:
            for h in self.graph_name_2_nodes:
                con = f"action_match_h={h}_p={act}"
                # Add the dual once to the action's reduced cost, and set per-graph weight.
                act_2_red_cost[act] += self.lp_dual_solution[con]
                # Edge weight used in shortest-paths (slightly nudged with EPS).
                graph_act_2_weight[h][act] = -self.lp_dual_solution[con] + EPS

    # --- 3) For each graph h, compute best edge-based marginal per action ------
    # best_by_h[h][act] = min over edges labeled act of  d_src[i] + d_sink[j] + w_ij
    best_by_h = {}

    for h in self.graph_name_2_nodes:
        mapping = self.hij_2_P[h]                 # dict: (i,j) -> act
        act_w = graph_act_2_weight[h]             # dict: act -> weight

        # Build weighted DiGraph quickly
        # Edges: (i, j, weight=act_w[mapping[(i,j)]])
        weighted_edges = []
        for (i, j), act in mapping.items():
            w = act_w.get(act)
            if w is not None:
                weighted_edges.append((i, j, w))
            else:
                input('error here')
        G = nx.DiGraph()
        G.add_weighted_edges_from(weighted_edges)

        source = self.h2SourceId[h]
        sink = self.h2SinkId[h]

        # Distances FROM source (Bellman–Ford, supports negative edges)
        try:
            d_src = nx.single_source_bellman_ford_path_length(G, source, weight="weight")
        except nx.NetworkXUnbounded:
            raise ValueError(f"Negative-weight cycle reachable from source in graph h={h}")

        # Distances TO sink: run from sink on the reversed graph
        Grev = G.reverse(copy=False)
        try:
            d_to_sink_rev = nx.single_source_bellman_ford_path_length(Grev, sink, weight="weight")
        except nx.NetworkXUnbounded:
            raise ValueError(f"Negative-weight cycle can reach sink in graph h={h} (on reversed graph)")

        # Edge marginal cost: d_src[i] + d_sink[j] + w_ij
        edge_min_marg = {}
        for (i, j), act in mapping.items():
            if not G.has_edge(i, j):
                continue  # edge weight wasn't available
            w_ij = G[i][j]["weight"]
            val = d_src.get(i, math.inf) + d_to_sink_rev.get(j, math.inf) + w_ij
            edge_min_marg[(i, j)] = val

        # Aggregate min per action
        best = defaultdict(lambda: math.inf)
        for (i, j), act in mapping.items():
            v = edge_min_marg.get((i, j))
            if v==None:
                input('error here 2')
            if v is not None and v < best[act]:
                best[act] = v
        best_by_h[h] = best

    # --- 4) Final path-based reduced costs per action --------------------------
    self.path_based_rc = {}
    for act in self.all_non_null_action:
        total = act_2_red_cost[act]
        if total == math.inf:
            self.path_based_rc[act] = math.inf
            continue
        # Sum the best contribution across all h (if none, it stays +inf and total -> +inf)
        for h in self.graph_name_2_nodes:
            total += best_by_h[h][act]
        self.path_based_rc[act] = total
