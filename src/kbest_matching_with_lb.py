import networkx as nx
from networkx.algorithms import bipartite, shortest_paths
import torch
import dgl


class GedLowerBound(object):
    def __init__(self, g1, g2, lb_setting=0):
        self.g1 = g1
        self.g2 = g2
        self.lb_setting = lb_setting
        self.n1 = g1.num_nodes()
        self.n2 = g2.num_nodes()
        assert self.n1 <= self.n2
        if g1.ndata['f'].shape[1] == 1:
            self.has_node_label = False
        else:
            self.has_node_label = True

    @staticmethod
    def mc(sg1, sg2):
        # calculate the ged between two aligned graphs
        A = (sg1.adj() - sg2.adj()).coalesce().values()
        A_ged = (A ** 2).sum().item()
        F = sg1.ndata['f'] - sg2.ndata['f']
        F_ged = (F ** 2).sum().item()
        return (A_ged + F_ged) / 2.0

    def label_set(self, left_nodes, right_nodes):
        # sp.second_matching may be None.
        # In this case, calculating sp.ged2 makes right_nodes None.
        if right_nodes is None:
            return None

        # left_nodes could be [] when a full mapping is given
        partial_n = len(left_nodes)
        if partial_n == 0 and len(right_nodes) == self.n1:
            left_nodes = list(range(self.n1))
            partial_n = self.n1
        assert partial_n == len(right_nodes) and partial_n <= self.n1

        sub_g1 = self.g1.subgraph(left_nodes)
        sub_g2 = self.g2.subgraph(right_nodes)
        lb = self.mc(sub_g1, sub_g2)
        # print(lb)

        # num of edges
        m1 = self.g1.num_edges() - self.n1 - sub_g1.num_edges()  # + len(left_nodes)
        m2 = self.g2.num_edges() - self.n2 - sub_g2.num_edges()  # + len(right_nodes)
        lb += abs(m1 - m2) / 2.0
        # print(lb)

        # node label
        if (not self.has_node_label) or (partial_n == self.n1):  # this is a full mapping
            lb += (self.n2 - self.n1)
        else:
            f1 = dgl.remove_nodes(self.g1, left_nodes).ndata['f'].sum(dim=0)
            f2 = dgl.remove_nodes(self.g2, right_nodes).ndata['f'].sum(dim=0)
            intersect = torch.min(f1, f2)
            lb += (max(f1.sum().item(), f2.sum().item()) - intersect.sum().item())

        return lb


class Subspace(object):
    def __init__(self, G, matching, res, I=None, O=None):
        """
        G is the original graph (a complete networkx bipartite DiGraph with edge attribute "weight"),
        and self.G is a view (not copy) of G.
        In other words, self.G of all subspaces are the same object.

        We use I (edges used) and O (edges not used) to describe the solution subspace,
        When calculating the second best matching, we make a copy of G and edit it according to I and O.
        Therefore, self.G is also a constant.

        For each solution subspace, the best matching and its weight (res) is given for initialization.
        Then apply get_second_matching to calculate the 2nd best matching,
        by finding a minimum alternating cycle on the best matching in O(n^3).

        Only the best matching of the initial full space is calculated by KM algorithm.
        The best matching of the following subspaces comes from its father space's best or second best matching.
        In other words, subspace split merely depends on finding second best matching.
        """
        self.G = G
        self.best_matching = matching
        self.best_res = res
        self.I = set() if I is None else I  # the set of nodes whose matching can't change: use (u, v) -> add u into I
        self.O = [] if O is None else O  # the list of edges we can not use: do not use (u, v) -> append (u, v) into O
        self.get_second_matching()
        self.lb = None  # the lower bound ged of this subspace (depends on I)
        self.ged = None  # the ged of best matching
        self.ged2 = None  # the ged of 2nd-best matching

    def __repr__(self):
        best_res = "1st matching: {} {}".format(self.best_matching, self.best_res)
        second_res = "2nd matching: {} {}".format(self.second_matching, self.second_res)
        IO = "I: {}\tO: {}\tbranch edge: {}".format(self.I, self.O, self.branch_edge)
        return best_res + "\n" + second_res + "\n" + IO

    def get_second_matching(self):
        """
        Solve the second best matching based on the (1st) best one.
        Apply floyd and the single source bellman ford algorithm to find the minimum alternating cycle.

        Reverse the direction of edges in best matching and set their weights to the opposite.
        Direction: top->bottom  --> bottom->top
        Weight: negative --> positive

        For each edge (matching[u], u) in the best matching,
        the edge itself and the shortest path from u to matching[u] forms an alternating cycle.
        Recall that the edges in the best matching have positive weights, and the ones not in have negative weights.
        Therefore, the weight (sum) of an alternating cycle denotes
        the decrease of weight after applying it on the best matching,
        which is always non-negative.
        It is clear that we could apply the minimum weight alternating cycle on the best matching
        to get the 2nd best one.
        """
        G = self.G.copy()
        matching = self.best_matching.copy()
        n1 = len(matching)
        n = G.number_of_nodes()
        n2 = n - n1

        for (u, v) in self.O:
            G[u][v]["weight"] = float("inf")

        matched = [False] * n2
        for u in range(n1):
            v = matching[u]
            matched[v] = True
            v += n1
            w = -G[u][v]["weight"]  # become positive
            if u in self.I:
                w = float("inf")
            G.remove_edge(u, v)
            G.add_edge(v, u, weight=w)

        """
        Add a virtual node n.
        For each bottom node v, add an edge between v and n whose weight is 0:
        The direction is (n -> v) if v has been matched else (v -> n),
        i.e., unmatched bottom nodes -> n -> matched bottom nodes.
        """
        G.add_node(n, bipartite=0)
        for v in range(n2):
            if matched[v]:
                G.add_edge(n, n1 + v, weight=0.0)
            else:
                G.add_edge(n1 + v, n, weight=0.0)

        dis = shortest_paths.dense.floyd_warshall(G)
        cycle_min_weight = float("inf")
        cycle_min_uv = None
        for u in range(n1):
            if u in self.I:
                continue
            v = matching[u] + n1
            res = dis[u][v] + G[v][u]["weight"]
            if res < cycle_min_weight:
                cycle_min_weight = res
                cycle_min_uv = (u, v)

        if cycle_min_uv is None:
            # the second best matching does not exist in this subspace
            self.second_matching = None
            self.second_res = None
            self.branch_edge = None
            return

        u, v = cycle_min_uv
        length, path = shortest_paths.weighted.single_source_bellman_ford(G, source=u, target=v)
        assert abs(length + G[v][u]["weight"] - cycle_min_weight) < 1e-12

        # print("best matching:", matching)
        # print(cycle_min_weight, path)

        self.branch_edge = (u, v)  # an edge in the best matching but not in the second best one
        for i in range(0, len(path), 2):
            u, v = path[i], path[i + 1] - n1
            if u != n:
                matching[u] = v
        self.second_matching = matching
        self.second_res = self.best_res - cycle_min_weight

    def split(self):
        """
        Suppose the branching edge is (u, v), which is in self.best_matching but not in self.second_matching.
        Then current solution space sp is further split by using (u, v) or not.
        sp1: use (u,v), add u into I, sp1's best solution is the same as sp's.
        sp2: do not use (u, v), append (u, v) into O, sp2's best solution is sp's second best solution.

        We conduct an in-place update which makes sp becomes sp1, and return sp2 as a new subspace object.
        sp1's second_matching is calculated by calling self.get_second_matching(),
        sp2's second_matching is automatically calculated while object initialization.
        """
        u, v = self.branch_edge

        I = self.I.copy()
        self.I.add(u)
        O = self.O.copy()
        O.append((u, v))

        G = self.G  # needn't copy, all subspaces use the same G
        second_matching = self.second_matching
        self.second_matching = None
        second_res = self.second_res
        self.second_res = None

        self.get_second_matching()
        sp_new = Subspace(G, second_matching, second_res, I, O)
        return sp_new


class KBestMSolver(object):
    """
    Maintain a sequence of disjoint subspaces whose union is the full space.
    The best matching of the i-th subspace is exactly the i-th best matching of the full space.
    Specifically, self.subspaces[0].best_matching is the best matching,
    self.subspaces[1].best_matching is the second best matching,
    and self.subspaces[k-1].best_matching is the k-th best matching respectively.

    self.k is the length of self.subspaces. In another word, self.k-best matching have been solved.
    Apply self.expand_subspaces() to get the (self.k+1)-th best matching
    and maintain the subspaces structure accordingly.
    """

    def __init__(self, a, g1, g2, pre_ged=None):
        """
        Initially, self.subspaces[0] is the full space.
        """
        G, best_matching, res = self.from_tensor_to_nx(a)
        sp = Subspace(G, best_matching, res)

        self.lb = GedLowerBound(g1, g2)  # lower bound function
        self.lb_value = sp.lb = self.lb.label_set([], [])
        sp.ged = self.lb.label_set([], sp.best_matching)
        self.min_ged = sp.ged  # current best(minimum) solution, i.e., an upper bound
        sp.ged2 = self.lb.label_set([], sp.second_matching)
        self.set_min_ged(sp.ged2)  # Note that sp.ged2 may be None.

        self.subspaces = [sp]
        self.k = 1  # the length of self.subspaces
        self.expandable = True

        self.pre_ged = pre_ged

    def set_min_ged(self, ged):
        if ged is None:
            return
        if ged < self.min_ged:
            self.min_ged = ged

    ''' actually not useful
    def cal_min_lb(self):
    lb = float('inf')
    for sp in self.subspaces:
        if sp.second_matching is None:
            # This subspace only has one matching, sp.best_matching.
            lb = min(lb, sp.ged)
        else:
            lb = min(lb, sp.lb)
    return lb
    '''

    @staticmethod
    def from_tensor_to_nx(A):
        """
        A is a pytorch tensor whose shape is [n1, n2],
        denoting the weight matrix of a complete bipartite graph with n1+n2 nodes.
        Suppose the weights in A are non-negative.

        Construct a directed (top->bottom) networkx graph G based on A.
        0 ~ n1-1 are top nodes, and n1 ~ n1 + n2 -1 are bottom nodes.
        !!! The weights of G are set as the opposite of A.

        The maximum weight full matching is also solved for further subspaces construction.
        """
        n1, n2 = A.shape
        assert n1 <= n2
        top_nodes = range(n1)
        bottom_nodes = range(n1, n1 + n2)

        G = nx.DiGraph()
        G.add_nodes_from(top_nodes, bipartite=0)
        G.add_nodes_from(bottom_nodes, bipartite=1)
        A = A.tolist()
        for u in top_nodes:
            for v in bottom_nodes:
                G.add_edge(u, v, weight=-A[u][v - n1])
        # weight is set as -A[u][v] to get the maximum weight full matching

        matching = bipartite.matching.minimum_weight_full_matching(G, top_nodes)
        matching = [matching[u] - n1 for u in top_nodes]
        res = 0  # the weight sum of best matching
        for u in top_nodes:
            v = matching[u]
            res += A[u][v]

        '''
        for u in top_nodes:
            for v in bottom_nodes:
                G[u][v]['weight'] *= -1
        # restore weight to be positive
        '''

        return G, matching, res

    def expand_subspaces(self):
        """
        Find the subspace whose second matching is the largest, i.e., the (k+1)th best matching.
        Then split this subspace
        """
        max_res = -1
        max_spid = None

        for spid, sp in enumerate(self.subspaces):
            if sp.lb < self.min_ged and sp.second_res is not None and sp.second_res > max_res:
                #if (self.pre_ged is not None) and (sp.lb < self.pre_ged):
                max_res = sp.second_res
                max_spid = spid

        if max_spid is None:
            self.expandable = False
            return

        sp = self.subspaces[max_spid]
        sp_new = sp.split()
        self.subspaces.append(sp_new)
        self.k += 1

        sp_new.lb = sp.lb
        sp_new.ged = sp.ged2
        sp_new.ged2 = self.lb.label_set([], sp_new.second_matching)
        self.set_min_ged(sp_new.ged2)

        left_nodes = list(sp.I)
        right_nodes = [sp.best_matching[u] for u in left_nodes]
        sp.lb = self.lb.label_set(left_nodes, right_nodes)
        # sp.ged does not change since sp.best_matching does not change
        sp.ged2 = self.lb.label_set([], sp.second_matching)
        self.set_min_ged(sp.ged2)

    def get_matching(self, k):  # k starts form 1
        while self.k < k and self.expandable:
            self.expand_subspaces()

        if self.k < k:
            return None, None, None
        else:
            sp = self.subspaces[k-1]
            return sp.best_matching, sp.best_res, sp.ged

