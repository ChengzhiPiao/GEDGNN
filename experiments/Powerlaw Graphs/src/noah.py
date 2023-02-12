# Implementations of A* Algorithm including A*-Beamsearch and A*-Pathlength, with different lower bounds.
# Reference:
# "Fast Suboptimal Algorithms for the Computation of Graph Edit Distance"
# "Efficient Graph Similarity Search Over Large Graph Databases"
# "Efficient Graph Edit Distance Computation and Verification via Anchor-aware Lower Bound Estimation"
# "Speeding Up Graph Edit Distance Computation through Fast Bipartite Matching"(VJ Algorithm)
# "Approximate graph edit distance computation by means of bipartite graph matching"(Hungarian Algorithm)
# Author: Lei Yang

import networkx as nx
import torch
import time
import pickle
import numpy as np
from munkres import Munkres

# Calculate the cost of edit path
def cost_edit_path(edit_path, u, v, lower_bound):
    cost = 0

    source_nodes = []
    target_nodes = []
    nodes_dict = {}
    for operation in edit_path:
        if operation[0] == None:
            cost += 1
            target_nodes.append(operation[1])
        elif operation[1] == None:
            cost += 1
            source_nodes.append(operation[0])
        else:
            if u.nodes[operation[0]]['label'] != v.nodes[operation[1]]['label']:
                cost += 1
            source_nodes.append(operation[0])
            target_nodes.append(operation[1])
        nodes_dict[operation[0]] = operation[1]

    edge_source = u.subgraph(source_nodes).edges()
    edge_target = v.subgraph(target_nodes).edges()

    sum = 0
    for edge in list(edge_source):
        (p, q) = (nodes_dict[edge[0]], nodes_dict[edge[1]])
        if (p, q) in edge_target:
            sum += 1
    cost = cost + len(edge_source) + len(edge_target) - 2 * sum

    if len(lower_bound) == 3 and lower_bound[2] == 'a':
        # Anchor
        anchor_cost = 0
        cross_edge_source = []
        cross_edge_target = []
        cross_edge_source_tmp = set(u.edges(source_nodes))
        for edge in cross_edge_source_tmp:
            if edge[0] not in source_nodes or edge[1] not in source_nodes:
                cross_edge_source.append(edge)
        cross_edge_target_tmp = set(v.edges(target_nodes))
        for edge in cross_edge_target_tmp:
            if edge[0] not in target_nodes or edge[1] not in target_nodes:
                cross_edge_target.append(edge)

        for edge in cross_edge_source:
            (p, q) = (nodes_dict[edge[0]], edge[1])
            if (p, q) in cross_edge_target:
                anchor_cost += 1

        return cost + anchor_cost
    else:
        return cost


# Check unprocessed nodes in graph u and v
def check_unprocessed(u, v, path):
    processed_u = []
    processed_v = []

    for operation in path:
        if operation[0] != None:
            processed_u.append(operation[0])

        if operation[1] != None:
            processed_v.append(operation[1])
    # print(processed_u, processed_v)
    unprocessed_u = set(u.nodes()) - set(processed_u)
    unprocessed_v = set(v.nodes()) - set(processed_v)
    return list(unprocessed_u), list(unprocessed_v)


def list_unprocessed_label(unprocessed_node, u):
    unprocessed_label = []
    for node in unprocessed_node:
        unprocessed_label.append(u.nodes[node]['label'])
    unprocessed_label.sort()
    return unprocessed_label


def transfer_to_torch(unprocessed_u, unprocessed_v, u, v):
    """
    Transferring the data to torch and creating a hash table with the indices, features and target.
    :param data: Data dictionary.
    :return new_data: Dictionary of Torch Tensors.
    """
    # global_labels_file = open('AIDS.pkl','rb')
    # global_labels = pickle.load(global_labels_file)
    # superLabel = str(len(global_labels)-1)
    superLabel = 0

    new_data = dict()
    g1 = u.subgraph(unprocessed_u)
    g2 = v.subgraph(unprocessed_v)
    reorder_u = {val: index for index, val in enumerate(unprocessed_u)}
    # print(reorder_u)
    g1_tmp = nx.Graph()
    for (val, index) in reorder_u.items():
        g1_tmp.add_node(index, label=g1.nodes[val]['label'])
    count = g1.number_of_nodes()
    g1_tmp.add_node(count, label=superLabel)
    for (i, j) in list(g1.edges()):
        g1_tmp.add_edge(reorder_u[i], reorder_u[j])
    for node in reorder_u.values():
        g1_tmp.add_edge(count, node)
    g1 = g1_tmp

    reorder_v = {val: index for index, val in enumerate(unprocessed_v)}
    g2_tmp = nx.Graph()
    for (val, index) in reorder_v.items():
        g2_tmp.add_node(index, label=g2.nodes[val]['label'])
    count = g2.number_of_nodes()
    g2_tmp.add_node(count, label=superLabel)
    for (i, j) in list(g2.edges()):
        g2_tmp.add_edge(reorder_v[i], reorder_v[j])
    for node in reorder_v.values():
        g2_tmp.add_edge(count, node)
    g2 = g2_tmp

    #new_data["hb"] = max(g1.number_of_nodes(), g2.number_of_nodes()) + max(g1.number_of_edges(), g2.number_of_edges())
    new_data["hb"] = g1.number_of_edges() // 5

    edges_1 = [[edge[0], edge[1]] for edge in g1.edges()] + [[edge[1], edge[0]] for edge in g1.edges()] + [[i, i] for i
                                                                                                           in
                                                                                                           g1.nodes()]
    edges_2 = [[edge[0], edge[1]] for edge in g2.edges()] + [[edge[1], edge[0]] for edge in g2.edges()] + [[i, i] for i
                                                                                                           in
                                                                                                           g2.nodes()]
    edges_1 = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long)
    edges_2 = torch.from_numpy(np.array(edges_2, dtype=np.int64).T).type(torch.long)
    label_1 = [g1.nodes[node]['label'] for node in g1.nodes()]
    label_2 = [g2.nodes[node]['label'] for node in g2.nodes()]
    # print(label_1)
    # print(label_2)
    if -1 in label_1:
        number_of_labels = 1
        features_1 = torch.tensor([[2.0] for i in g1.nodes()])
        features_2 = torch.tensor([[2.0] for i in g2.nodes()])
    else:
        # AIDS
        # number_of_labels = 29
        # PL25:5; PL50, PL100: 10
        number_of_labels = 5
        features_1 = torch.tensor(
            [[1.0 if l == l_index else 0. for l_index in range(number_of_labels)] for l in label_1])
        features_2 = torch.tensor(
            [[1.0 if l == l_index else 0. for l_index in range(number_of_labels)] for l in label_2])
    if torch.cuda.is_available():
        new_data["edge_index_1"] = edges_1.cuda()
        new_data["edge_index_2"] = edges_2.cuda()
        new_data["features_1"] = features_1.cuda()
        new_data["features_2"] = features_2.cuda()
    else:
        new_data["edge_index_1"] = edges_1
        new_data["edge_index_2"] = edges_2
        new_data["features_1"] = features_1
        new_data["features_2"] = features_2

    '''
    ged = 4 # randomly
    normalized_ged = ged / (0.5 * (len(label_1) + len(label_2)))
    new_data["target"] = torch.from_numpy(np.exp(-normalized_ged).reshape(1, 1)).view(-1).float()
    '''
    return new_data


def star_cost(p, q):
    cost = 0
    if p == None:
        cost += 2 * len(q) - 1
        return cost
    if q == None:
        cost += 2 * len(p) - 1
        return cost
    if p[0] != q[0]:
        cost += 1
    if len(p) > 1 and len(q) > 1:
        p[1:].sort()
        q[1:].sort()
        i = 1
        j = 1
        cross_node = 0
        while (i < len(p) and j < len(q)):
            if p[i] == q[j]:
                cross_node += 1
                i += 1
                j += 1
            elif p[i] < q[j]:
                i += 1
            else:
                j += 1
        cost = cost + max(len(p), len(q)) - 1 - cross_node
    cost += abs(len(q) - len(p))
    return cost


def unprocessed_cost(model, unprocessed_u_set, unprocessed_v_set, u, v):
    global lower_bound
    # print (lower_bound)
    if lower_bound == 'heuristic':
        # heuristic
        cost_set = []
        for i in range(len(unprocessed_u_set)):
            unprocessed_u = unprocessed_u_set[i]
            unprocessed_v = unprocessed_v_set[i]
            inter_node = set(unprocessed_u).intersection(set(unprocessed_v))
            cost = max(len(unprocessed_u), len(unprocessed_v)) - len(inter_node)
            cost_set.append(cost)
        return cost_set
    elif lower_bound[0:2] == 'LS':
        # LS
        cost_set = []
        for i in range(len(unprocessed_u_set)):
            unprocessed_u = unprocessed_u_set[i]
            unprocessed_v = unprocessed_v_set[i]
            cross_node = 0
            u_label = list_unprocessed_label(unprocessed_u, u)
            v_label = list_unprocessed_label(unprocessed_v, v)

            i = 0
            j = 0
            while (i < len(u_label) and j < len(v_label)):
                if u_label[i] == v_label[j]:
                    cross_node += 1
                    i += 1
                    j += 1
                elif u_label[i] < v_label[j]:
                    i += 1
                else:
                    j += 1

            node_cost = max(len(unprocessed_u), len(unprocessed_v)) - cross_node
            edge_u = u.subgraph(unprocessed_u).edges()
            edge_v = v.subgraph(unprocessed_v).edges()
            inter_edge = set(edge_u).intersection(set(edge_v))
            edge_cost = max(len(edge_u), len(edge_v)) - len(inter_edge)
            cost = node_cost + edge_cost
            cost_set.append(cost)
        return cost_set
    elif lower_bound == 'Noah':  # and min(len(unprocessed_u),len(unprocessed_v)) > 1: # add terminate condition
        # model.eval()
        cost_set = []
        for i in range(len(unprocessed_u_set)):
            unprocessed_u = unprocessed_u_set[i]
            unprocessed_v = unprocessed_v_set[i]
            if unprocessed_u and unprocessed_v:
                data = transfer_to_torch(unprocessed_u, unprocessed_v, u, v)
                prediction = model(data)[0]
                # cost = prediction * (max(len(unprocessed_u),len(unprocessed_v)) + max(len(u.subgraph(unprocessed_u).edges()), len(v.subgraph(unprocessed_v).edges())))
                cost = prediction.item() * data["hb"]
                cost -= abs(len(unprocessed_u) - len(unprocessed_v))
                if cost < 3:
                    lower_bound = 'BM'
                cost_set.append(int(cost))
            else:
                cost = max(len(unprocessed_u), len(unprocessed_v))
                cost_set.append(cost)
        # print (cost_set)
        return cost_set

    elif lower_bound == 'BM':
        # BM
        cost_set = []
        for i in range(len(unprocessed_u_set)):
            unprocessed_u = unprocessed_u_set[i]
            unprocessed_v = unprocessed_v_set[i]
            cost = 0
            u_label = list_unprocessed_label(unprocessed_u, u)
            v_label = list_unprocessed_label(unprocessed_v, v)
            i = 0
            j = 0
            while (i < len(u_label) and j < len(v_label)):
                if u_label[i] == v_label[j] and u.edges(unprocessed_u[i]) == v.edges(unprocessed_v[j]):
                    u_label.pop(i)
                    v_label.pop(j)
                    i += 1
                    j += 1
                elif u_label[i] < v_label[j]:
                    i += 1
                else:
                    j += 1
            i = 0
            j = 0
            while (i < len(u_label) and j < len(v_label)):
                if u_label[i] == v_label[j]:
                    cost += 0.5
                    u_label.pop(i)
                    v_label.pop(j)
                    i += 1
                    j += 1
                elif u_label[i] < v_label[j]:
                    i += 1
                else:
                    j += 1
            cost = cost + max(len(u_label), len(v_label))
            cost_set.append(cost)
        return cost_set
    else:
        # SM
        cost_set = []
        for i in range(len(unprocessed_u_set)):
            unprocessed_u = unprocessed_u_set[i]
            unprocessed_v = unprocessed_v_set[i]
            stars_u = []
            temp_u = u.subgraph(unprocessed_u)
            for node in unprocessed_u:
                node_list = []
                node_list.append(node)
                for k in temp_u.neighbors(node):
                    node_list.append(k)
                stars_u.append(node_list)

            stars_v = []
            temp_v = v.subgraph(unprocessed_v)
            for node in unprocessed_v:
                node_list = []
                node_list.append(node)
                for k in temp_v.neighbors(node):
                    node_list.append(k)
                stars_v.append(node_list)

            max_degree = 0
            for i in stars_u:
                if len(i) > max_degree:
                    max_degree = len(i)

            for i in stars_v:
                if len(i) > max_degree:
                    max_degree = len(i)
            # Initial cost matrix
            if len(stars_u) > len(stars_v):
                for i in range(len(stars_u) - len(stars_v)):
                    stars_v.append(None)
            if len(stars_u) < len(stars_v):
                for i in range(len(stars_v) - len(stars_u)):
                    stars_u.append(None)
            cost_matrix = []
            for star1 in stars_u:
                cost_tmp = []
                for star2 in stars_v:
                    cost_tmp.append(star_cost(star1, star2))
                cost_matrix.append(cost_tmp)
            if cost_matrix == []:
                cost_set.append(0)
            else:
                m = Munkres()
                indexes = m.compute(cost_matrix)
                cost = 0
                for row, column in indexes:
                    value = cost_matrix[row][column]
                    cost += value
                cost = cost / max(4, max_degree)
                cost_set.append(cost)
        return cost_set


def graph_edit_distance(model, u, v, in_lower_bound, beam_size, start_node=None):
    # Partial edit path
    global lower_bound
    lower_bound = in_lower_bound
    open_set = []
    cost_open_set = []
    partial_cost_set = []
    path_idx_list = []
    time_count = 0.0
    # For each node w in V2, insert the substitution {u1 -> w} into OPEN
    if start_node == None or start_node not in list(u.nodes()):
        u1 = list(u.nodes())[0]  # randomly access a node
    else:
        u1 = start_node
    call_count = 0
    unprocessed_u_set = []
    unprocessed_v_set = []
    for w in list(v.nodes()):
        edit_path = []
        edit_path.append((u1, w))
        unprocessed_u, unprocessed_v = check_unprocessed(u, v, edit_path)
        new_cost = cost_edit_path(edit_path, u, v, lower_bound)
        cost_list = [new_cost]
        unprocessed_u_set.append(unprocessed_u)
        unprocessed_v_set.append(unprocessed_v)
        # new_cost += unprocessed_cost(unprocessed_u, unprocessed_v, u, v)
        call_count += 1
        open_set.append(edit_path)
        partial_cost_set.append(cost_list)
    unprocessed_cost_set = unprocessed_cost(model, unprocessed_u_set, unprocessed_v_set, u, v)
    start = time.time()
    for i in range(len(unprocessed_cost_set)):
        new_cost = unprocessed_cost_set[i] + partial_cost_set[i][0]
        cost_open_set.append(new_cost)
    end = time.time()
    time_count = time_count + end - start

    # Insert the deletion {u1 -> none} into OPEN
    edit_path = []
    edit_path.append((u1, None))
    unprocessed_u, unprocessed_v = check_unprocessed(u, v, edit_path)
    new_cost = cost_edit_path(edit_path, u, v, lower_bound)
    cost_list = [new_cost]
    start = time.time()
    new_cost_set = unprocessed_cost(model, [unprocessed_u], [unprocessed_v], u, v)
    new_cost += new_cost_set[0]
    end = time.time()
    time_count = time_count + end - start
    call_count += 1
    open_set.append(edit_path)
    cost_open_set.append(new_cost)
    partial_cost_set.append(cost_list)

    while cost_open_set:
        if beam_size:
            # BeamSearch
            tmp_path_set = []
            tmp_cost_set = []
            tmp_partial_cost_set = []
            if len(cost_open_set) > beam_size:
                zipped = zip(open_set, cost_open_set, partial_cost_set)
                sort_zipped = sorted(zipped, key=lambda x: x[1])
                result = zip(*sort_zipped)
                open_set, cost_open_set, partial_cost_set = [list(x)[0:beam_size] for x in result]
                # for i in range(beam_size):
                #     path_idx = cost_open_set.index(min(cost_open_set))
                #     if idx_flag == 0:
                #         path_idx_list.append(path_idx)
                #         idx_flag = 1
                #     print (cost_open_set, path_idx)
                #     tmp_path_set.append(open_set.pop(path_idx))
                #     tmp_cost_set.append(cost_open_set.pop(path_idx))
                #     tmp_partial_cost_set.append(partial_cost_set.pop(path_idx))

                # open_set = tmp_path_set
                # cost_open_set = tmp_cost_set
                # partial_cost_set = tmp_partial_cost_set

        # Retrieve minimum-cost partial edit path pmin from OPEN
        # print (cost_open_set)
        path_idx = cost_open_set.index(min(cost_open_set))
        path_idx_list.append(path_idx)
        min_path = open_set.pop(path_idx)
        cost = cost_open_set.pop(path_idx)
        cost_list = partial_cost_set.pop(path_idx)

        # print(len(open_set))
        # Check p_min is a complete edit path
        unprocessed_u, unprocessed_v = check_unprocessed(u, v, min_path)

        # Return if p_min is a complete edit path
        if not unprocessed_u and not unprocessed_v:
            return min_path, cost, cost_list, call_count, time_count, path_idx_list

        else:
            if unprocessed_u:
                u_next = unprocessed_u.pop()
                unprocessed_u_set = []
                unprocessed_v_set = []
                for v_next in unprocessed_v:
                    new_path = min_path.copy()
                    new_path.append((u_next, v_next))
                    unprocessed_u, unprocessed_v = check_unprocessed(u, v, new_path)
                    new_cost = cost_edit_path(new_path, u, v, lower_bound)
                    new_cost_list = cost_list.copy()
                    new_cost_list.append(new_cost)
                    unprocessed_u_set.append(unprocessed_u)
                    unprocessed_v_set.append(unprocessed_v)
                    # new_cost += unprocessed_cost(unprocessed_u, unprocessed_v, u, v)
                    call_count += 1
                    open_set.append(new_path)
                    # cost_open_set.append(new_cost)
                    partial_cost_set.append(new_cost_list)
                start = time.time()
                new_cost_set = unprocessed_cost(model, unprocessed_u_set, unprocessed_v_set, u, v)
                for i in range(len(new_cost_set)):
                    new_cost = new_cost_set[i] + partial_cost_set[i - len(new_cost_set)][-1]
                    cost_open_set.append(new_cost)
                end = time.time()
                time_count = time_count + end - start

                new_path = new_path = min_path.copy()
                new_path.append((u_next, None))
                unprocessed_u, unprocessed_v = check_unprocessed(u, v, new_path)
                new_cost = cost_edit_path(new_path, u, v, lower_bound)
                new_cost_list = cost_list.copy()
                new_cost_list.append(new_cost)
                start = time.time()
                new_cost_set = unprocessed_cost(model, [unprocessed_u], [unprocessed_v], u, v)
                new_cost += new_cost_set[0]
                end = time.time()
                time_count = time_count + end - start
                call_count += 1
                open_set.append(new_path)
                cost_open_set.append(new_cost)
                partial_cost_set.append(new_cost_list)


            else:
                # All nodes in u have been processed, all nodes in v should be Added.
                unprocessed_u_set = []
                unprocessed_v_set = []
                for v_next in unprocessed_v:
                    new_path = min_path.copy()
                    new_path.append((None, v_next))
                    new_cost = cost_edit_path(new_path, u, v, lower_bound)
                    new_cost_list = cost_list.copy()
                    new_cost_list.append(new_cost)
                    unprocessed_u_set.append(unprocessed_u)
                    unprocessed_v_set.append(unprocessed_v)
                    call_count += 1
                    open_set.append(new_path)
                    # cost_open_set.append(new_cost)
                    partial_cost_set.append(new_cost_list)
                start = time.time()
                new_cost_set = unprocessed_cost(model, unprocessed_u_set, unprocessed_v_set, u, v)
                for i in range(len(new_cost_set)):
                    new_cost = new_cost_set[i] + partial_cost_set[i - len(new_cost_set)][-1]
                    cost_open_set.append(new_cost)
                end = time.time()
                time_count = time_count + end - start
    return None, None, None, None, None, None